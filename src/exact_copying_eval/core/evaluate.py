import datetime
import logging
from typing import Any, Literal

import editdistance
import litellm
import tqdm
from pydantic import BaseModel

from exact_copying_eval.core.create_dataset import load_evaluation_dataset

logger = logging.getLogger(__name__)


def get_exact_copying_qa_prompt(question: str, context: str) -> list[dict[str, str]]:
    messages = []
    system_prompt = (
        "コンテキストの各要素は改行で区切られています。"
        "コンテキストのうち、Questionに対する回答を含む要素（行）を、"
        "過不足なくそのまま抜き出してください。"
    )
    messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    )
    return messages


def get_exact_copying_simple_prompt(
    question: str, context: str
) -> list[dict[str, str]]:
    messages = []
    system_prompt = (
        "次に与えるコンテキスト（1行1文章）のうち2行目を、"
        "過不足なくそのまま抜き出してください。"
    )
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": context})
    return messages


def extract_answer_text_by_llm(
    questions: list[str],
    contexts: list[str],
    *,
    model="gpt-5-nano",
    prompt_type: Literal["qa", "simple"],
) -> list[str]:
    """
    Extract the answer text from the context based on the question using an LLM.

    Args:
        questions (list[str]): The questions to be answered.
        contexts (list[str]): The contexts from which to extract the answers.
        model (str): The model to use for extraction.
        prompt_type (Literal["qa", "simple"]): The type of prompt to use.

    Returns:
        list[str]: The extracted answer texts.
    """

    class ExtractedText(BaseModel):
        """Extracted Text Information"""

        sentence: str

    messages_list = []
    for question, context in zip(questions, contexts, strict=False):
        if prompt_type == "qa":
            messages = get_exact_copying_qa_prompt(question, context)
        else:
            messages = get_exact_copying_simple_prompt(question, context)
        messages_list.append(messages)
    response_list = litellm.batch_completion(
        messages=messages_list, model=model, response_format=ExtractedText
    )
    answers = []
    for response in response_list:
        content = response["choices"][0]["message"]["content"]
        answer = ExtractedText.model_validate_json(content)
        answers.append(answer.sentence)
    return answers


def calculate_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[str],
    generated_answers: list[str],
) -> dict[str, Any]:
    """指標を計算する関数.

    Args:
        questions (list[str]): 質問のリスト
        answers (list[str]): 期待される回答のリスト
        contexts (list[str]): コンテキストのリスト
        generated_answers (list[str]): 生成された回答のリスト

    Returns:
        dict[str, Any]: 計算された指標と詳細結果
    """
    # Calculate multiple metrics
    exact_match_count = sum(
        1 for a, b in zip(answers, generated_answers, strict=False) if a == b
    )
    answer_inclusion_count = sum(
        1 for a, b in zip(answers, generated_answers, strict=False) if b in a
    )
    context_inclusion_count = sum(
        1 for c, b in zip(contexts, generated_answers, strict=False) if b in c
    )

    # Calculate average edit distance
    edit_distances = [
        editdistance.eval(a, b) / len(a)
        for a, b in zip(answers, generated_answers, strict=False)
    ]
    avg_edit_distance = (
        sum(edit_distances) / len(edit_distances) if edit_distances else 0.0
    )

    # Calculate average answer length
    answer_lengths = [len(answer) for answer in answers]
    avg_answer_length = (
        sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0
    )

    # Prepare detailed results for wrong answers
    wrong_details = []
    detailed_results = []

    for i, (expected, actual, context) in enumerate(
        zip(answers, generated_answers, contexts, strict=False)
    ):
        is_exact_match = expected == actual
        is_answer_inclusion = actual in expected
        is_context_inclusion = actual in context
        edit_dist = editdistance.eval(expected, actual) / len(expected)

        detailed_result = {
            "index": i,
            "question": questions[i],
            "context": contexts[i],
            "expected": expected,
            "actual": actual,
            "exact_match": is_exact_match,
            "answer_inclusion": is_answer_inclusion,
            "context_inclusion": is_context_inclusion,
            "edit_distance": edit_dist,
        }
        detailed_results.append(detailed_result)

        if not is_exact_match:
            wrong_details.append(detailed_result)

    summary = {
        "total": len(questions),
        "exact_match_count": exact_match_count,
        "exact_match_accuracy": exact_match_count / len(questions)
        if questions
        else 0.0,
        "answer_inclusion_count": answer_inclusion_count,
        "answer_inclusion_accuracy": answer_inclusion_count / len(questions)
        if questions
        else 0.0,
        "context_inclusion_count": context_inclusion_count,
        "context_inclusion_accuracy": context_inclusion_count / len(questions)
        if questions
        else 0.0,
        "avg_edit_distance": avg_edit_distance,
        "avg_answer_length": avg_answer_length,
    }

    detail = {
        "detailed_results": detailed_results,
        "wrong_details": wrong_details,
    }

    return {
        "summary": summary,
        "detail": detail,
    }


def evaluate(
    *,
    dataset_file: str,
    model: str = "gpt-5-nano",
    batch_size: int = 10,
    prompt_type: Literal["qa", "simple"] = "qa",
) -> dict[str, Any]:
    """評価を実行する関数.

    Args:
        dataset_file (str): 評価用データセットのJSONファイルパス
        model (str): 使用するモデル名
        batch_size (int): バッチサイズ
        prompt_type (Literal["qa", "simple"]): プロンプトタイプ

    Returns:
        dict[str, Any]: 評価結果
    """
    logger.info(
        "Starting evaluation - dataset_file=%s, model=%s, prompt_type=%s",
        dataset_file,
        model,
        prompt_type,
    )

    # Load evaluation dataset from JSON
    dataset = load_evaluation_dataset(dataset_file)

    questions = [item.question for item in dataset.items]
    contexts = [item.context for item in dataset.items]
    answers = [item.expected_answer for item in dataset.items]

    logger.info("Dataset loaded: %d items", len(dataset.items))

    generated_answers = []
    for i in tqdm.tqdm(range(0, len(dataset.items), batch_size)):
        batch_questions = questions[i : i + batch_size]
        batch_contexts = contexts[i : i + batch_size]
        if not batch_questions or not batch_contexts:
            continue
        generated_answers.extend(
            extract_answer_text_by_llm(
                batch_questions, batch_contexts, model=model, prompt_type=prompt_type
            )
        )

    # Calculate metrics using the separated function
    metrics_result = calculate_metrics(questions, answers, contexts, generated_answers)

    # Add additional metadata to summary
    metrics_result["summary"]["model"] = model
    metrics_result["summary"]["prompt_type"] = prompt_type
    metrics_result["summary"]["dataset_file"] = dataset_file
    metrics_result["summary"]["timestamp"] = datetime.datetime.now().isoformat()

    logger.info(
        "Evaluation completed - total=%d, exact_match_count=%d, "
        "exact_match_accuracy=%.4f, answer_inclusion_count=%d, "
        "answer_inclusion_accuracy=%.4f, context_inclusion_count=%d, "
        "context_inclusion_accuracy=%.4f, avg_edit_distance=%.4f, "
        "avg_answer_length=%.2f",
        metrics_result["summary"]["total"],
        metrics_result["summary"]["exact_match_count"],
        metrics_result["summary"]["exact_match_accuracy"],
        metrics_result["summary"]["answer_inclusion_count"],
        metrics_result["summary"]["answer_inclusion_accuracy"],
        metrics_result["summary"]["context_inclusion_count"],
        metrics_result["summary"]["context_inclusion_accuracy"],
        metrics_result["summary"]["avg_edit_distance"],
        metrics_result["summary"]["avg_answer_length"],
    )

    return metrics_result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Evaluate the model on prepared dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5-nano", help="Model to use for evaluation."
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for processing."
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["qa", "simple"],
        default="qa",
        help="プロンプトのタイプを指定します（qa または simple）",
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="Output file path for results."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for results."
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="既存の結果ファイルを読み込み、メトリクスのみを再計算して上書きします。",
    )
    args = parser.parse_args()

    # 引数の検証
    if args.output_file and args.output_dir:
        parser.error("--output_file と --output_dir の両方を指定することはできません。")

    if args.update and not args.output_file:
        parser.error(
            "--update フラグを使用する場合は --output_file を指定してください。"
        )

    if not args.update and not args.dataset:
        parser.error("--update フラグを使用しない場合は --dataset を指定してください。")

    # 出力パスの決定
    if args.output_file:
        output_path = args.output_file
    else:
        from pathlib import Path

        dataset_path = Path(args.dataset)
        dataset_name = dataset_path.stem
        default_filename = f"result_{args.model}_{dataset_name}_{args.prompt_type}.json"

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / default_filename
        else:
            output_path = default_filename

    if args.update:
        # updateフラグが指定された場合：既存ファイルを読み込み、メトリクスのみ再計算
        from pathlib import Path

        output_file = Path(args.output_file)
        if not output_file.exists():
            parser.error(f"更新対象のファイルが見つかりません: {args.output_file}")

        # 既存の結果ファイルを読み込み
        with output_file.open("r", encoding="utf-8") as f:
            existing_result = json.load(f)

        # 詳細結果から元データを抽出
        if (
            "detail" not in existing_result
            or "detailed_results" not in existing_result["detail"]
        ):
            parser.error("更新対象のファイルに詳細結果が含まれていません。")

        detailed_results = existing_result["detail"]["detailed_results"]
        questions = [item["question"] for item in detailed_results]
        contexts = [item["context"] for item in detailed_results]
        answers = [item["expected"] for item in detailed_results]
        generated_answers = [item["actual"] for item in detailed_results]

        print(f"既存ファイルから {len(detailed_results)} 件のデータを読み込みました")

        # メトリクスを再計算
        metrics_result = calculate_metrics(
            questions, answers, contexts, generated_answers
        )

        # 既存の結果から必要な情報を維持
        if "summary" in existing_result:
            if "model" in existing_result["summary"]:
                metrics_result["summary"]["model"] = existing_result["summary"]["model"]
            if "prompt_type" in existing_result["summary"]:
                metrics_result["summary"]["prompt_type"] = existing_result["summary"][
                    "prompt_type"
                ]
            if "dataset_file" in existing_result["summary"]:
                metrics_result["summary"]["dataset_file"] = existing_result["summary"][
                    "dataset_file"
                ]

        # タイムスタンプを更新
        metrics_result["summary"]["timestamp"] = datetime.datetime.now().isoformat()
        metrics_result["summary"]["updated"] = True

        result = metrics_result
    else:
        # 通常の評価処理
        result = evaluate(
            dataset_file=args.dataset,
            model=args.model,
            batch_size=args.batch_size,
            prompt_type=args.prompt_type,
        )
    print("評価結果:")
    print(f"Exact Match Accuracy: {result['summary']['exact_match_accuracy']:.4f}")
    print(
        "Answer Inclusion Accuracy: "
        f"{result['summary']['answer_inclusion_accuracy']:.4f}"
    )
    print(
        "Context Inclusion Accuracy: "
        f"{result['summary']['context_inclusion_accuracy']:.4f}"
    )
    print(f"Average Edit Distance: {result['summary']['avg_edit_distance']:.2f}")
    print(f"Average Answer Length: {result['summary']['avg_answer_length']:.2f}")

    from pathlib import Path

    output_file = Path(output_path)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
