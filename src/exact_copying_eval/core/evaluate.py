import logging
from typing import Any, Literal

import litellm
import tqdm
from pydantic import BaseModel, Field

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
        question (str): The question to be answered.
        context (str): The context from which to extract the answer.

    Returns:
        str: The extracted answer text.
    """

    class Answer(BaseModel):
        """Answer Information"""

        sentence: str = Field(..., description="The sentence including the answer.")

    messages_list = []
    for question, context in zip(questions, contexts, strict=False):
        if prompt_type == "qa":
            messages = get_exact_copying_qa_prompt(question, context)
        else:
            messages = get_exact_copying_simple_prompt(question, context)
        messages_list.append(messages)
    response_list = litellm.batch_completion(
        messages=messages_list, model=model, response_format=Answer
    )
    answers = []
    for response in response_list:
        content = response["choices"][0]["message"]["content"]
        answer = Answer.model_validate_json(content)
        answers.append(answer.sentence)
    return answers


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
        "Starting evaluation",
        dataset_file=dataset_file,
        model=model,
        prompt_type=prompt_type,
    )

    # Load evaluation dataset from JSON
    dataset = load_evaluation_dataset(dataset_file)

    questions = [item.question for item in dataset.items]
    contexts = [item.context for item in dataset.items]
    answers = [item.expected_answer for item in dataset.items]

    logger.info("Dataset loaded", num_items=len(dataset.items))

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

    correct_count = sum(1 for a, b in zip(answers, generated_answers, strict=False) if a == b)
    wrong_details = []
    for i, (a, b) in enumerate(zip(answers, generated_answers, strict=False)):
        if a != b:
            wrong_details.append(
                {
                    "index": i,
                    "question": questions[i],
                    "context": contexts[i],
                    "expected": a,
                    "actual": b,
                }
            )

    summary = {
        "total": len(questions),
        "correct_count": correct_count,
        "accuracy": correct_count / len(questions) if questions else 0.0,
        "model": model,
    }

    logger.info(
        "Evaluation completed",
        total=summary["total"],
        correct=summary["correct_count"],
        accuracy=summary["accuracy"],
    )

    return {"summary": summary, "wrong_details": wrong_details}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Evaluate the model on prepared dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
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
        "--output", type=str, default="", help="Output file for results."
    )
    args = parser.parse_args()

    output_path = args.output
    if not args.output:
        from pathlib import Path

        dataset_path = Path(args.dataset)
        dataset_name = dataset_path.stem
        output_path = f"result_{args.model}_{dataset_name}_{args.prompt_type}.json"

    result = evaluate(
        dataset_file=args.dataset,
        model=args.model,
        batch_size=args.batch_size,
        prompt_type=args.prompt_type,
    )
    print("評価結果:")
    print(result["summary"]["correct_count"] / result["summary"]["total"])

    from pathlib import Path
    output_file = Path(output_path)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
