from typing import Any
import litellm
from pydantic import BaseModel, Field


def get_answer_sentence(sentences: list[str], answer_start: int) -> int:
    """
    Get the sentence ID that contains the answer span based on the answer start index.

    Args:
        sentences (list[str]): List of sentences in the context.
        answer_start (int): The starting index of the answer in the joined context.

    Returns:
        int: The ID (index) of the sentence containing the answer span.
    """
    # Track cumulative character position in the joined text
    current_pos = 0

    for i, sentence in enumerate(sentences):
        # Check if answer_start falls within this sentence
        if current_pos <= answer_start < current_pos + len(sentence):
            return i
        # Update position (no space separator for Japanese text)
        current_pos += len(sentence)

    # If answer_start is beyond all sentences, return the last sentence index
    return len(sentences) - 1


def split_text_with_periods(text: str) -> list[str]:
    """
    文字列を句点でスプリットして、句点を保持したまま分割する。
    分割した結果を空文字でjoinすると元の文字列に戻る。

    Args:
        text (str): 分割対象の文字列

    Returns:
        list[str]: 句点を含む文に分割されたリスト

    Examples:
        >>> split_text_with_periods("これは文です。これも文です。最後の文。")
        ['これは文です。', 'これも文です。', '最後の文。']

        >>> split_text_with_periods("句点なし")
        ['句点なし']

        >>> text = "文章。。別の文章。"
        >>> "".join(split_text_with_periods(text)) == text
        True
    """
    if not text:
        return []

    # 句点で分割
    parts = text.split("。")

    # 最後の要素が空文字列でない場合（文末に句点がない場合）
    if parts[-1]:
        # 句点を付けずにそのまま残す
        sentences = [part + "。" for part in parts[:-1]] + [parts[-1]]
    else:
        # 文末に句点がある場合、最後の空文字列を除去
        sentences = [part + "。" for part in parts[:-1]]

    # 空文字列は保持（元の文字列を復元するため）
    # ただし、最初が空文字列の場合のみ除去（"。始まり"の場合）
    if sentences and sentences[0] == "。":
        return sentences

    return sentences


def add_answer_sentence(example: dict[str, Any]) -> dict[str, Any]:
    """
    Add the sentence containing the answer to the example.

    Args:
        example (dict[str, Any]): The example dictionary containing 'context' and 'answer_start'.

    Returns:
        dict[str, Any]: Updated example with 'answer_sentence' added.
    """
    context = example["context"]
    answer_start = example["answer"]["answer_start"][0]
    
    sentences = split_text_with_periods(context)
    answer_sentence_id = get_answer_sentence(sentences, answer_start)
    example["answer_sentence"] = sentences[answer_sentence_id]
    return example

def extract_answer_text_by_llm(questions: list[str], contexts: list[str],*,model="gpt-5-nano") -> list[str]:
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
    for question, context in zip(questions, contexts):
        messages=[
            {"role": "system", "content": "コンテキストのうち、Questionに対する回答を含む文章を、句点含めて、そのまま抜き出してください。"},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ]
        messages_list.append(messages)
    response = litellm.batch_completion(
        messages=messages_list,
        model=model,
        response_format=Answer
    )
    answers = []
    for response in response:
        content = response['choices'][0]['message']['content']
        answer = Answer.model_validate_json(content)
        answers.append(answer.sentence)
    return answers
