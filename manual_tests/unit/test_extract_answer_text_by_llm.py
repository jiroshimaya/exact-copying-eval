"""Manual tests for extract_answer_text_by_llm function.

これらのテストは実際のLLM APIを呼び出すため、手動で実行する必要があります。
APIキーが必要であり、課金が発生する可能性があります。

実行方法:
    uv run pytest manual_tests/unit/test_extract_answer_text_by_llm.py -v
"""

import pytest
from exact_copying_eval.core.evaluate import extract_answer_text_by_llm


class TestExtractAnswerTextByLlm:
    """extract_answer_text_by_llm関数のテストクラス."""

    def test_正常系(self):
        """長い文脈から正しく回答が抽出できることを確認."""
        # Arrange
        questions = ["機械学習とは何ですか？"]
        contexts = [
            "人工知能（AI）は現代技術の重要な分野です。"
            "その中でも機械学習は、コンピュータがデータから自動的にパターンを学習する技術です。"
            "深層学習やニューラルネットワークなど、様々な手法があります。"
            "これらの技術は画像認識、自然言語処理、音声認識など幅広い分野で活用されています。"
            "最近では大規模言語モデルも注目を集めています。"
        ]
        
        # Act
        answers = extract_answer_text_by_llm(questions, contexts)
        
        # Assert
        assert len(answers) == 1
        assert "機械学習" in answers[0]
        assert "コンピュータ" in answers[0]
        assert "データから自動的にパターンを学習する技術" in answers[0]
        print(f"実際の回答: {answers[0]}")

