"""Test script for evaluate.py functions."""

from unittest.mock import patch

from exact_copying_eval.core.evaluate import (
    evaluate,
    extract_answer_text_by_llm,
    get_exact_copying_qa_prompt,
    get_exact_copying_simple_prompt,
)


class TestGetExactCopyingQaPrompt:
    """Test class for get_exact_copying_qa_prompt function."""

    def test_正常系_QAプロンプト生成(self):
        """QA用のプロンプトが正しく生成されること"""
        question = "テスト質問"
        context = "テストコンテキスト"

        result = get_exact_copying_qa_prompt(question, context)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "コンテキストの各要素は改行で区切られています" in result[0]["content"]
        assert result[1]["role"] == "user"
        assert f"Context: {context}" in result[1]["content"]
        assert f"Question: {question}" in result[1]["content"]


class TestGetExactCopyingSimplePrompt:
    """Test class for get_exact_copying_simple_prompt function."""

    def test_正常系_Simpleプロンプト生成(self):
        """Simple用のプロンプトが正しく生成されること"""
        question = "テスト質問"  # この関数では使用されない
        context = "テストコンテキスト"

        result = get_exact_copying_simple_prompt(question, context)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "2行目を、過不足なくそのまま抜き出してください" in result[0]["content"]
        assert result[1]["role"] == "user"
        assert result[1]["content"] == context


class TestExtractAnswerTextByLlm:
    """Test class for extract_answer_text_by_llm function."""

    @patch("exact_copying_eval.core.evaluate.litellm.batch_completion")
    def test_正常系_単一質問と回答(self, mock_batch_completion):
        """単一の質問に対して正しく回答が抽出されること"""
        # モックレスポンス設定
        mock_response = [
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"sentence": "This is the answer sentence."}'
                        }
                    }
                ]
            }
        ]
        mock_batch_completion.return_value = mock_response

        questions = ["What is the answer?"]
        contexts = ["Context line 1\nThis is the answer sentence.\nContext line 3"]

        result = extract_answer_text_by_llm(
            questions, contexts, model="gpt-5-nano", prompt_type="qa"
        )

        assert result == ["This is the answer sentence."]
        mock_batch_completion.assert_called_once()

    @patch("exact_copying_eval.core.evaluate.litellm.batch_completion")
    def test_正常系_複数質問と回答(self, mock_batch_completion):
        """複数の質問に対して正しく回答が抽出されること"""
        # モックレスポンス設定
        mock_responses = [
            {"choices": [{"message": {"content": '{"sentence": "Answer 1"}'}}]},
            {"choices": [{"message": {"content": '{"sentence": "Answer 2"}'}}]},
        ]
        mock_batch_completion.return_value = mock_responses

        questions = ["Question 1?", "Question 2?"]
        contexts = [
            "Context 1\nAnswer 1\nMore context",
            "Context 2\nAnswer 2\nMore context",
        ]

        result = extract_answer_text_by_llm(
            questions, contexts, model="test-model", prompt_type="qa"
        )

        assert result == ["Answer 1", "Answer 2"]

    @patch("exact_copying_eval.core.evaluate.litellm.batch_completion")
    def test_正常系_モデル指定(self, mock_batch_completion):
        """指定したモデルが正しく使用されること"""
        mock_response = [
            {"choices": [{"message": {"content": '{"sentence": "Test answer"}'}}]}
        ]
        mock_batch_completion.return_value = mock_response

        questions = ["Test question?"]
        contexts = ["Test context\nTest answer\nMore context"]

        extract_answer_text_by_llm(
            questions, contexts, model="custom-model", prompt_type="qa"
        )

        # モデル引数が正しく渡されているか確認
        call_args = mock_batch_completion.call_args
        assert call_args[1]["model"] == "custom-model"

    @patch("exact_copying_eval.core.evaluate.litellm.batch_completion")
    def test_正常系_prompt_type_simple(self, mock_batch_completion):
        """prompt_type='simple'が正しく動作すること"""
        mock_response = [
            {"choices": [{"message": {"content": '{"sentence": "Test answer"}'}}]}
        ]
        mock_batch_completion.return_value = mock_response

        questions = ["Test question?"]
        contexts = ["Test context\nTest answer\nMore context"]

        result = extract_answer_text_by_llm(
            questions, contexts, model="test-model", prompt_type="simple"
        )

        assert result == ["Test answer"]
        # prompt_typeがsimpleの場合の処理が呼ばれているかは実装の詳細で検証

    @patch("exact_copying_eval.core.evaluate.litellm.batch_completion")
    def test_正常系_prompt_type_qa(self, mock_batch_completion):
        """prompt_type='qa'が正しく動作すること"""
        mock_response = [
            {"choices": [{"message": {"content": '{"sentence": "Test answer"}'}}]}
        ]
        mock_batch_completion.return_value = mock_response

        questions = ["Test question?"]
        contexts = ["Test context\nTest answer\nMore context"]

        result = extract_answer_text_by_llm(
            questions, contexts, model="test-model", prompt_type="qa"
        )

        assert result == ["Test answer"]


class TestEvaluate:
    """Test class for evaluate function."""

    @patch("exact_copying_eval.core.evaluate.load_evaluation_dataset")
    @patch("exact_copying_eval.core.evaluate.extract_answer_text_by_llm")
    def test_正常系_基本的な評価実行(self, mock_extract, mock_load_dataset):
        """基本的な評価が正しく実行されること"""
        from exact_copying_eval.core.create_dataset import (
            EvaluationDataset,
            EvaluationItem,
        )

        # モックデータセット設定
        items = [
            EvaluationItem(
                question=f"Question {i}",
                context=f"Context {i}",
                expected_answer=f"Answer {i}",
            )
            for i in range(3)
        ]
        mock_dataset = EvaluationDataset(items=items, metadata={})
        mock_load_dataset.return_value = mock_dataset

        # 完全一致の回答を設定
        mock_extract.return_value = ["Answer 0", "Answer 1", "Answer 2"]

        result = evaluate(
            dataset_file="test_dataset.json",
            model="test-model",
            batch_size=2,
            prompt_type="qa",
        )

        assert "summary" in result
        assert "detail" in result
        assert result["summary"]["total"] == 3
        assert result["summary"]["model"] == "test-model"
        assert result["summary"]["exact_match_accuracy"] == 1.0  # 全て正解
        assert result["summary"]["exact_match_count"] == 3
        assert result["summary"]["inclusion_accuracy"] == 1.0
        assert result["summary"]["inclusion_count"] == 3

    @patch("exact_copying_eval.core.evaluate.load_evaluation_dataset")
    @patch("exact_copying_eval.core.evaluate.extract_answer_text_by_llm")
    def test_正常系_部分的正解の場合(self, mock_extract, mock_load_dataset):
        """部分的に正解した場合の評価結果が正しいこと"""
        from exact_copying_eval.core.create_dataset import (
            EvaluationDataset,
            EvaluationItem,
        )

        # モックデータセット設定
        items = [
            EvaluationItem(
                question="Q1", context="Context 1", expected_answer="Answer 1"
            ),
            EvaluationItem(
                question="Q2", context="Context 2", expected_answer="Answer 2"
            ),
            EvaluationItem(
                question="Q3", context="Context 3", expected_answer="Answer 3"
            ),
        ]
        mock_dataset = EvaluationDataset(items=items, metadata={})
        mock_load_dataset.return_value = mock_dataset

        # 一部間違った回答を設定
        mock_extract.return_value = ["Answer 1", "Wrong answer", "Answer 3"]

        result = evaluate(
            dataset_file="test_dataset.json",
            model="test-model",
            batch_size=2,
            prompt_type="qa",
        )

        assert result["summary"]["exact_match_count"] == 2
        assert result["summary"]["exact_match_accuracy"] == 2 / 3
        assert len(result["detail"]["wrong_details"]) == 1
        assert result["detail"]["wrong_details"][0]["index"] == 1
        assert result["detail"]["wrong_details"][0]["expected"] == "Answer 2"
        assert result["detail"]["wrong_details"][0]["actual"] == "Wrong answer"

    @patch("exact_copying_eval.core.evaluate.load_evaluation_dataset")
    @patch("exact_copying_eval.core.evaluate.extract_answer_text_by_llm")
    def test_正常系_inclusion_テスト(self, mock_extract, mock_load_dataset):
        """inclusion（部分一致）のテストが正しく動作すること"""
        from exact_copying_eval.core.create_dataset import (
            EvaluationDataset,
            EvaluationItem,
        )

        # モックデータセット設定
        items = [
            EvaluationItem(
                question="Q1",
                context="Context 1",
                expected_answer="This is a long answer",
            ),
            EvaluationItem(
                question="Q2", context="Context 2", expected_answer="Another answer"
            ),
        ]
        mock_dataset = EvaluationDataset(items=items, metadata={})
        mock_load_dataset.return_value = mock_dataset

        # 部分一致の回答を設定
        mock_extract.return_value = ["long answer", "Another answer"]

        result = evaluate(
            dataset_file="test_dataset.json",
            model="test-model",
            batch_size=2,
            prompt_type="qa",
        )

        assert result["summary"]["exact_match_count"] == 1  # 2番目のみ完全一致
        assert result["summary"]["inclusion_count"] == 2  # 両方とも部分一致

    @patch("exact_copying_eval.core.evaluate.load_evaluation_dataset")
    @patch("exact_copying_eval.core.evaluate.extract_answer_text_by_llm")
    def test_正常系_prompt_type_simple(self, mock_extract, mock_load_dataset):
        """prompt_type='simple'が正しく動作すること"""
        from exact_copying_eval.core.create_dataset import (
            EvaluationDataset,
            EvaluationItem,
        )

        # モックデータセット設定
        items = [
            EvaluationItem(
                question="Q1", context="Context 1", expected_answer="Answer 1"
            ),
        ]
        mock_dataset = EvaluationDataset(items=items, metadata={})
        mock_load_dataset.return_value = mock_dataset

        mock_extract.return_value = ["Answer 1"]

        result = evaluate(
            dataset_file="test_dataset.json",
            model="test-model",
            batch_size=2,
            prompt_type="simple",
        )

        assert result["summary"]["total"] == 1
        # extract_answer_text_by_llmがprompt_type='simple'で呼ばれているかチェック
        mock_extract.assert_called()
        call_args = mock_extract.call_args
        assert call_args[1]["prompt_type"] == "simple"
