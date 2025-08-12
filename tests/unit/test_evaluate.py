"""Test script for evaluate.py functions."""

from unittest.mock import patch

from exact_copying_eval.core.evaluate import (
    calculate_metrics,
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


class TestCalculateMetrics:
    """Test class for calculate_metrics function."""

    def test_正常系_完全一致(self):
        """すべて完全一致する場合の指標計算を確認"""
        questions = ["Q1", "Q2", "Q3"]
        answers = ["Answer 1", "Answer 2", "Answer 3"]
        contexts = [
            "Context 1\nAnswer 1\nMore",
            "Context 2\nAnswer 2\nMore",
            "Context 3\nAnswer 3\nMore",
        ]
        generated_answers = ["Answer 1", "Answer 2", "Answer 3"]

        result = calculate_metrics(questions, answers, contexts, generated_answers)

        assert result["summary"]["total"] == 3
        assert result["summary"]["exact_match_count"] == 3
        assert result["summary"]["exact_match_accuracy"] == 1.0
        assert result["summary"]["answer_inclusion_count"] == 3
        assert result["summary"]["answer_inclusion_accuracy"] == 1.0
        assert result["summary"]["context_inclusion_count"] == 3
        assert result["summary"]["context_inclusion_accuracy"] == 1.0
        assert result["summary"]["avg_edit_distance"] == 0.0
        assert len(result["detail"]["detailed_results"]) == 3
        assert len(result["detail"]["wrong_details"]) == 0

    def test_正常系_部分一致のケース(self):
        """answer_inclusionとcontext_inclusionの違いを確認"""
        questions = ["Q1", "Q2", "Q3"]
        answers = ["期待される回答文です", "期待される別の回答", "第三の回答"]
        contexts = [
            "これは長いコンテキストです。期待される回答文です。追加情報もあります。",
            "別のコンテキスト。他の情報。",
            "第三のコンテキスト。関連する情報。",
        ]
        generated_answers = ["期待される", "別のコンテキスト", "全く違う回答"]

        result = calculate_metrics(questions, answers, contexts, generated_answers)

        assert result["summary"]["total"] == 3
        assert result["summary"]["exact_match_count"] == 0
        assert (
            result["summary"]["answer_inclusion_count"] == 1
        )  # "期待される" in "期待される回答文です"
        assert result["summary"]["context_inclusion_count"] == 2  # 1番目と2番目

        # 詳細結果の確認
        detailed_results = result["detail"]["detailed_results"]
        assert detailed_results[0]["answer_inclusion"] is True
        assert detailed_results[0]["context_inclusion"] is True
        assert detailed_results[1]["answer_inclusion"] is False
        assert detailed_results[1]["context_inclusion"] is True
        assert detailed_results[2]["answer_inclusion"] is False
        assert detailed_results[2]["context_inclusion"] is False

    def test_正常系_編集距離計算(self):
        """編集距離が正しく計算されることを確認"""
        questions = ["Q1", "Q2"]
        answers = ["abcdef", "xyz"]  # 6文字, 3文字
        contexts = ["Context1\nabcdef\nMore", "Context2\nxyz\nMore"]
        generated_answers = ["abcdXX", "xyz"]  # 編集距離2, 0

        result = calculate_metrics(questions, answers, contexts, generated_answers)

        detailed_results = result["detail"]["detailed_results"]
        # 1番目: 編集距離2/6 = 0.333...
        assert abs(detailed_results[0]["edit_distance"] - 2 / 6) < 0.001
        # 2番目: 編集距離0/3 = 0.0
        assert detailed_results[1]["edit_distance"] == 0.0

        # 平均編集距離
        expected_avg = (2 / 6 + 0.0) / 2
        assert abs(result["summary"]["avg_edit_distance"] - expected_avg) < 0.001

    def test_正常系_空のリスト(self):
        """空のリストが正しく処理されることを確認"""
        questions = []
        answers = []
        contexts = []
        generated_answers = []

        result = calculate_metrics(questions, answers, contexts, generated_answers)

        assert result["summary"]["total"] == 0
        assert result["summary"]["exact_match_count"] == 0
        assert result["summary"]["exact_match_accuracy"] == 0.0
        assert result["summary"]["answer_inclusion_count"] == 0
        assert result["summary"]["answer_inclusion_accuracy"] == 0.0
        assert result["summary"]["context_inclusion_count"] == 0
        assert result["summary"]["context_inclusion_accuracy"] == 0.0
        assert result["summary"]["avg_edit_distance"] == 0.0
        assert result["summary"]["avg_answer_length"] == 0.0
        assert len(result["detail"]["detailed_results"]) == 0
        assert len(result["detail"]["wrong_details"]) == 0

    def test_正常系_平均回答長計算(self):
        """平均回答長が正しく計算されることを確認"""
        questions = ["Q1", "Q2", "Q3"]
        answers = ["abc", "defgh", "ijklmnop"]  # 3, 5, 8文字
        contexts = ["C1\nabc\nMore", "C2\ndefgh\nMore", "C3\nijklmnop\nMore"]
        generated_answers = ["abc", "defgh", "ijklmnop"]

        result = calculate_metrics(questions, answers, contexts, generated_answers)

        expected_avg_length = (3 + 5 + 8) / 3
        assert result["summary"]["avg_answer_length"] == expected_avg_length


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
                context=f"Context {i}\nAnswer {i}\nMore context",
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
        assert result["summary"]["answer_inclusion_accuracy"] == 1.0
        assert result["summary"]["answer_inclusion_count"] == 3
        assert result["summary"]["context_inclusion_accuracy"] == 1.0
        assert result["summary"]["context_inclusion_count"] == 3

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
                question="Q1",
                context="Context 1\nAnswer 1\nMore context",
                expected_answer="Answer 1",
            ),
            EvaluationItem(
                question="Q2",
                context="Context 2\nAnswer 2\nMore context",
                expected_answer="Answer 2",
            ),
            EvaluationItem(
                question="Q3",
                context="Context 3\nAnswer 3\nMore context",
                expected_answer="Answer 3",
            ),
        ]
        mock_dataset = EvaluationDataset(items=items, metadata={})
        mock_load_dataset.return_value = mock_dataset

        # 一部間違った回答を設定
        mock_extract.return_value = ["Answer 1", "Completely wrong", "Answer 3"]

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
        assert result["detail"]["wrong_details"][0]["actual"] == "Completely wrong"

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
                context="Context 1\nThis is a long answer\nMore context",
                expected_answer="This is a long answer",
            ),
            EvaluationItem(
                question="Q2",
                context="Context 2\nAnother answer\nMore context",
                expected_answer="Another answer",
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
        assert result["summary"]["answer_inclusion_count"] == 2  # 両方とも部分一致
        assert (
            result["summary"]["context_inclusion_count"] == 2
        )  # 両方ともコンテキストに含まれる

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
                question="Q1",
                context="Context 1\nAnswer 1\nMore context",
                expected_answer="Answer 1",
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

    @patch("exact_copying_eval.core.evaluate.load_evaluation_dataset")
    @patch("exact_copying_eval.core.evaluate.extract_answer_text_by_llm")
    def test_正常系_answer_inclusion_と_context_inclusion_の違い(
        self, mock_extract, mock_load_dataset
    ):
        """answer_inclusionとcontext_inclusionの違いが正しく評価されること"""
        from exact_copying_eval.core.create_dataset import (
            EvaluationDataset,
            EvaluationItem,
        )

        # モックデータセット設定
        items = [
            EvaluationItem(
                question="Q1",
                context="これは長いコンテキストです。期待される回答文です。追加情報もあります。",
                expected_answer="期待される回答文です。",
            ),
            EvaluationItem(
                question="Q2",
                context="別のコンテキスト。他の情報。",
                expected_answer="期待される別の回答",
            ),
            EvaluationItem(
                question="Q3",
                context="第三のコンテキスト。関連する情報。",
                expected_answer="第三の回答",
            ),
        ]
        mock_dataset = EvaluationDataset(items=items, metadata={})
        mock_load_dataset.return_value = mock_dataset

        # テストケース:
        # 1. "期待される" - 期待回答の一部だが、コンテキストにも含まれる
        # 2. "別のコンテキスト" - コンテキストには含まれるが、期待回答には含まれない
        # 3. "全く違う回答" - どちらにも含まれない
        mock_extract.return_value = ["期待される", "別のコンテキスト", "全く違う回答"]

        result = evaluate(
            dataset_file="test_dataset.json",
            model="test-model",
            batch_size=3,
            prompt_type="qa",
        )

        # 検証
        assert result["summary"]["exact_match_count"] == 0  # 完全一致なし
        assert (
            result["summary"]["answer_inclusion_count"] == 1
        )  # 1番目のみ期待回答に含まれる
        assert (
            result["summary"]["context_inclusion_count"] == 2
        )  # 1番目と2番目はコンテキストに含まれる

        # 詳細結果の確認
        detailed_results = result["detail"]["detailed_results"]

        # 1番目のケース: answer_inclusion=True, context_inclusion=True
        assert detailed_results[0]["answer_inclusion"] is True
        assert detailed_results[0]["context_inclusion"] is True

        # 2番目のケース: answer_inclusion=False, context_inclusion=True
        assert detailed_results[1]["answer_inclusion"] is False
        assert detailed_results[1]["context_inclusion"] is True

        # 3番目のケース: answer_inclusion=False, context_inclusion=False
        assert detailed_results[2]["answer_inclusion"] is False
        assert detailed_results[2]["context_inclusion"] is False

    @patch("exact_copying_eval.core.evaluate.load_evaluation_dataset")
    @patch("exact_copying_eval.core.evaluate.extract_answer_text_by_llm")
    def test_正常系_edit_distance_計算(self, mock_extract, mock_load_dataset):
        """編集距離が正しく計算されること"""
        from exact_copying_eval.core.create_dataset import (
            EvaluationDataset,
            EvaluationItem,
        )

        # モックデータセット設定
        items = [
            EvaluationItem(
                question="Q1",
                context="コンテキスト1\nabcdef\nその他情報",
                expected_answer="abcdef",  # 6文字
            ),
            EvaluationItem(
                question="Q2",
                context="コンテキスト2\nxyz\nその他情報",
                expected_answer="xyz",  # 3文字
            ),
        ]
        mock_dataset = EvaluationDataset(items=items, metadata={})
        mock_load_dataset.return_value = mock_dataset

        # 編集距離のテスト
        # "abcdef" -> "abcdXX" (編集距離2, 正規化: 2/6 = 0.333...)
        # "xyz" -> "xyz" (編集距離0, 正規化: 0/3 = 0.0)
        mock_extract.return_value = ["abcdXX", "xyz"]

        result = evaluate(
            dataset_file="test_dataset.json",
            model="test-model",
            batch_size=2,
            prompt_type="qa",
        )

        detailed_results = result["detail"]["detailed_results"]

        # 1番目の編集距離確認（許容範囲で比較）
        assert abs(detailed_results[0]["edit_distance"] - 2 / 6) < 0.001

        # 2番目の編集距離確認
        assert detailed_results[1]["edit_distance"] == 0.0

        # 平均編集距離確認
        expected_avg = (2 / 6 + 0.0) / 2
        assert abs(result["summary"]["avg_edit_distance"] - expected_avg) < 0.001

    @patch("exact_copying_eval.core.evaluate.load_evaluation_dataset")
    @patch("exact_copying_eval.core.evaluate.extract_answer_text_by_llm")
    def test_正常系_空のデータセット(self, mock_extract, mock_load_dataset):
        """空のデータセットが正しく処理されること"""
        from exact_copying_eval.core.create_dataset import EvaluationDataset

        # 空のデータセット
        mock_dataset = EvaluationDataset(items=[], metadata={})
        mock_load_dataset.return_value = mock_dataset
        mock_extract.return_value = []

        result = evaluate(
            dataset_file="empty_dataset.json",
            model="test-model",
            batch_size=2,
            prompt_type="qa",
        )

        assert result["summary"]["total"] == 0
        assert result["summary"]["exact_match_count"] == 0
        assert result["summary"]["exact_match_accuracy"] == 0.0
        assert result["summary"]["answer_inclusion_count"] == 0
        assert result["summary"]["answer_inclusion_accuracy"] == 0.0
        assert result["summary"]["context_inclusion_count"] == 0
        assert result["summary"]["context_inclusion_accuracy"] == 0.0
        assert result["summary"]["avg_edit_distance"] == 0.0
        assert result["summary"]["avg_answer_length"] == 0.0
        assert len(result["detail"]["detailed_results"]) == 0
        assert len(result["detail"]["wrong_details"]) == 0
