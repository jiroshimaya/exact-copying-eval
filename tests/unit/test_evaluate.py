"""Test script for evaluate.py functions."""

from unittest.mock import Mock, patch

import pytest
from datasets import Dataset

from exact_copying_eval.core.evaluate import (
    create_context,
    evaluate,
    extract_answer_text_by_llm,
    get_dummy_indices,
    get_exact_copying_qa_prompt,
    get_exact_copying_simple_prompt,
    load_jsquad,
    remove_linebreaks,
)


class TestLoadJsquad:
    """Test class for load_jsquad function."""

    @patch("exact_copying_eval.core.evaluate.load_dataset")
    def test_正常系_データセット読み込み成功(self, mock_load_dataset):
        """JSQuADデータセットの読み込みが成功すること"""
        # モックデータセットを作成
        mock_dataset = Mock(spec=Dataset)
        mock_load_dataset.return_value = mock_dataset

        result = load_jsquad()

        assert result == mock_dataset
        mock_load_dataset.assert_called_once_with(
            "sbintuitions/JSQuAD", split="validation"
        )

    @patch("exact_copying_eval.core.evaluate.load_dataset")
    def test_異常系_データセット型不正(self, mock_load_dataset):
        """読み込んだデータセットがDataset型でない場合、ValueErrorが発生すること"""
        # 不正な型をモックとして設定
        mock_load_dataset.return_value = "not_a_dataset"

        with pytest.raises(ValueError, match="Loaded dataset is not of type Dataset."):
            load_jsquad()


class TestRemoveLinebreaks:
    """Test class for remove_linebreaks function."""

    def test_正常系_改行文字除去(self):
        """改行文字が正しく除去されること"""
        text = "This is\na test\nwith\nlinebreaks"

        result = remove_linebreaks(text)

        assert result == "This isa testwithlinebreaks"

    def test_正常系_改行なしテキスト(self):
        """改行がないテキストはそのまま返されること"""
        text = "This is a text without linebreaks"

        result = remove_linebreaks(text)

        assert result == "This is a text without linebreaks"

    def test_エッジケース_空文字列(self):
        """空文字列の場合、空文字列が返されること"""
        text = ""

        result = remove_linebreaks(text)

        assert result == ""

    def test_正常系_複数種類の改行文字(self):
        """複数種類の改行文字が除去されること"""
        text = "Line1\nLine2\r\nLine3\rLine4"

        result = remove_linebreaks(text)

        assert result == "Line1Line2Line3Line4"


class TestCreateContext:
    """Test class for create_context function."""

    def test_正常系_複数インデックスからコンテキスト作成(self):
        """複数のインデックスから正しくコンテキストが作成されること"""
        # モックデータセット作成
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(
            side_effect=lambda i: {"context": f"Context {i}\nwith\nlinebreaks"}
        )
        indices = [0, 2, 5]

        result = create_context(mock_dataset, indices)

        expected = (
            "Context 0withlinebreaks\nContext 2withlinebreaks\nContext 5withlinebreaks"
        )
        assert result == expected

    def test_正常系_単一インデックス(self):
        """単一インデックスの場合も正しく処理されること"""
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(return_value={"context": "Single\ncontext"})
        indices = [1]

        result = create_context(mock_dataset, indices)

        assert result == "Singlecontext"

    def test_エッジケース_空インデックスリスト(self):
        """空のインデックスリストの場合、空文字列が返されること"""
        mock_dataset = Mock()
        indices = []

        result = create_context(mock_dataset, indices)

        assert result == ""


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


class TestGetDummyIndices:
    """Test class for get_dummy_indices function."""

    def test_正常系_前半インデックスの場合(self):
        """answer_indexが前半の場合、後半からダミーインデックスが選ばれること"""
        answer_index = 10
        dataset_length = 100

        result = get_dummy_indices(answer_index, dataset_length)

        expected = [75, 99]  # dataset_length//4 * 3, dataset_length-1
        assert result == expected

    def test_正常系_後半インデックスの場合(self):
        """answer_indexが後半の場合、前半からダミーインデックスが選ばれること"""
        answer_index = 60
        dataset_length = 100

        result = get_dummy_indices(answer_index, dataset_length)

        expected = [0, 25]  # 0, dataset_length//4
        assert result == expected

    def test_境界値_ちょうど中間の場合(self):
        """answer_indexがちょうど中間の場合の動作確認"""
        answer_index = 50
        dataset_length = 100  # 50 == 100//2

        result = get_dummy_indices(answer_index, dataset_length)

        expected = [75, 99]  # 前半として扱われる
        assert result == expected

    def test_エッジケース_小さなデータセット(self):
        """小さなデータセットでも正しく動作すること"""
        answer_index = 1
        dataset_length = 4

        result = get_dummy_indices(answer_index, dataset_length)

        expected = [3, 3]  # dataset_length//4 * 3 = 3, dataset_length-1 = 3
        assert result == expected


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

        result = extract_answer_text_by_llm(questions, contexts, model="gpt-5-nano", prompt_type="qa")

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

        result = extract_answer_text_by_llm(questions, contexts, model="test-model", prompt_type="qa")

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

        extract_answer_text_by_llm(questions, contexts, model="custom-model", prompt_type="qa")

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

        result = extract_answer_text_by_llm(questions, contexts, model="test-model", prompt_type="simple")

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

        result = extract_answer_text_by_llm(questions, contexts, model="test-model", prompt_type="qa")

        assert result == ["Test answer"]


class TestEvaluate:
    """Test class for evaluate function."""

    @patch("exact_copying_eval.core.evaluate.load_jsquad")
    @patch("exact_copying_eval.core.evaluate.extract_answer_text_by_llm")
    def test_正常系_基本的な評価実行(self, mock_extract, mock_load):
        """基本的な評価が正しく実行されること"""
        # モックデータセット設定
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)

        # オリジナルデータセットの__getitem__設定
        def mock_original_index_access(index):
            return {"context": f"Context {index} with content"}
        
        mock_dataset.__getitem__ = Mock(side_effect=mock_original_index_access)

        # selectメソッドも同じモックデータセットを返すように設定
        mock_selected_dataset = Mock()
        mock_selected_dataset.__len__ = Mock(return_value=10)  # len()メソッドを追加
        mock_selected_dataset.__getitem__ = Mock(
            side_effect=lambda key: {
                "question": [f"Question {i}" for i in range(10)],
                "context": [f"Context {i} with content" for i in range(10)],
            }[key]
        )

        # 個別インデックスアクセス用のモック設定
        def mock_index_access(index):
            return {"context": f"Context {index} with content"}

        mock_selected_dataset.__getitem__ = Mock(
            side_effect=lambda key: {
                "question": [f"Question {i}" for i in range(10)],
                "context": [f"Context {i} with content" for i in range(10)],
            }[key]
            if isinstance(key, str)
            else mock_index_access(key)
        )

        mock_dataset.select = Mock(return_value=mock_selected_dataset)
        mock_load.return_value = mock_dataset

        # モック回答設定 - 改行を除去した形になる
        mock_extract.return_value = [f"Context {i} with content" for i in range(10)]

        result = evaluate(num=10, model="test-model", batch_size=5, prompt_type="qa")

        assert "summary" in result
        assert "wrong_details" in result
        assert result["summary"]["total"] == 10
        assert result["summary"]["model"] == "test-model"
        assert result["summary"]["accuracy"] == 1.0  # 全て正解
        assert result["summary"]["correct_count"] == 10

    @patch("exact_copying_eval.core.evaluate.load_jsquad")
    @patch("exact_copying_eval.core.evaluate.extract_answer_text_by_llm")
    def test_正常系_部分的正解の場合(self, mock_extract, mock_load):
        """部分的に正解した場合の評価結果が正しいこと"""
        # モックデータセット設定
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=3)

        # オリジナルデータセットの__getitem__設定
        def mock_original_index_access(index):
            return {"context": f"Context {index}"}
        
        mock_dataset.__getitem__ = Mock(side_effect=mock_original_index_access)

        mock_selected_dataset = Mock()
        mock_selected_dataset.__len__ = Mock(return_value=3)  # len()メソッドを追加

        def mock_index_access(index):
            return {"context": f"Context {index}"}

        mock_selected_dataset.__getitem__ = Mock(
            side_effect=lambda key: {
                "question": ["Q1", "Q2", "Q3"],
                "context": ["Context 0", "Context 1", "Context 2"],
            }[key]
            if isinstance(key, str)
            else mock_index_access(key)
        )

        mock_dataset.select = Mock(return_value=mock_selected_dataset)
        mock_load.return_value = mock_dataset

        # 一部間違った回答を設定
        mock_extract.return_value = ["Context 0", "Wrong answer", "Context 2"]

        result = evaluate(num=3, model="test-model", batch_size=2, prompt_type="qa")

        assert result["summary"]["correct_count"] == 2
        assert result["summary"]["accuracy"] == 2 / 3
        assert len(result["wrong_details"]) == 1
        assert result["wrong_details"][0]["index"] == 1
        assert result["wrong_details"][0]["expected"] == "Context 1"
        assert result["wrong_details"][0]["actual"] == "Wrong answer"

    @patch("exact_copying_eval.core.evaluate.load_jsquad")
    def test_正常系_データセットサイズ調整(self, mock_load):
        """指定したnum値がデータセットサイズより大きい場合の調整"""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=5)  # データセットサイズは5

        # オリジナルデータセットの__getitem__設定
        def mock_original_index_access(index):
            return {"context": f"Context {index}"}
        
        mock_dataset.__getitem__ = Mock(side_effect=mock_original_index_access)

        mock_selected_dataset = Mock()
        mock_selected_dataset.__len__ = Mock(return_value=5)  # len()メソッドを追加

        def mock_index_access(index):
            return {"context": f"Context {index}"}

        mock_selected_dataset.__getitem__ = Mock(
            side_effect=lambda key: {
                "question": [f"Q{i}" for i in range(5)],
                "context": [f"Context {i}" for i in range(5)],
            }[key]
            if isinstance(key, str)
            else mock_index_access(key)
        )

        mock_dataset.select = Mock(return_value=mock_selected_dataset)
        mock_load.return_value = mock_dataset

        with patch(
            "exact_copying_eval.core.evaluate.extract_answer_text_by_llm"
        ) as mock_extract:
            mock_extract.return_value = [f"Context {i}" for i in range(5)]

            result = evaluate(num=10, model="test-model", prompt_type="qa")  # num=10だがデータセットは5

            assert result["summary"]["total"] == 5  # データセットサイズに調整される

    @patch("exact_copying_eval.core.evaluate.load_jsquad")
    def test_正常系_負の値指定(self, mock_load):
        """num=-1の場合、全データセットが使用されること"""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=7)

        # オリジナルデータセットの__getitem__設定
        def mock_original_index_access(index):
            return {"context": f"Context {index}"}
        
        mock_dataset.__getitem__ = Mock(side_effect=mock_original_index_access)

        mock_selected_dataset = Mock()
        mock_selected_dataset.__len__ = Mock(return_value=7)  # len()メソッドを追加

        def mock_index_access(index):
            return {"context": f"Context {index}"}

        mock_selected_dataset.__getitem__ = Mock(
            side_effect=lambda key: {
                "question": [f"Q{i}" for i in range(7)],
                "context": [f"Context {i}" for i in range(7)],
            }[key]
            if isinstance(key, str)
            else mock_index_access(key)
        )

        mock_dataset.select = Mock(return_value=mock_selected_dataset)
        mock_load.return_value = mock_dataset

        with patch(
            "exact_copying_eval.core.evaluate.extract_answer_text_by_llm"
        ) as mock_extract:
            mock_extract.return_value = [f"Context {i}" for i in range(7)]

            result = evaluate(num=-1, model="test-model", prompt_type="qa")

            assert result["summary"]["total"] == 7

    @patch("exact_copying_eval.core.evaluate.load_jsquad")
    @patch("exact_copying_eval.core.evaluate.extract_answer_text_by_llm")
    def test_正常系_prompt_type_simple(self, mock_extract, mock_load):
        """prompt_type='simple'が正しく動作すること"""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=3)

        # オリジナルデータセットの__getitem__設定
        def mock_original_index_access(index):
            return {"context": f"Context {index}"}
        
        mock_dataset.__getitem__ = Mock(side_effect=mock_original_index_access)

        mock_selected_dataset = Mock()
        mock_selected_dataset.__len__ = Mock(return_value=3)

        def mock_index_access(index):
            return {"context": f"Context {index}"}

        mock_selected_dataset.__getitem__ = Mock(
            side_effect=lambda key: {
                "question": [f"Q{i}" for i in range(3)],
                "context": [f"Context {i}" for i in range(3)],
            }[key]
            if isinstance(key, str)
            else mock_index_access(key)
        )

        mock_dataset.select = Mock(return_value=mock_selected_dataset)
        mock_load.return_value = mock_dataset

        mock_extract.return_value = [f"Context {i}" for i in range(3)]

        result = evaluate(num=3, model="test-model", batch_size=2, prompt_type="simple")

        assert result["summary"]["total"] == 3
        # extract_answer_text_by_llmがprompt_type='simple'で呼ばれているかチェック
        mock_extract.assert_called()
        call_args = mock_extract.call_args
        assert call_args[1]["prompt_type"] == "simple"
