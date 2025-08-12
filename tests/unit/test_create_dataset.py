"""Unit tests for create_dataset module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from exact_copying_eval.core.create_dataset import (
    EvaluationDataset,
    EvaluationItem,
    convert_to_random_string,
    create_context,
    create_evaluation_dataset,
    get_dummy_indices,
    load_evaluation_dataset,
    load_jsquad,
    remove_linebreaks,
)


class TestRemoveLinebreaks:
    """Test remove_linebreaks function."""

    def test_正常系_改行を削除する(self) -> None:
        """改行文字が正しく削除されることを確認。"""
        text = "line1\nline2\rline3\r\nline4"
        result = remove_linebreaks(text)
        assert result == "line1line2line3line4"

    def test_正常系_改行がない場合そのまま返す(self) -> None:
        """改行がない場合、そのまま返されることを確認。"""
        text = "no linebreaks here"
        result = remove_linebreaks(text)
        assert result == "no linebreaks here"

    def test_正常系_空文字列の場合空文字列を返す(self) -> None:
        """空文字列の場合、空文字列が返されることを確認。"""
        result = remove_linebreaks("")
        assert result == ""


class TestConvertToRandomString:
    """Test convert_to_random_string function."""

    def test_正常系_同じ長さのランダム文字列を生成する(self) -> None:
        """入力テキストと同じ長さのランダム文字列が生成されることを確認。"""
        text = "これはテストです"
        result = convert_to_random_string(text)
        assert len(result) == len(text)

    def test_正常系_元の文字列と異なる文字列を生成する(self) -> None:
        """元の文字列と異なる文字列が生成されることを確認。"""
        text = "これはテストです"
        result = convert_to_random_string(text)
        # ランダムなので、極稀に同じになる可能性があるが、現実的には異なるはず
        assert result != text

    def test_正常系_空文字列の場合空文字列を返す(self) -> None:
        """空文字列の場合、空文字列が返されることを確認。"""
        result = convert_to_random_string("")
        assert result == ""

    def test_正常系_日本語文字のみで構成される(self) -> None:
        """生成された文字列が日本語文字のみで構成されることを確認。"""
        text = "テスト文字列"
        result = convert_to_random_string(text)

        # 日本語文字（ひらがな、カタカナ、漢字）の範囲をチェック
        japanese_ranges = [
            (0x3041, 0x3096),  # ひらがな
            (0x30A1, 0x30FA),  # カタカナ
            (0x4E00, 0x9FFF),  # 漢字
        ]

        for char in result:
            char_code = ord(char)
            is_japanese = any(
                start <= char_code <= end for start, end in japanese_ranges
            )
            assert (
                is_japanese
            ), f"Non-Japanese character found: {char} (code: {char_code})"


class TestGetDummyIndices:
    """Test get_dummy_indices function."""

    def test_正常系_前半のインデックスの場合(self) -> None:
        """answer_indexが前半の場合の動作を確認。"""
        dataset_length = 100
        answer_index = 25  # <= dataset_length // 2 (50)

        result = get_dummy_indices(answer_index, dataset_length)
        expected = [75, 99]  # [dataset_length//4 * 3, dataset_length-1]

        assert result == expected

    def test_正常系_後半のインデックスの場合(self) -> None:
        """answer_indexが後半の場合の動作を確認。"""
        dataset_length = 100
        answer_index = 75  # > dataset_length // 2 (50)

        result = get_dummy_indices(answer_index, dataset_length)
        expected = [0, 25]  # [0, dataset_length//4]

        assert result == expected

    def test_正常系_境界値での動作(self) -> None:
        """境界値での動作を確認。"""
        dataset_length = 100
        answer_index = 50  # == dataset_length // 2

        result = get_dummy_indices(answer_index, dataset_length)
        expected = [75, 99]  # <= なので前半扱い

        assert result == expected


class TestCreateContext:
    """Test create_context function."""

    def test_正常系_複数のコンテキストを結合する(self) -> None:
        """複数のインデックスのコンテキストが正しく結合されることを確認。"""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.side_effect = [
            {"context": "context1\nwith\nlinebreaks"},
            {"context": "context2\rwith\rlinebreaks"},
            {"context": "context3"},
        ]

        indices = [0, 1, 2]
        result = create_context(mock_dataset, indices)

        expected = "context1withlinebreaks\ncontext2withlinebreaks\ncontext3"
        assert result == expected

    def test_正常系_単一のコンテキスト(self) -> None:
        """単一のインデックスの場合の動作を確認。"""
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = {"context": "single context"}

        indices = [0]
        result = create_context(mock_dataset, indices)

        assert result == "single context"


@patch("exact_copying_eval.core.create_dataset.load_dataset")
class TestLoadJsquad:
    """Test load_jsquad function."""

    def test_正常系_データセットを正常にロードする(
        self, mock_load_dataset: MagicMock
    ) -> None:
        """JSQuADデータセットが正常にロードされることを確認。"""
        mock_dataset = MagicMock(spec=Dataset)
        mock_load_dataset.return_value = mock_dataset

        result = load_jsquad()

        mock_load_dataset.assert_called_once_with(
            "sbintuitions/JSQuAD", split="validation"
        )
        assert result == mock_dataset

    def test_異常系_Datasetでない場合ValueErrorを発生(
        self, mock_load_dataset: MagicMock
    ) -> None:
        """ロードしたデータがDatasetでない場合、ValueErrorが発生することを確認。"""
        mock_load_dataset.return_value = "not a dataset"

        with pytest.raises(ValueError, match="Loaded dataset is not of type Dataset"):
            load_jsquad()


class TestEvaluationItem:
    """Test EvaluationItem model."""

    def test_正常系_有効なデータで作成される(self) -> None:
        """有効なデータでEvaluationItemが作成されることを確認。"""
        item = EvaluationItem(
            question="What is this?",
            context="This is a test context.",
            expected_answer="This is a test answer.",
        )

        assert item.question == "What is this?"
        assert item.context == "This is a test context."
        assert item.expected_answer == "This is a test answer."

    def test_異常系_必須フィールドが不足している場合(self) -> None:
        """必須フィールドが不足している場合、エラーが発生することを確認。"""
        with pytest.raises(ValueError):
            EvaluationItem(
                question="What is this?"
            )  # context and expected_answer missing


class TestEvaluationDataset:
    """Test EvaluationDataset model."""

    def test_正常系_有効なデータで作成される(self) -> None:
        """有効なデータでEvaluationDatasetが作成されることを確認。"""
        items = [
            EvaluationItem(question="Q1", context="C1", expected_answer="A1"),
            EvaluationItem(question="Q2", context="C2", expected_answer="A2"),
        ]
        metadata = {"num_examples": 2, "source": "test"}

        dataset = EvaluationDataset(items=items, metadata=metadata)

        assert len(dataset.items) == 2
        assert dataset.metadata == {"num_examples": 2, "source": "test"}


@patch("exact_copying_eval.core.create_dataset.load_jsquad")
class TestCreateEvaluationDataset:
    """Test create_evaluation_dataset function."""

    def test_正常系_データセットが正常に作成される(
        self, mock_load_jsquad: MagicMock
    ) -> None:
        """評価用データセットが正常に作成されることを確認。"""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100

        # Mock original dataset access for creating context
        mock_dataset.__getitem__.return_value = {"question": "Q1", "context": "A1"}
        mock_load_jsquad.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "test_dataset.json")

            result_paths = create_evaluation_dataset(nums=[1], output_path=output_path)

            assert len(result_paths) == 1
            assert result_paths[0].endswith("test_dataset_1.json")
            assert Path(result_paths[0]).exists()

            # Check file contents
            with Path(result_paths[0]).open("r", encoding="utf-8") as f:
                data = json.load(f)

            dataset = EvaluationDataset.model_validate(data)
            assert len(dataset.items) == 1
            assert dataset.metadata["num_examples"] == 1
            assert dataset.metadata["source_dataset"] == "sbintuitions/JSQuAD"

    def test_正常系_複数のデータセットが作成される(
        self, mock_load_jsquad: MagicMock
    ) -> None:
        """複数のサイズのデータセットが作成されることを確認。"""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.__getitem__.return_value = {"question": "Q1", "context": "A1"}
        mock_load_jsquad.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "test_dataset.json")

            result_paths = create_evaluation_dataset(
                nums=[1, 2], output_path=output_path
            )

            assert len(result_paths) == 2
            assert any("_1.json" in path for path in result_paths)
            assert any("_2.json" in path for path in result_paths)

            # Check both files exist
            for path in result_paths:
                assert Path(path).exists()

    def test_正常系_出力パスが指定されていない場合自動生成される(
        self, mock_load_jsquad: MagicMock
    ) -> None:
        """出力パスが指定されていない場合、自動生成されることを確認。"""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.__getitem__.return_value = {"question": "Q1", "context": "A1"}
        mock_load_jsquad.return_value = mock_dataset

        result_paths = create_evaluation_dataset(nums=[1])

        assert len(result_paths) == 1
        assert result_paths[0] == "evaluation_dataset_1.json"
        # Clean up
        Path(result_paths[0]).unlink(missing_ok=True)

    def test_正常系_numが負の値の場合全データセットを使用(
        self, mock_load_jsquad: MagicMock
    ) -> None:
        """numが負の値の場合、全データセットが使用されることを確認。"""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 5
        mock_dataset.__getitem__.return_value = {"question": "Q1", "context": "A1"}
        mock_load_jsquad.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "test_dataset.json")

            result_paths = create_evaluation_dataset(nums=[-1], output_path=output_path)

            assert len(result_paths) == 1
            # Check file contents
            with Path(result_paths[0]).open("r", encoding="utf-8") as f:
                data = json.load(f)

            dataset = EvaluationDataset.model_validate(data)
            assert dataset.metadata["num_examples"] == 5

    def test_正常系_ランダム文字列オプションが動作する(
        self, mock_load_jsquad: MagicMock
    ) -> None:
        """ランダム文字列オプションが正常に動作することを確認。"""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.__getitem__.return_value = {"question": "Q1", "context": "A1"}
        mock_load_jsquad.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "test_dataset.json")

            result_paths = create_evaluation_dataset(
                nums=[1], output_path=output_path, use_random_strings=True
            )

            assert len(result_paths) == 1
            assert "_random" in result_paths[0]

            # Check file contents
            with Path(result_paths[0]).open("r", encoding="utf-8") as f:
                data = json.load(f)

            dataset = EvaluationDataset.model_validate(data)
            assert dataset.metadata["use_random_strings"] is True


class TestLoadEvaluationDataset:
    """Test load_evaluation_dataset function."""

    def test_正常系_JSONファイルからデータセットをロードする(self) -> None:
        """JSONファイルから評価用データセットが正常にロードされることを確認。"""
        # Create test data
        test_data = {
            "items": [{"question": "Q1", "context": "C1", "expected_answer": "A1"}],
            "metadata": {"num_examples": 1, "source_dataset": "test"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_dataset.json"
            with test_file.open("w", encoding="utf-8") as f:
                json.dump(test_data, f)

            result = load_evaluation_dataset(str(test_file))

            assert len(result.items) == 1
            assert result.items[0].question == "Q1"
            assert result.metadata["num_examples"] == 1

    def test_異常系_ファイルが存在しない場合FileNotFoundError(self) -> None:
        """ファイルが存在しない場合、FileNotFoundErrorが発生することを確認。"""
        with pytest.raises(
            FileNotFoundError, match="Evaluation dataset file not found"
        ):
            load_evaluation_dataset("nonexistent_file.json")

    def test_異常系_無効なJSONファイルの場合ValueError(self) -> None:
        """無効なJSONファイルの場合、ValueErrorが発生することを確認。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "invalid.json"
            test_file.write_text("invalid json content")

            with pytest.raises(ValueError, match="Failed to load evaluation dataset"):
                load_evaluation_dataset(str(test_file))

    def test_異常系_スキーマに合わないデータの場合ValueError(self) -> None:
        """スキーマに合わないデータの場合、ValueErrorが発生することを確認。"""
        invalid_data = {
            "items": [
                {
                    "question": "Q1",
                    # missing context and expected_answer
                }
            ],
            "metadata": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "invalid_schema.json"
            with test_file.open("w", encoding="utf-8") as f:
                json.dump(invalid_data, f)

            with pytest.raises(ValueError, match="Failed to load evaluation dataset"):
                load_evaluation_dataset(str(test_file))
