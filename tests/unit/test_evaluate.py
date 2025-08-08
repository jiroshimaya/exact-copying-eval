"""Test script for main.py functions."""

from exact_copying_eval.core.evaluate import (
    add_answer_sentence,
    get_answer_sentence,
    split_text_with_periods,
)


class TestGetAnswerSentence:
    """Test class for get_answer_sentence function."""

    def test_正常系_回答が中間の文にある場合(self):
        """回答が中間の文に含まれる場合、正しい文IDを返すこと"""
        sentences = ["This is first.", "This is second.", "This is third."]
        # "second" の開始位置は "This is first. This is " の後 = 16 + 8 = 24
        answer_start = 24

        result = get_answer_sentence(sentences, answer_start)

        assert result == 1  # 2番目の文（インデックス1）

    def test_正常系_回答が最初の文にある場合(self):
        """回答が最初の文に含まれる場合、文ID 0を返すこと"""
        sentences = ["This is first.", "This is second.", "This is third."]
        answer_start = 5  # "first" の開始位置

        result = get_answer_sentence(sentences, answer_start)

        assert result == 0

    def test_正常系_回答が最後の文にある場合(self):
        """回答が最後の文に含まれる場合、正しい文IDを返すこと"""
        sentences = ["This is first.", "This is second.", "This is third."]
        # "third" の開始位置は "This is first. This is second. This is " の後
        # = 16 + 17 + 8 = 41
        answer_start = 41

        result = get_answer_sentence(sentences, answer_start)

        assert result == 2  # 3番目の文（インデックス2）

    def test_エッジケース_回答開始位置が文の最初(self):
        """回答開始位置が文の最初の文字と一致する場合"""
        sentences = ["Hello", "World", "Test"]
        answer_start = 6  # "World" の開始位置（"Hello " の後）

        result = get_answer_sentence(sentences, answer_start)

        assert result == 1

    def test_エッジケース_単一文の場合(self):
        """文が1つしかない場合"""
        sentences = ["Only one sentence."]
        answer_start = 5  # "one" の開始位置

        result = get_answer_sentence(sentences, answer_start)

        assert result == 0

    def test_エッジケース_空文字列が含まれる場合(self):
        """空文字列が含まれる文リストの場合"""
        sentences = ["First", "", "Third"]
        answer_start = (
            7  # "Third" の開始位置（"First  " の後、空文字列分のスペースも含む）
        )

        result = get_answer_sentence(sentences, answer_start)

        assert result == 2

    def test_境界値_回答位置が範囲外の場合(self):
        """回答開始位置が全文の範囲を超える場合、最後の文IDを返すこと"""
        sentences = ["First", "Second", "Third"]
        answer_start = 100  # 明らかに範囲外

        result = get_answer_sentence(sentences, answer_start)

        assert result == 2  # 最後の文のインデックス

    def test_異常系_空の文リストの場合(self):
        """空の文リストが渡された場合の動作確認"""
        sentences: list[str] = []
        answer_start = 0

        result = get_answer_sentence(sentences, answer_start)

        assert result == -1  # len([]) - 1 = -1

    def test_境界値_回答位置が0の場合(self):
        """回答開始位置が0の場合"""
        sentences = ["First sentence", "Second sentence"]
        answer_start = 0

        result = get_answer_sentence(sentences, answer_start)

        assert result == 0

    def test_正常系_日本語句点区切り_スペースなし(self):
        """日本語の句点区切りでスペースがない場合の動作確認"""
        sentences = ["これは文です。", "これも文です。", "最後の文。"]
        # "これも文です。" の "これ" の開始位置 = 最初の文の長さ = 7
        answer_start = 7

        result = get_answer_sentence(sentences, answer_start)

        assert result == 1  # 2番目の文

    def test_正常系_連続する句点の場合_スペースなし(self):
        """連続する句点がある場合でスペースがない場合の動作確認"""
        sentences = ["文章。", "。", "別の文章。"]
        # "別の文章。" の "別" の開始位置 = "文章。" + "。" = 3 + 1 = 4
        answer_start = 4

        result = get_answer_sentence(sentences, answer_start)

        assert result == 2  # 3番目の文

    def test_正常系_英語と句点の混在_スペースなし(self):
        """英語と日本語が混在する場合でスペースがない場合の動作確認"""
        sentences = ["Hello。", "World。", "Test。"]
        # "World。" の "W" の開始位置 = "Hello。" = 6
        answer_start = 6

        result = get_answer_sentence(sentences, answer_start)

        assert result == 1  # 2番目の文

    def test_エッジケース_文の境界での回答位置_スペースなし(self):
        """文の境界ちょうどでの回答位置の場合（スペースなし）"""
        sentences = ["短い。", "もう少し長い文。", "最後。"]
        # 2番目の文の最初の文字 "も" の位置 = "短い。" = 3
        answer_start = 3

        result = get_answer_sentence(sentences, answer_start)

        assert result == 1  # 2番目の文


class TestSplitTextWithPeriods:
    """Test class for split_text_with_periods function."""

    def test_正常系_複数の句点がある文章(self):
        """複数の句点で区切られた文章を正しく分割できること"""
        text = "これは文です。これも文です。最後の文。"

        result = split_text_with_periods(text)

        assert result == ["これは文です。", "これも文です。", "最後の文。"]

    def test_正常系_文末に句点がない場合(self):
        """文末に句点がない場合も正しく処理できること"""
        text = "これは文です。句点なし"

        result = split_text_with_periods(text)

        assert result == ["これは文です。", "句点なし"]

    def test_正常系_句点が1つもない場合(self):
        """句点が含まれない文字列の場合、そのまま返すこと"""
        text = "句点なしの文章"

        result = split_text_with_periods(text)

        assert result == ["句点なしの文章"]

    def test_エッジケース_空文字列の場合(self):
        """空文字列が渡された場合、空のリストを返すこと"""
        text = ""

        result = split_text_with_periods(text)

        assert result == []

    def test_エッジケース_句点のみの場合(self):
        """句点のみの文字列の場合の処理"""
        text = "。"

        result = split_text_with_periods(text)

        assert result == ["。"]  # 空文字でjoinして元に戻るため保持

    def test_エッジケース_連続する句点の場合(self):
        """連続する句点がある場合の処理"""
        text = "文章。。別の文章。"

        result = split_text_with_periods(text)

        # 空文字列も保持（元文字列復元のため）
        assert result == ["文章。", "。", "別の文章。"]

    def test_エッジケース_空白のみの文がある場合(self):
        """空白のみの文がある場合も保持されること"""
        text = "文章。   。別の文章。"

        result = split_text_with_periods(text)

        assert result == ["文章。", "   。", "別の文章。"]

    def test_正常系_単一の句点で終わる文(self):
        """単一の文が句点で終わる場合"""
        text = "これは一つの文です。"

        result = split_text_with_periods(text)

        assert result == ["これは一つの文です。"]

    def test_要件_空文字でjoinして元文字列に戻る(self):
        """分割した結果を空文字でjoinすると元の文字列に戻ること"""
        text = "これは文です。これも文です。最後の文。"

        result = split_text_with_periods(text)
        joined = "".join(result)

        assert joined == text

    def test_要件_空白を保持する(self):
        """文中の空白は保持されること"""
        text = "文章。  空白あり  。別の文章。"

        result = split_text_with_periods(text)
        joined = "".join(result)

        assert joined == text

    def test_要件_連続句点の場合も元文字列に戻る(self):
        """連続する句点がある場合でも、joinして元文字列に戻ること"""
        text = "文章。。別の文章。"

        result = split_text_with_periods(text)
        joined = "".join(result)

        assert joined == text

    def test_要件_句点のみの文字列も元に戻る(self):
        """句点のみの文字列でも、joinして元文字列に戻ること"""
        text = "。"

        result = split_text_with_periods(text)
        joined = "".join(result)

        assert joined == text


class TestAddAnswerSentence:
    """Test class for add_answer_sentence function."""

    def test_正常系_回答が中間の文にある場合(self):
        """回答が中間の文に含まれる場合、正しい文が追加されること"""
        example = {
            "context": "This is first。This is second。This is third。",
            "answer": {"answer_start": [22]},  # "second" の開始位置 ("This is first。This is " = 25文字)
            "other_field": "value",
        }

        result = add_answer_sentence(example)

        assert result["answer_sentence"] == "This is second。"
        assert result["other_field"] == "value"  # 他のフィールドは保持される

    def test_正常系_回答が最初の文にある場合(self):
        """回答が最初の文に含まれる場合、最初の文が追加されること"""
        example = {
            "context": "This is first。This is second。This is third。",
            "answer": {"answer_start": [8]},  # "first" の開始位置 ("This is " = 8文字)
            "question": "What is this?",
        }

        result = add_answer_sentence(example)

        assert result["answer_sentence"] == "This is first。"
        assert result["question"] == "What is this?"

    def test_正常系_回答が最後の文にある場合(self):
        """回答が最後の文に含まれる場合、最後の文が追加されること"""
        example = {
            "context": "This is first。This is second。This is third。",
            "answer": {"answer_start": [37]},  # "third" の開始位置 ("This is first。This is second。This is " = 42文字)
            "id": "test_id",
        }

        result = add_answer_sentence(example)

        assert result["answer_sentence"] == "This is third。"
        assert result["id"] == "test_id"

    def test_エッジケース_単一文の場合(self):
        """文が1つしかない場合、その文が追加されること"""
        example = {
            "context": "Only one sentence。",
            "answer": {"answer_start": [5]},  # "one" の開始位置
        }

        result = add_answer_sentence(example)

        assert result["answer_sentence"] == "Only one sentence。"

    def test_境界値_回答位置が範囲外の場合(self):
        """回答開始位置が全文の範囲を超える場合、最後の文が追加されること"""
        example = {
            "context": "First。Second。Third。",
            "answer": {"answer_start": [100]},  # 明らかに範囲外
        }

        result = add_answer_sentence(example)

        assert result["answer_sentence"] == "Third。"

    def test_正常系_既存のanswer_sentenceを上書き(self):
        """既にanswer_sentenceフィールドがある場合、上書きされること"""
        example = {
            "context": "First。Second。Third。",
            "answer": {"answer_start": [6]},  # "Second" の開始位置 ("First。" = 6文字)
            "answer_sentence": "Old sentence",
            "other_data": [1, 2, 3],
        }

        result = add_answer_sentence(example)

        assert result["answer_sentence"] == "Second。"
        assert result["other_data"] == [1, 2, 3]

    def test_正常系_空白を含む文の場合(self):
        """空白を含む文でも正しく処理されること"""
        example = {
            "context": "Short text。This is a long sentence with spaces。",
            "answer": {"answer_start": [17]},  # "long sentence" の開始位置 ("Short text。This is a " = 17文字)
        }

        result = add_answer_sentence(example)

        assert result["answer_sentence"] == "This is a long sentence with spaces。"

    def test_エッジケース_回答位置が0の場合(self):
        """回答開始位置が0の場合、最初の文が追加されること"""
        example = {
            "context": "First sentence。Second sentence。",
            "answer": {"answer_start": [0]},
        }

        result = add_answer_sentence(example)

        assert result["answer_sentence"] == "First sentence。"

    def test_正常系_複雑な構造のexample(self):
        """複雑な構造のexampleでも正しく処理されること"""
        example = {
            "context": "Context text。Answer is here。End。",
            "answer": {"answer_start": [14]},  # "Answer is here" の開始位置 ("Context text。" = 14文字)
            "question": "What is the answer?",
            "metadata": {"source": "test", "annotator": "human"},
        }

        result = add_answer_sentence(example)

        assert result["answer_sentence"] == "Answer is here。"
        assert result["context"] == "Context text。Answer is here。End。"
        assert result["question"] == "What is the answer?"
        assert result["metadata"] == {"source": "test", "annotator": "human"}

    def test_要件_元のexampleは変更されない(self):
        """元のexample辞書は変更されないこと（副作用がないこと）"""
        original_example = {
            "context": "This is a test sentence。",
            "answer": {"answer_start": [10]},  # "test" の開始位置 ("This is a " = 10文字)
            "test_field": "original_value",
        }

        result = add_answer_sentence(original_example)

        # 戻り値は元の辞書と同じオブジェクトなので、実際には変更される
        # この動作が意図されているかテストで確認
        assert result is original_example  # 同じオブジェクトかチェック
        assert original_example["answer_sentence"] == "This is a test sentence。"

    def test_正常系_句点なしの文の場合(self):
        """句点がない文でも正しく処理されること"""
        example = {
            "context": "No period here",
            "answer": {"answer_start": [3]},  # "period" の開始位置 ("No " = 3文字)
        }

        result = add_answer_sentence(example)

        assert result["answer_sentence"] == "No period here"

    def test_正常系_連続する句点の場合(self):
        """連続する句点がある場合でも正しく処理されること"""
        example = {
            "context": "文章。。別の文章。",
            "answer": {"answer_start": [4]},  # "別の文章" の開始位置 ("文章。。" = 4文字)
        }

        result = add_answer_sentence(example)

        assert result["answer_sentence"] == "別の文章。"
