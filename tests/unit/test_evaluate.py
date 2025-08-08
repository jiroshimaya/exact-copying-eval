"""Test script for main.py functions."""

from exact_copying_eval.core.evaluate import get_answer_text, split_text_with_periods


class TestGetAnswerText:
    """Test class for get_answer_text function."""

    def test_正常系_回答が中間の文にある場合(self):
        """回答が中間の文に含まれる場合、正しい文IDを返すこと"""
        sentences = ["This is first.", "This is second.", "This is third."]
        # "second" の開始位置は "This is first. This is " の後 = 16 + 8 = 24
        answer_start = 24

        result = get_answer_text(sentences, answer_start)

        assert result == 1  # 2番目の文（インデックス1）

    def test_正常系_回答が最初の文にある場合(self):
        """回答が最初の文に含まれる場合、文ID 0を返すこと"""
        sentences = ["This is first.", "This is second.", "This is third."]
        answer_start = 5  # "first" の開始位置

        result = get_answer_text(sentences, answer_start)

        assert result == 0

    def test_正常系_回答が最後の文にある場合(self):
        """回答が最後の文に含まれる場合、正しい文IDを返すこと"""
        sentences = ["This is first.", "This is second.", "This is third."]
        # "third" の開始位置は "This is first. This is second. This is " の後
        # = 16 + 17 + 8 = 41
        answer_start = 41

        result = get_answer_text(sentences, answer_start)

        assert result == 2  # 3番目の文（インデックス2）

    def test_エッジケース_回答開始位置が文の最初(self):
        """回答開始位置が文の最初の文字と一致する場合"""
        sentences = ["Hello", "World", "Test"]
        answer_start = 6  # "World" の開始位置（"Hello " の後）

        result = get_answer_text(sentences, answer_start)

        assert result == 1

    def test_エッジケース_単一文の場合(self):
        """文が1つしかない場合"""
        sentences = ["Only one sentence."]
        answer_start = 5  # "one" の開始位置

        result = get_answer_text(sentences, answer_start)

        assert result == 0

    def test_エッジケース_空文字列が含まれる場合(self):
        """空文字列が含まれる文リストの場合"""
        sentences = ["First", "", "Third"]
        answer_start = (
            7  # "Third" の開始位置（"First  " の後、空文字列分のスペースも含む）
        )

        result = get_answer_text(sentences, answer_start)

        assert result == 2

    def test_境界値_回答位置が範囲外の場合(self):
        """回答開始位置が全文の範囲を超える場合、最後の文IDを返すこと"""
        sentences = ["First", "Second", "Third"]
        answer_start = 100  # 明らかに範囲外

        result = get_answer_text(sentences, answer_start)

        assert result == 2  # 最後の文のインデックス

    def test_異常系_空の文リストの場合(self):
        """空の文リストが渡された場合の動作確認"""
        sentences: list[str] = []
        answer_start = 0

        result = get_answer_text(sentences, answer_start)

        assert result == -1  # len([]) - 1 = -1

    def test_境界値_回答位置が0の場合(self):
        """回答開始位置が0の場合"""
        sentences = ["First sentence", "Second sentence"]
        answer_start = 0

        result = get_answer_text(sentences, answer_start)

        assert result == 0


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
        
        assert result == ["文章。", "。", "別の文章。"]  # 空文字列も保持（元文字列復元のため）

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

