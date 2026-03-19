"""Tests for CJK-aware word counting."""

from __future__ import annotations

from markitai.webextract.utils import count_words


class TestCountWords:
    def test_english_words(self):
        assert count_words("hello world") == 2

    def test_empty_string(self):
        assert count_words("") == 0

    def test_whitespace_only(self):
        assert count_words("   \n\t  ") == 0

    def test_chinese_characters(self):
        """Each CJK character counts as one word."""
        assert count_words("你好世界") == 4

    def test_japanese_hiragana(self):
        assert count_words("こんにちは") == 5

    def test_japanese_katakana(self):
        assert count_words("カタカナ") == 4

    def test_korean_hangul(self):
        assert count_words("안녕하세요") == 5

    def test_mixed_cjk_and_english(self):
        """CJK characters + English words should both count."""
        result = count_words("Hello 你好 world 世界")
        # 2 English words + 4 CJK characters = 6
        assert result == 6

    def test_cjk_with_punctuation(self):
        """CJK punctuation should not count."""
        result = count_words("你好，世界！")
        assert result == 4

    def test_realistic_chinese_sentence(self):
        text = "人工智能正在改变世界"  # 10 CJK characters
        assert count_words(text) == 10
