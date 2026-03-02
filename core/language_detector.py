"""
CasaAI 言語検出モジュール
Whisperの検出結果を正規化し、対応言語かどうかを判定する
"""
import logging
from enum import Enum


class SupportedLanguage(Enum):
    """サポート対象言語の列挙型"""
    ITALIAN = "it"
    JAPANESE = "ja"
    ENGLISH = "en"
    UNKNOWN = "unknown"


# Whisperの言語コードからSupportedLanguageへのマッピング
_LANGUAGE_MAP: dict[str, SupportedLanguage] = {
    "it": SupportedLanguage.ITALIAN,
    "ja": SupportedLanguage.JAPANESE,
    "en": SupportedLanguage.ENGLISH,
}

# UI表示用の言語名マッピング
LANGUAGE_DISPLAY_NAMES: dict[SupportedLanguage, str] = {
    SupportedLanguage.ITALIAN: "Italiano 🇮🇹",
    SupportedLanguage.JAPANESE: "日本語 🇯🇵",
    SupportedLanguage.ENGLISH: "English 🇬🇧",
    SupportedLanguage.UNKNOWN: "Unknown ❓",
}


class LanguageDetector:
    """
    Whisperの言語検出結果を正規化するサービス

    Whisperが返す言語コードをSupportedLanguage列挙型に変換し、
    サポート対象言語かどうかを判定する。
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def detect(self, whisper_language_code: str) -> SupportedLanguage:
        """
        Whisperの言語コードをSupportedLanguageに変換

        Args:
            whisper_language_code: Whisperが検出した言語コード（例: "it", "ja", "en"）

        Returns:
            対応するSupportedLanguage列挙値
        """
        code = whisper_language_code.lower().strip()
        lang = _LANGUAGE_MAP.get(code, SupportedLanguage.UNKNOWN)
        self._logger.debug("言語検出: '%s' → %s", code, lang.value)
        return lang

    def get_display_name(self, language: SupportedLanguage) -> str:
        """
        言語のUI表示用名称を取得

        Args:
            language: SupportedLanguage列挙値

        Returns:
            表示用文字列
        """
        return LANGUAGE_DISPLAY_NAMES.get(language, "Unknown ❓")

    def is_supported(self, language: SupportedLanguage) -> bool:
        """
        言語がサポート対象かどうかを判定

        Args:
            language: 判定対象の言語

        Returns:
            サポート対象ならTrue
        """
        return language != SupportedLanguage.UNKNOWN