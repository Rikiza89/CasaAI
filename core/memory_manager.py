"""
CasaAI 会話メモリ管理モジュール
直近N件の会話をJSONファイルに永続化する
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class MemoryConfig:
    """メモリ設定の値オブジェクト"""
    max_entries: int
    file_path: str


@dataclass
class ConversationEntry:
    """会話エントリの構造体"""
    timestamp: str
    user_text: str
    user_language: str
    assistant_text: str

    def to_dict(self) -> dict[str, str]:
        """辞書形式に変換"""
        return {
            "timestamp": self.timestamp,
            "user_text": self.user_text,
            "user_language": self.user_language,
            "assistant_text": self.assistant_text,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "ConversationEntry":
        """辞書から生成"""
        return cls(
            timestamp=d["timestamp"],
            user_text=d["user_text"],
            user_language=d["user_language"],
            assistant_text=d["assistant_text"],
        )


class MemoryManagerProtocol(Protocol):
    """メモリ管理の抽象インターフェース"""
    def add_entry(self, user_text: str, user_language: str, assistant_text: str) -> None: ...
    def get_history_for_llm(self) -> list[dict[str, str]]: ...
    def get_entries(self) -> list[ConversationEntry]: ...


class MemoryManager:
    """
    JSON永続化ベースの会話メモリ管理サービス

    直近max_entries件の会話を保持し、ファイルに書き出す。
    LLMコンテキスト用にrole/content形式への変換機能を提供する。
    ファイルI/Oエラーはログに記録し、メモリ上のデータは維持する。
    """

    def __init__(self, config: MemoryConfig, logger: logging.Logger) -> None:
        """
        Args:
            config: メモリ設定
            logger: ロガーインスタンス
        """
        self._config = config
        self._logger = logger
        self._entries: list[ConversationEntry] = []
        self._file_path = Path(config.file_path)

        # ディレクトリ作成と既存データの読み込み
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def add_entry(
        self,
        user_text: str,
        user_language: str,
        assistant_text: str,
    ) -> None:
        """
        会話エントリを追加し、永続化する

        Args:
            user_text: ユーザーの発話テキスト
            user_language: 検出された言語コード
            assistant_text: アシスタントの応答テキスト

        Notes:
            max_entriesを超えた場合、古いエントリを自動削除
        """
        entry = ConversationEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_text=user_text,
            user_language=user_language,
            assistant_text=assistant_text,
        )
        self._entries.append(entry)

        # 最大件数を超えた場合、古いエントリを削除
        if len(self._entries) > self._config.max_entries:
            removed = len(self._entries) - self._config.max_entries
            self._entries = self._entries[-self._config.max_entries:]
            self._logger.debug("古い会話エントリを%d件削除", removed)

        self._save()
        self._logger.info("会話エントリ追加: 合計%d件", len(self._entries))

    def get_history_for_llm(self) -> list[dict[str, str]]:
        """
        LLMコンテキスト用に会話履歴をrole/content形式で返す

        Returns:
            {"role": "user"/"assistant", "content": "..."} のリスト
        """
        history: list[dict[str, str]] = []
        for e in self._entries:
            history.append({"role": "user", "content": e.user_text})
            history.append({"role": "assistant", "content": e.assistant_text})
        return history

    def get_entries(self) -> list[ConversationEntry]:
        """保存済み全エントリを返す"""
        return list(self._entries)

    def get_entry_count(self) -> int:
        """保存済みエントリ数を返す"""
        return len(self._entries)

    def _load(self) -> None:
        """JSONファイルから会話履歴を読み込む"""
        if not self._file_path.exists():
            self._logger.info("会話履歴ファイルなし、新規作成")
            return
        try:
            raw = self._file_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            self._entries = [ConversationEntry.from_dict(d) for d in data]
            self._logger.info("会話履歴を%d件読み込み", len(self._entries))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self._logger.error("会話履歴ファイルの読み込みエラー: %s", e)
            self._entries = []

    def _save(self) -> None:
        """会話履歴をJSONファイルに書き出す"""
        try:
            data = [e.to_dict() for e in self._entries]
            self._file_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            self._logger.error("会話履歴の保存エラー: %s", e)