"""
CasaAI LLMクライアントモジュール
OllamaのREST APIを通じてローカルLLMと対話する
"""
import logging
from dataclasses import dataclass
from typing import Protocol

import requests

from core.language_detector import SupportedLanguage


@dataclass(frozen=True)
class LLMConfig:
    """LLM設定の値オブジェクト"""
    base_url: str
    model: str
    max_tokens: int
    temperature: float
    timeout_sec: int
    context_window: int


@dataclass(frozen=True)
class LLMResponse:
    """LLM応答の値オブジェクト"""
    text: str
    model: str
    total_duration_ms: float


class LLMClientProtocol(Protocol):
    """LLMクライアントの抽象インターフェース"""
    def generate_response(
        self,
        user_text: str,
        detected_language: SupportedLanguage,
        conversation_history: list[dict[str, str]],
    ) -> LLMResponse | None: ...


# システムプロンプト: イタリア語アシスタントの振る舞いを定義
_SYSTEM_PROMPT = """Sei CasaAI, un assistente domestico familiare italiano. 

REGOLE FONDAMENTALI:
1. Rispondi SEMPRE in italiano, indipendentemente dalla lingua dell'utente.
2. Se l'utente parla in giapponese o inglese, fornisci prima la traduzione in italiano di ciò che hanno detto, poi rispondi alla loro domanda in italiano.
3. Se l'utente parla in italiano, rispondi normalmente. Se noti errori grammaticali, correggili gentilmente.
4. Sii amichevole, educato e adatto a tutta la famiglia.
5. Non discutere di politica, violenza o contenuti inappropriati.
6. Mantieni le risposte concise (massimo 3-4 frasi).
7. Usa un tono caldo e familiare, come un membro della famiglia.

FORMATO RISPOSTA:
- Se la lingua rilevata NON è italiano: "[Traduzione: ...] + risposta"
- Se la lingua è italiano: risposta diretta con eventuale correzione grammaticale
"""


class LLMClient:
    """
    OllamaのREST APIを使用するLLMクライアント

    ローカルのOllamaサーバーにHTTPリクエストを送信し、
    テキスト生成結果を受け取る。会話履歴をコンテキストとして
    送信し、マルチターン対話を実現する。
    """

    def __init__(self, config: LLMConfig, logger: logging.Logger) -> None:
        """
        Args:
            config: LLM設定
            logger: ロガーインスタンス
        """
        self._config = config
        self._logger = logger
        self._api_url = f"{config.base_url}/api/chat"

    def health_check(self) -> bool:
        """
        Ollamaサーバーの死活確認

        Returns:
            サーバーが応答すればTrue
        """
        try:
            resp = requests.get(
                f"{self._config.base_url}/api/tags",
                timeout=5,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def generate_response(
        self,
        user_text: str,
        detected_language: SupportedLanguage,
        conversation_history: list[dict[str, str]],
    ) -> LLMResponse | None:
        """
        ユーザー入力に対するLLM応答を生成する

        Args:
            user_text: ユーザーの発話テキスト
            detected_language: 検出された言語
            conversation_history: 過去の会話履歴（role/contentのリスト）

        Returns:
            LLMResponse（応答テキスト、モデル名、処理時間）
            エラー時はNone

        Notes:
            コンテキストウィンドウ管理のため、履歴は最新のものに制限される。
            システムプロンプトに検出言語の情報を追加して精度を向上させる。
        """
        # 検出言語情報をシステムプロンプトに付加
        lang_hint = self._build_language_hint(detected_language)
        system_msg = _SYSTEM_PROMPT + f"\n\nLingua rilevata dell'utente: {lang_hint}"

        messages = [{"role": "system", "content": system_msg}]

        # 会話履歴を追加（コンテキストウィンドウ管理）
        for entry in conversation_history[-6:]:
            messages.append(entry)

        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self._config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": self._config.max_tokens,
                "temperature": self._config.temperature,
            },
        }

        self._logger.info("LLMリクエスト送信: model=%s, msgs=%d", self._config.model, len(messages))

        try:
            resp = requests.post(
                self._api_url,
                json=payload,
                timeout=self._config.timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()

            reply = data.get("message", {}).get("content", "").strip()
            total_ns = data.get("total_duration", 0)
            total_ms = total_ns / 1_000_000

            if not reply:
                self._logger.warning("LLM空応答")
                return None

            result = LLMResponse(
                text=reply,
                model=data.get("model", self._config.model),
                total_duration_ms=round(total_ms, 1),
            )
            self._logger.info("LLM応答完了: %.1fms, text='%s'", result.total_duration_ms, reply[:80])
            return result

        except requests.Timeout:
            self._logger.error("LLMリクエストタイムアウト (%d秒)", self._config.timeout_sec)
            return None
        except requests.RequestException as e:
            self._logger.error("LLMリクエストエラー: %s", e)
            return None
        except (KeyError, ValueError) as e:
            self._logger.error("LLMレスポンスパースエラー: %s", e)
            return None

    def _build_language_hint(self, lang: SupportedLanguage) -> str:
        """
        LLMに渡す言語ヒント文字列を構築

        Args:
            lang: 検出された言語

        Returns:
            イタリア語での言語名
        """
        hints = {
            SupportedLanguage.ITALIAN: "Italiano",
            SupportedLanguage.JAPANESE: "Giapponese",
            SupportedLanguage.ENGLISH: "Inglese",
            SupportedLanguage.UNKNOWN: "Sconosciuta",
        }
        return hints.get(lang, "Sconosciuta")