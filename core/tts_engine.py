"""
CasaAI テキスト読み上げ（TTS）エンジン
Piperを使用してイタリア語テキストをWAV音声に変換する
"""
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class TTSConfig:
    """TTS設定の値オブジェクト"""
    piper_executable: str
    voice_model: str
    output_sample_rate: int
    speaker_id: int


class TTSEngineProtocol(Protocol):
    """TTSエンジンの抽象インターフェース"""
    def synthesize(self, text: str) -> bytes | None: ...


class TTSEngine:
    """
    Piperベースのオフラインテキスト読み上げエンジン

    Piperのコマンドラインインターフェースをサブプロセスで呼び出し、
    テキストをWAVバイトデータに変換する。
    一時ファイルを使用して出力を受け取り、完了後に削除する。
    """

    def __init__(self, config: TTSConfig, logger: logging.Logger) -> None:
        """
        Args:
            config: TTS設定
            logger: ロガーインスタンス

        Raises:
            FileNotFoundError: Piper実行ファイルまたはボイスモデルが存在しない場合
        """
        self._config = config
        self._logger = logger

        # 起動時にファイル存在を確認
        exe_path = Path(config.piper_executable)
        voice_path = Path(config.voice_model)

        if not exe_path.exists():
            raise FileNotFoundError(f"Piper実行ファイルが見つかりません: {exe_path}")
        if not voice_path.exists():
            raise FileNotFoundError(f"Piperボイスモデルが見つかりません: {voice_path}")

        self._logger.info("TTSエンジン初期化完了: voice=%s", voice_path.name)

    @staticmethod
    def _sanitize_for_italian_tts(text: str) -> str:
        """
        Piperイタリア語モデル向けにテキストをサニタイズする

        CJK文字、特殊記号、制御文字を除去し、
        Latin系文字とイタリア語で使用される文字のみを保持する。

        Args:
            text: 入力テキスト

        Returns:
            サニタイズ済みテキスト
        """
        # [Traduzione: ...] ブラケット記法はそのまま保持するが中のCJKは除去
        # CJK統合漢字、ひらがな、カタカナ、ハングルを除去
        text = re.sub(r'[\u3000-\u9fff\uac00-\ud7af\uff00-\uffef]', '', text)
        # 制御文字を除去
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        # 連続スペースを1つに
        text = re.sub(r'\s+', ' ', text).strip()
        # 空のブラケットを除去
        text = re.sub(r'\[Traduzione:\s*\]', '', text).strip()
        return text

    def synthesize(self, text: str) -> bytes | None:
        """
        テキストをWAV音声バイトデータに変換する

        Args:
            text: 読み上げ対象のテキスト（イタリア語想定）

        Returns:
            WAV形式のバイトデータ
            変換失敗時はNone

        Notes:
            Piperはstdinからテキストを受け取り、指定ファイルにWAVを出力する。
            処理完了後に一時ファイルを削除する。
        """
        if not text or not text.strip():
            self._logger.warning("空テキストがTTSに渡された")
            return None

        # テキスト前処理: 改行をスペースに、過度な空白を除去
        cleaned = " ".join(text.strip().split())

        # Piperイタリア語モデルが処理できない文字を除去
        # CJK文字、特殊Unicode、制御文字を削除し、Latin系のみ保持
        cleaned = self._sanitize_for_italian_tts(cleaned)

        if not cleaned:
            self._logger.warning("サニタイズ後にテキストが空になった")
            return None

        self._logger.info("TTS合成開始: '%s'", cleaned[:80])

        tmp_path: Path | None = None
        try:
            # 一時WAVファイルのパスを生成
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            cmd = [
                self._config.piper_executable,
                "--model", self._config.voice_model,
                "--output_file", str(tmp_path),
                "--speaker", str(self._config.speaker_id),
            ]

            # Piperプロセスを実行、stdinにテキストを渡す
            # Windows環境ではcp932がデフォルトになるため、UTF-8を明示指定
            proc = subprocess.run(
                cmd,
                input=cleaned.encode("utf-8"),
                capture_output=True,
                timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

            if proc.returncode != 0:
                stderr_text = proc.stderr.decode("utf-8", errors="replace")
                self._logger.error("Piperエラー (code=%d): %s", proc.returncode, stderr_text)
                return None

            # 出力WAVファイルを読み込み
            wav_data = tmp_path.read_bytes()
            if len(wav_data) < 100:
                self._logger.warning("Piper出力が不正に小さい: %d bytes", len(wav_data))
                return None

            self._logger.info("TTS合成完了: %d bytes", len(wav_data))
            return wav_data

        except subprocess.TimeoutExpired:
            self._logger.error("Piperプロセスタイムアウト (30秒)")
            return None
        except (OSError, subprocess.SubprocessError) as e:
            self._logger.error("TTSサブプロセスエラー: %s", e)
            return None
        finally:
            # 一時ファイルのクリーンアップ
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass