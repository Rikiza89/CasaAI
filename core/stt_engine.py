"""
CasaAI 音声認識（STT）エンジン
faster-whisperをGPUで実行し、音声をテキストに変換する
"""
import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from faster_whisper import WhisperModel


@dataclass(frozen=True)
class STTConfig:
    """STT設定の値オブジェクト"""
    model_size: str
    device: str
    compute_type: str
    beam_size: int
    language: str | None
    vad_filter: bool
    vad_min_silence_ms: int
    vad_speech_pad_ms: int


@dataclass(frozen=True)
class STTResult:
    """STT結果の値オブジェクト"""
    text: str
    language: str
    language_probability: float
    duration_sec: float


class STTEngineProtocol(Protocol):
    """STTエンジンの抽象インターフェース"""
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> STTResult | None: ...


class STTEngine:
    """
    faster-whisperベースの音声認識エンジン

    CUDAデバイスでfloat16推論を行い、低レイテンシを実現。
    VADフィルターで無音区間を自動的にスキップする。
    モデルは初回利用時にロードされ、以降はキャッシュされる。
    """

    def __init__(self, config: STTConfig, logger: logging.Logger) -> None:
        """
        Args:
            config: STT設定
            logger: ロガーインスタンス

        Notes:
            モデルのロードはコンストラクタで実行（起動時の遅延を許容）
        """
        self._config = config
        self._logger = logger
        # RTX 50XX (Blackwell) ではint8系compute_typeが非対応
        # cuBLAS_STATUS_NOT_SUPPORTED回避のため安全なtypeに強制
        safe_compute = self._ensure_safe_compute_type(config.compute_type)

        self._logger.info(
            "Whisperモデルロード開始: size=%s, device=%s, compute=%s",
            config.model_size, config.device, safe_compute,
        )
        self._model = WhisperModel(
            model_size_or_path=config.model_size,
            device=config.device,
            compute_type=safe_compute,
        )
        self._logger.info("Whisperモデルロード完了")

    @staticmethod
    def _ensure_safe_compute_type(compute_type: str) -> str:
        """
        RTX 50XX (Blackwell) 互換のcompute_typeを保証する

        int8系はBlackwell GPUでcuBLAS_STATUS_NOT_SUPPORTEDエラーを
        引き起こすため、float16にフォールバックする。

        Args:
            compute_type: 要求されたcompute_type

        Returns:
            Blackwell互換のcompute_type
        """
        # int8系をブロックリスト
        unsafe_types = {"int8", "int8_float16", "int8_float32", "int8_bfloat16", "auto"}
        if compute_type.lower() in unsafe_types:
            return "float16"
        return compute_type

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language_override: str | None = None,
    ) -> STTResult | None:
        """
        音声データをテキストに変換する

        Args:
            audio: float32 numpy配列（モノラル）
            sample_rate: サンプリングレート（16000推奨）
            language_override: 言語を強制指定する場合のコード（例: "it", "ja"）
                              Noneの場合は設定値またはWhisper自動検出を使用

        Returns:
            STTResult（テキスト、検出言語、確率、再生時間）
            音声が空またはエラー時はNone

        Notes:
            入力は16kHz float32を期待。異なる場合は事前にリサンプル必要。
            language_overrideはウェイクワード検出時にイタリア語を
            強制するために使用する。
        """
        if audio is None or len(audio) == 0:
            self._logger.warning("空の音声データが渡された")
            return None

        duration = len(audio) / sample_rate

        # 言語決定: override > config > None(自動検出)
        effective_language = language_override or self._config.language
        self._logger.info(
            "STT処理開始: %.1f秒の音声, lang=%s",
            duration, effective_language or "auto",
        )

        try:
            vad_params = None
            if self._config.vad_filter:
                vad_params = {
                    "min_silence_duration_ms": self._config.vad_min_silence_ms,
                    "speech_pad_ms": self._config.vad_speech_pad_ms,
                }

            segments, info = self._model.transcribe(
                audio,
                beam_size=self._config.beam_size,
                language=effective_language,
                vad_filter=self._config.vad_filter,
                vad_parameters=vad_params,
            )

            # セグメントを結合してテキスト生成
            text_parts: list[str] = []
            for segment in segments:
                text_parts.append(segment.text.strip())

            full_text = " ".join(text_parts).strip()

            if not full_text:
                self._logger.info("STT結果: テキストなし")
                return None

            result = STTResult(
                text=full_text,
                language=info.language,
                language_probability=round(info.language_probability, 3),
                duration_sec=round(duration, 2),
            )
            self._logger.info(
                "STT完了: lang=%s (%.1f%%), text='%s'",
                result.language,
                result.language_probability * 100,
                result.text[:80],
            )
            return result

        except Exception as e:
            self._logger.error("STT処理エラー: %s", e, exc_info=True)
            return None