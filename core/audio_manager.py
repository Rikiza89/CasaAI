"""
CasaAI 音声入出力管理モジュール
マイク録音とスピーカー再生を制御する
"""
import io
import logging
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class AudioConfig:
    """オーディオ設定の値オブジェクト"""
    sample_rate: int
    channels: int
    dtype: str
    block_size: int
    silence_threshold: float
    silence_duration_sec: float
    max_record_sec: float
    input_device: int | None
    output_device: int | None


class AudioManagerProtocol(Protocol):
    """音声管理の抽象インターフェース"""
    def record_until_silence(self) -> np.ndarray | None: ...
    def play_wav_bytes(self, wav_data: bytes) -> None: ...


class AudioManager:
    """
    音声録音・再生を管理するサービス

    無音検出による自動録音停止機能を持つ。
    録音はブロッキングで実行し、完了後にnumpy配列を返す。
    """

    def __init__(self, config: AudioConfig, logger: logging.Logger) -> None:
        """
        Args:
            config: オーディオ設定
            logger: ロガーインスタンス
        """
        self._config = config
        self._logger = logger

    def record_until_silence(self) -> np.ndarray | None:
        """
        マイクから音声を録音し、無音検出で自動停止する

        Returns:
            録音された音声データ（float32 numpy配列）
            録音失敗または音声なしの場合はNone

        Notes:
            silence_duration_sec 秒間の無音で録音を停止。
            max_record_sec を超えた場合も停止。
        """
        sr = self._config.sample_rate
        ch = self._config.channels
        threshold = self._config.silence_threshold
        max_frames = int(self._config.max_record_sec * sr)
        silence_frames = int(self._config.silence_duration_sec * sr)

        self._logger.info("録音開始 (閾値=%.4f, 最大=%.1f秒)", threshold, self._config.max_record_sec)

        recorded_chunks: list[np.ndarray] = []
        silent_count = 0
        speech_detected = False

        try:
            with sd.InputStream(
                samplerate=sr,
                channels=ch,
                dtype=self._config.dtype,
                blocksize=self._config.block_size,
                device=self._config.input_device,
            ) as stream:
                total_frames = 0
                while total_frames < max_frames:
                    data, overflowed = stream.read(self._config.block_size)
                    if overflowed:
                        self._logger.warning("オーディオバッファオーバーフロー検出")

                    recorded_chunks.append(data.copy())
                    total_frames += len(data)

                    # 音声エネルギー計算
                    energy = np.sqrt(np.mean(data ** 2))

                    if energy > threshold:
                        speech_detected = True
                        silent_count = 0
                    else:
                        if speech_detected:
                            silent_count += len(data)

                    # 発話後の無音が閾値を超えたら停止
                    if speech_detected and silent_count >= silence_frames:
                        self._logger.info("無音検出により録音停止 (%.1f秒)", total_frames / sr)
                        break

        except sd.PortAudioError as e:
            self._logger.error("オーディオ録音エラー: %s", e)
            return None

        if not speech_detected:
            self._logger.debug("音声未検出")
            return None

        audio = np.concatenate(recorded_chunks, axis=0).flatten()
        self._logger.info("録音完了: %d サンプル (%.1f秒)", len(audio), len(audio) / sr)
        return audio

    def play_wav_bytes(self, wav_data: bytes) -> None:
        """
        WAVバイトデータをスピーカーで再生する

        Args:
            wav_data: WAV形式のバイトデータ

        Raises:
            再生エラーはログに記録し、例外は抑制
        """
        try:
            buf = io.BytesIO(wav_data)
            with wave.open(buf, "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())
                # 16bit PCM → float32変換
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                if ch > 1:
                    audio = audio.reshape(-1, ch)
                sd.play(audio, samplerate=sr, device=self._config.output_device)
                sd.wait()
                self._logger.info("音声再生完了")
        except Exception as e:
            self._logger.error("音声再生エラー: %s", e)