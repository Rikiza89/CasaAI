"""
CasaAI ウェイクワード検出モジュール
エネルギーベースの音声活性検出でウェイクワードトリガーを実現する

設計判断:
    外部ウェイクワードライブラリ（Porcupine等）はオフライン制約と
    ライセンス問題があるため、エネルギーベースVADを採用。
    faster-whisperで短い録音をSTTし、キーワード一致で判定する。

    RTX 5050環境での精度向上:
    - ウェイクワードSTTにイタリア語を強制指定
    - ファジーマッチングで音声認識のゆらぎに対応
    - 類似語リストで誤認識パターンをカバー
"""
import logging
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class WakeWordConfig:
    """ウェイクワード設定"""
    keyword: str
    energy_threshold: float
    listen_duration_sec: float
    forced_language: str | None = None


# "casa"の音声認識における一般的な誤認識パターン
# Whisperは短い発話で以下のような変換をしがち
_FUZZY_VARIANTS: dict[str, list[str]] = {
    "casa": [
        "casa", "kasa", "cassa", "caza", "causa",
        "kasah", "casal", "cas",
        # 日本語誤認識パターン
        "かさ", "かーさ", "カサ", "カーサ",
        # 英語誤認識パターン
        "kasar", "kassar",
    ],
}


class WakeWordDetector:
    """
    エネルギーベースVAD + STTによるウェイクワード検出器

    マイクからの音声エネルギーが閾値を超えた場合に
    短い録音を行い、STTで文字起こしした結果にキーワードが
    含まれるかを判定する。

    ファジーマッチングにより、音声認識の揺らぎを吸収する。
    """

    def __init__(
        self,
        config: WakeWordConfig,
        audio_sample_rate: int,
        audio_input_device: int | None,
        logger: logging.Logger,
    ) -> None:
        """
        Args:
            config: ウェイクワード設定
            audio_sample_rate: サンプリングレート
            audio_input_device: 入力デバイスID
            logger: ロガーインスタンス
        """
        self._config = config
        self._sr = audio_sample_rate
        self._device = audio_input_device
        self._logger = logger
        self._keyword_lower = config.keyword.lower().strip()

        # ファジーマッチング用の変換候補を構築
        self._variants = set()
        base_variants = _FUZZY_VARIANTS.get(self._keyword_lower, [self._keyword_lower])
        for v in base_variants:
            self._variants.add(v.lower())
        # キーワード自体も追加
        self._variants.add(self._keyword_lower)

        self._logger.info(
            "ウェイクワード初期化: keyword='%s', variants=%d, forced_lang=%s",
            self._keyword_lower, len(self._variants), config.forced_language,
        )

    @property
    def forced_language(self) -> str | None:
        """ウェイクワードSTT用の強制言語設定"""
        return self._config.forced_language

    def listen_for_energy(self) -> np.ndarray | None:
        """
        マイクを監視し、エネルギーが閾値を超えたら短い音声を録音して返す

        Returns:
            録音された音声データ、または検出なしの場合None

        Notes:
            ブロッキングで0.5秒単位のチャンクを読み、
            エネルギーが閾値を超えたらlisten_duration_sec分の録音を返す。
        """
        chunk_sec = 0.5
        chunk_samples = int(self._sr * chunk_sec)

        try:
            with sd.InputStream(
                samplerate=self._sr,
                channels=1,
                dtype="float32",
                blocksize=chunk_samples,
                device=self._device,
            ) as stream:
                data, _ = stream.read(chunk_samples)
                energy = np.sqrt(np.mean(data ** 2))

                if energy < self._config.energy_threshold:
                    return None

                # エネルギー検出 → 指定秒数分録音
                self._logger.debug("エネルギー検出: %.4f", energy)
                record_samples = int(self._sr * self._config.listen_duration_sec)
                chunks = [data.copy()]
                remaining = record_samples - len(data)

                while remaining > 0:
                    read_size = min(chunk_samples, remaining)
                    d, _ = stream.read(read_size)
                    chunks.append(d.copy())
                    remaining -= len(d)

                audio = np.concatenate(chunks, axis=0).flatten()
                return audio

        except sd.PortAudioError as e:
            self._logger.error("ウェイクワード録音エラー: %s", e)
            return None

    def check_keyword(self, transcribed_text: str) -> bool:
        """
        文字起こし結果にウェイクワードが含まれるかファジー判定

        Args:
            transcribed_text: STTの文字起こし結果

        Returns:
            キーワード（またはその変換候補）が含まれていればTrue

        Notes:
            完全一致ではなく、文字起こし結果の中に変換候補の
            いずれかが部分文字列として含まれるかを判定する。
        """
        text_lower = transcribed_text.lower().strip()

        # 各変換候補で部分一致チェック
        for variant in self._variants:
            if variant in text_lower:
                self._logger.info(
                    "ウェイクワード検出: variant='%s' in text='%s'",
                    variant, text_lower,
                )
                return True

        self._logger.debug("ウェイクワード不一致: text='%s'", text_lower)
        return False