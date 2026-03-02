"""
CasaAI メインオーケストレーター
全コンポーネントを初期化し、音声アシスタントループを制御する

アーキテクチャ:
    UIスレッド（メイン） + アシスタントワーカースレッド + モニタースレッド
    ウェイクワード検出 → 録音 → STT → 言語検出 → LLM → TTS → 再生 → メモリ保存

重要:
    NVIDIA CUDAランタイムDLLのパス解決をimport前に実行する必要がある。
    Windows環境ではos.add_dll_directory()でDLL検索パスを明示追加する。
"""
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

# ====================================================================
# CUDA DLLパス解決（他モジュールimport前に実行必須）
# ctranslate2がcublas64_12.dll等を検索できるよう、
# pip installされたnvidia-*パッケージのlib/binディレクトリを
# DLL検索パスに追加する
# ====================================================================
def _register_nvidia_dll_paths() -> None:
    """
    nvidia-cublas-cu12等のpipパッケージに含まれるDLLパスを
    WindowsのDLL検索パスに登録する

    Notes:
        ctranslate2はC言語レベルでLoadLibrary()を使用してDLLを読み込むため、
        Pythonのos.add_dll_directory()だけでは不十分。
        PATH環境変数への追加が最も確実な方法。
        両方の手法を併用して互換性を最大化する。
    """
    if sys.platform != "win32":
        return

    # venvのsite-packages内のnvidiaディレクトリを探索
    venv_sp = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not venv_sp.is_dir():
        print("[CasaAI] WARNING: nvidia package directory not found")
        return

    # DLLを含むディレクトリを収集
    dll_dirs: list[str] = []
    for bin_dir in venv_sp.rglob("bin"):
        if bin_dir.is_dir() and any(bin_dir.glob("*.dll")):
            dll_dirs.append(str(bin_dir))
    for lib_dir in venv_sp.rglob("lib"):
        if lib_dir.is_dir() and any(lib_dir.glob("*.dll")):
            dll_dirs.append(str(lib_dir))

    if not dll_dirs:
        print("[CasaAI] WARNING: No NVIDIA DLL files found")
        return

    # 方法1: PATH環境変数に追加（ctranslate2のLoadLibrary()対応）
    current_path = os.environ.get("PATH", "")
    new_entries = [d for d in dll_dirs if d not in current_path]
    if new_entries:
        os.environ["PATH"] = ";".join(new_entries) + ";" + current_path

    # 方法2: os.add_dll_directory()も併用（Python側のimport対応）
    for d in dll_dirs:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass

    print(f"[CasaAI] Registered {len(dll_dirs)} NVIDIA DLL directories to PATH")
    for d in dll_dirs:
        print(f"  -> {d}")


# DLLパス登録を最初に実行
_register_nvidia_dll_paths()

# ====================================================================
# 以降、通常のimport（ctranslate2/faster-whisperを含むモジュール）
# ====================================================================
from core.audio_manager import AudioManager, AudioConfig
from core.wake_word import WakeWordDetector, WakeWordConfig
from core.stt_engine import STTEngine, STTConfig
from core.language_detector import LanguageDetector
from core.llm_client import LLMClient, LLMConfig
from core.tts_engine import TTSEngine, TTSConfig
from core.memory_manager import MemoryManager, MemoryConfig
from infrastructure.logger import setup_logger, LogConfig
from infrastructure.system_monitor import SystemMonitor
from ui.main_window import MainWindow, UIConfig, AssistantStatus


class CasaAIApp:
    """
    CasaAI アプリケーションオーケストレーター

    全コンポーネントのライフサイクルを管理し、
    音声アシスタントのメインループを制御する。
    UIはメインスレッドで実行し、アシスタントロジックは
    バックグラウンドスレッドで非同期に動作する。
    """

    def __init__(self, config_path: str = "config/settings.json") -> None:
        """
        Args:
            config_path: 設定ファイルのパス

        Raises:
            FileNotFoundError: 設定ファイルが存在しない場合
            json.JSONDecodeError: 設定ファイルの形式が不正な場合
        """
        self._running = False
        self._config = self._load_config(config_path)

        # ログ初期化（最優先）
        log_cfg = LogConfig(**self._config["logging"])
        self._logger = setup_logger("CasaAI", log_cfg)
        self._logger.info("=" * 60)
        self._logger.info("CasaAI 起動開始")
        self._logger.info("=" * 60)

        # コンポーネント初期化
        self._init_components()

    def _load_config(self, path: str) -> dict:
        """
        設定ファイルを読み込んで返す

        Args:
            path: JSONファイルのパス

        Returns:
            設定辞書

        Raises:
            FileNotFoundError: ファイルが存在しない場合
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _init_components(self) -> None:
        """全コンポーネントを設定に基づいて初期化する"""
        cfg = self._config

        # オーディオマネージャー
        audio_cfg = AudioConfig(**cfg["audio"])
        self._audio = AudioManager(audio_cfg, setup_logger("Audio", LogConfig(**cfg["logging"])))

        # ウェイクワード検出器
        wake_cfg = WakeWordConfig(**cfg["wake_word"])
        self._wake = WakeWordDetector(
            config=wake_cfg,
            audio_sample_rate=cfg["audio"]["sample_rate"],
            audio_input_device=cfg["audio"]["input_device"],
            logger=setup_logger("WakeWord", LogConfig(**cfg["logging"])),
        )

        # STTエンジン（GPU）
        stt_raw = cfg["stt"]
        vad_params = stt_raw.get("vad_parameters", {})
        stt_cfg = STTConfig(
            model_size=stt_raw["model_size"],
            device=stt_raw["device"],
            compute_type=stt_raw["compute_type"],
            beam_size=stt_raw["beam_size"],
            language=stt_raw["language"],
            vad_filter=stt_raw["vad_filter"],
            vad_min_silence_ms=vad_params.get("min_silence_duration_ms", 500),
            vad_speech_pad_ms=vad_params.get("speech_pad_ms", 200),
        )
        self._stt = STTEngine(stt_cfg, setup_logger("STT", LogConfig(**cfg["logging"])))

        # 言語検出器
        self._lang_detector = LanguageDetector(
            setup_logger("LangDetect", LogConfig(**cfg["logging"]))
        )

        # LLMクライアント
        llm_cfg = LLMConfig(**cfg["llm"])
        self._llm = LLMClient(llm_cfg, setup_logger("LLM", LogConfig(**cfg["logging"])))

        # TTSエンジン
        tts_cfg = TTSConfig(**cfg["tts"])
        self._tts = TTSEngine(tts_cfg, setup_logger("TTS", LogConfig(**cfg["logging"])))

        # メモリマネージャー
        mem_cfg = MemoryConfig(**cfg["memory"])
        self._memory = MemoryManager(mem_cfg, setup_logger("Memory", LogConfig(**cfg["logging"])))

        # システムモニター
        self._monitor = SystemMonitor()

        # UI
        ui_cfg = UIConfig(**cfg["ui"])
        self._window = MainWindow(
            config=ui_cfg,
            logger=setup_logger("UI", LogConfig(**cfg["logging"])),
            on_exit=self._shutdown,
        )

        self._logger.info("全コンポーネント初期化完了")

    def run(self) -> None:
        """
        アプリケーションを起動する

        UIメインループはメインスレッドで実行。
        アシスタントワーカーとシステムモニターは別スレッドで起動。
        """
        self._running = True

        # Ollama死活確認
        if not self._llm.health_check():
            self._logger.error("Ollamaサーバーに接続できません。ollama serve を実行してください。")
            self._window.update_state(
                status=AssistantStatus.ERROR,
                assistant_text="Errore: Ollama non raggiungibile. Avvia 'ollama serve'.",
            )
        else:
            self._logger.info("Ollamaサーバー接続確認OK")

        # バックグラウンドスレッド起動
        assistant_thread = threading.Thread(
            target=self._assistant_loop, daemon=True, name="AssistantWorker"
        )
        assistant_thread.start()

        monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="SystemMonitor"
        )
        monitor_thread.start()

        # 初期状態を設定
        self._window.update_state(
            status=AssistantStatus.LISTENING_WAKE,
            memory_count=self._memory.get_entry_count(),
        )

        # UIメインループ（ブロッキング）
        self._window.start_mainloop()

    def _assistant_loop(self) -> None:
        """
        アシスタントのメインループ

        ウェイクワード検出 → 録音 → STT → 言語検出 → LLM → TTS → 再生
        のサイクルを繰り返す。

        Notes:
            例外はキャッチしてログに記録し、ループを継続する。
            self._runningがFalseになると終了。
        """
        # 起動直後の安定化待ち
        time.sleep(2.0)

        while self._running:
            try:
                self._single_interaction_cycle()
            except Exception as e:
                self._logger.error("アシスタントループ例外: %s", e, exc_info=True)
                self._window.update_state(
                    status=AssistantStatus.ERROR,
                    assistant_text=f"Errore interno: {e}",
                )
                time.sleep(2.0)

    def _single_interaction_cycle(self) -> None:
        """
        1回のインタラクションサイクルを実行する

        ウェイクワード検出から応答再生までの全フローを含む。
        """
        # Phase 1: ウェイクワード待機
        self._window.update_state(status=AssistantStatus.LISTENING_WAKE)
        wake_audio = self._wake.listen_for_energy()

        if wake_audio is None:
            time.sleep(0.1)
            return

        # ウェイクワード音声をSTTで確認（イタリア語強制で精度向上）
        wake_result = self._stt.transcribe(
            wake_audio,
            self._config["audio"]["sample_rate"],
            language_override=self._wake.forced_language,
        )
        if wake_result is None:
            return

        if not self._wake.check_keyword(wake_result.text):
            return

        # Phase 2: ユーザー発話の録音
        self._logger.info("ウェイクワード確認、ユーザー発話録音開始")
        self._window.update_state(status=AssistantStatus.LISTENING)
        time.sleep(0.3)  # ウェイクワード音声の残響回避

        user_audio = self._audio.record_until_silence()
        if user_audio is None:
            self._logger.info("ユーザー発話なし、待機に戻る")
            return

        # Phase 3: STT
        self._window.update_state(status=AssistantStatus.THINKING)
        stt_result = self._stt.transcribe(user_audio, self._config["audio"]["sample_rate"])
        if stt_result is None:
            self._window.update_state(
                assistant_text="Non ho capito. Riprova per favore."
            )
            return

        # STT結果の品質チェック: 短すぎるテキストや低信頼度は棄却
        if len(stt_result.text.strip()) < 2:
            self._logger.info("STT結果が短すぎるため棄却: '%s'", stt_result.text)
            self._window.update_state(
                assistant_text="Non ho capito bene. Riprova per favore."
            )
            return

        # Phase 4: 言語検出
        detected_lang = self._lang_detector.detect(stt_result.language)
        lang_display = self._lang_detector.get_display_name(detected_lang)

        self._window.update_state(
            detected_language=lang_display,
            user_text=stt_result.text,
        )

        # Phase 5: LLM応答生成
        history = self._memory.get_history_for_llm()
        llm_result = self._llm.generate_response(
            user_text=stt_result.text,
            detected_language=detected_lang,
            conversation_history=history,
        )

        if llm_result is None:
            fallback = "Mi dispiace, non sono riuscito a elaborare la risposta. Riprova."
            self._window.update_state(assistant_text=fallback)
            # フォールバック応答をTTSで再生
            wav = self._tts.synthesize(fallback)
            if wav:
                self._window.update_state(status=AssistantStatus.SPEAKING)
                self._audio.play_wav_bytes(wav)
            return

        self._window.update_state(assistant_text=llm_result.text)

        # Phase 6: TTS合成と再生
        wav_data = self._tts.synthesize(llm_result.text)
        if wav_data:
            self._window.update_state(status=AssistantStatus.SPEAKING)
            self._audio.play_wav_bytes(wav_data)

        # Phase 7: メモリ保存
        self._memory.add_entry(
            user_text=stt_result.text,
            user_language=detected_lang.value,
            assistant_text=llm_result.text,
        )
        self._window.update_state(memory_count=self._memory.get_entry_count())

    def _monitor_loop(self) -> None:
        """
        システムモニタリングループ

        定期的にCPU、RAM、GPU使用率を取得してUIに反映する。
        """
        while self._running:
            try:
                metrics = self._monitor.get_metrics()
                self._window.update_state(
                    cpu_percent=metrics.cpu_percent,
                    ram_percent=metrics.ram_percent,
                    ram_used_gb=metrics.ram_used_gb,
                    ram_total_gb=metrics.ram_total_gb,
                    gpu_name=metrics.gpu_name,
                    gpu_util_percent=metrics.gpu_util_percent,
                    gpu_mem_used_mb=metrics.gpu_mem_used_mb,
                    gpu_mem_total_mb=metrics.gpu_mem_total_mb,
                )
            except Exception as e:
                self._logger.error("モニタリングエラー: %s", e)
            time.sleep(2.0)

    def _shutdown(self) -> None:
        """アプリケーションのシャットダウン処理"""
        self._logger.info("シャットダウン開始")
        self._running = False
        self._logger.info("CasaAI 終了完了")


def main() -> None:
    """エントリーポイント"""
    try:
        app = CasaAIApp()
        app.run()
    except FileNotFoundError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()