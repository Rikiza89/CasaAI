"""
CasaAI メインUIウィンドウ
フルスクリーンのダークテーマTkinterインターフェース
"""
import logging
import tkinter as tk
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class AssistantStatus(Enum):
    """アシスタントの状態を表す列挙型"""
    INITIALIZING = "Initializing..."
    LISTENING_WAKE = "🎤 Waiting for wake word..."
    LISTENING = "🎤 Listening..."
    THINKING = "🧠 Thinking..."
    SPEAKING = "🔊 Speaking..."
    ERROR = "⚠️ Error"
    IDLE = "💤 Idle"


@dataclass
class UIState:
    """UIに表示する状態データ"""
    status: AssistantStatus = AssistantStatus.INITIALIZING
    detected_language: str = ""
    user_text: str = ""
    assistant_text: str = ""
    memory_count: int = 0
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_name: str = "N/A"
    gpu_util_percent: float = 0.0
    gpu_mem_used_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0


@dataclass(frozen=True)
class UIConfig:
    """UI設定の値オブジェクト"""
    bg_color: str
    fg_color: str
    accent_color: str
    warning_color: str
    error_color: str
    font_family: str
    title_font_size: int
    label_font_size: int
    content_font_size: int
    refresh_interval_ms: int


class MainWindow:
    """
    CasaAIのフルスクリーンダークテーマUI

    Tkinterを使用してシステム状態をリアルタイム表示する。
    スレッドセーフな状態更新メカニズムを提供し、
    バックグラウンドスレッドからの安全なUI更新を実現。
    """

    def __init__(
        self,
        config: UIConfig,
        logger: logging.Logger,
        on_exit: Callable[[], None] | None = None,
    ) -> None:
        """
        Args:
            config: UI設定
            logger: ロガーインスタンス
            on_exit: アプリ終了時に呼ばれるコールバック
        """
        self._config = config
        self._logger = logger
        self._on_exit = on_exit
        self._state = UIState()

        self._root = tk.Tk()
        self._root.title("CasaAI 🇮🇹")
        self._root.configure(bg=config.bg_color)
        self._root.attributes("-fullscreen", True)

        # キーボードショートカット
        self._root.bind("<Escape>", lambda e: self._exit())
        self._root.bind("<F4>", lambda e: self._exit())

        # フォント定義
        self._title_font = (config.font_family, config.title_font_size, "bold")
        self._label_font = (config.font_family, config.label_font_size, "bold")
        self._content_font = (config.font_family, config.content_font_size)
        self._small_font = (config.font_family, config.content_font_size - 2)

        # UIコンポーネントの構築
        self._labels: dict[str, tk.Label] = {}
        self._build_ui()

        self._logger.info("UIウィンドウ初期化完了")

    def _build_ui(self) -> None:
        """UIレイアウトを構築する"""
        bg = self._config.bg_color
        fg = self._config.fg_color
        accent = self._config.accent_color

        # メインフレーム（中央寄せ用パディング付き）
        main_frame = tk.Frame(self._root, bg=bg, padx=60, pady=40)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # タイトル
        title_label = tk.Label(
            main_frame,
            text="CasaAI 🇮🇹",
            font=self._title_font,
            fg=accent,
            bg=bg,
        )
        title_label.pack(anchor=tk.W, pady=(0, 20))

        # セパレーター
        sep = tk.Frame(main_frame, height=2, bg=accent)
        sep.pack(fill=tk.X, pady=(0, 20))

        # 状態表示行を生成するヘルパー
        def add_row(parent: tk.Frame, label_key: str, label_text: str) -> None:
            """ラベル-値ペアの行を追加"""
            row = tk.Frame(parent, bg=bg)
            row.pack(fill=tk.X, pady=4)

            lbl = tk.Label(
                row, text=label_text, font=self._label_font,
                fg=self._config.warning_color, bg=bg, width=22, anchor=tk.W,
            )
            lbl.pack(side=tk.LEFT)

            val = tk.Label(
                row, text="", font=self._content_font,
                fg=fg, bg=bg, anchor=tk.W, wraplength=900, justify=tk.LEFT,
            )
            val.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._labels[label_key] = val

        # 各表示行
        add_row(main_frame, "status", "Status:")
        add_row(main_frame, "language", "Detected Language:")
        add_row(main_frame, "user_text", "User Said:")
        add_row(main_frame, "assistant_text", "Assistant Response:")
        add_row(main_frame, "memory", "Memory Entries:")

        # セパレーター
        sep2 = tk.Frame(main_frame, height=1, bg="#333333")
        sep2.pack(fill=tk.X, pady=(20, 10))

        # システムモニター表示
        monitor_label = tk.Label(
            main_frame, text="System Monitor",
            font=self._label_font, fg="#888888", bg=bg,
        )
        monitor_label.pack(anchor=tk.W, pady=(0, 8))

        add_row(main_frame, "cpu", "CPU:")
        add_row(main_frame, "ram", "RAM:")
        add_row(main_frame, "gpu", "GPU:")

        # フッター
        footer = tk.Label(
            main_frame,
            text="Press ESC or F4 to exit  |  Say 'Casa' to activate",
            font=self._small_font,
            fg="#555555",
            bg=bg,
        )
        footer.pack(side=tk.BOTTOM, pady=(20, 0))

    def update_state(self, **kwargs: object) -> None:
        """
        スレッドセーフにUI状態を更新する

        Args:
            **kwargs: UIStateのフィールド名と値

        Notes:
            Tkinterのafter()を使用してメインスレッドで実行する。
            バックグラウンドスレッドから安全に呼び出し可能。
        """
        def _apply() -> None:
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
            self._refresh_display()

        try:
            self._root.after(0, _apply)
        except tk.TclError:
            # ウィンドウが既に閉じている場合
            pass

    def _refresh_display(self) -> None:
        """現在の状態をUIラベルに反映する"""
        s = self._state

        # ステータスの色を状態に応じて変更
        status_colors = {
            AssistantStatus.LISTENING_WAKE: self._config.accent_color,
            AssistantStatus.LISTENING: "#00E5FF",
            AssistantStatus.THINKING: "#FFD600",
            AssistantStatus.SPEAKING: "#AA00FF",
            AssistantStatus.ERROR: self._config.error_color,
        }
        status_color = status_colors.get(s.status, self._config.fg_color)

        self._labels["status"].configure(text=s.status.value, fg=status_color)
        self._labels["language"].configure(text=s.detected_language or "—")
        self._labels["user_text"].configure(text=s.user_text or "—")
        self._labels["assistant_text"].configure(text=s.assistant_text or "—")
        self._labels["memory"].configure(text=str(s.memory_count))

        self._labels["cpu"].configure(text=f"{s.cpu_percent:.1f}%")
        self._labels["ram"].configure(
            text=f"{s.ram_percent:.1f}% ({s.ram_used_gb:.1f} / {s.ram_total_gb:.1f} GB)"
        )
        gpu_text = (
            f"{s.gpu_name} | Util: {s.gpu_util_percent:.0f}% | "
            f"VRAM: {s.gpu_mem_used_mb:.0f} / {s.gpu_mem_total_mb:.0f} MB"
        )
        self._labels["gpu"].configure(text=gpu_text)

    def start_mainloop(self) -> None:
        """Tkinterメインループを開始"""
        self._logger.info("UIメインループ開始")
        self._root.mainloop()

    def _exit(self) -> None:
        """アプリケーション終了処理"""
        self._logger.info("UI終了リクエスト")
        if self._on_exit:
            self._on_exit()
        try:
            self._root.quit()
            self._root.destroy()
        except tk.TclError:
            pass

    @property
    def root(self) -> tk.Tk:
        """Tkinterルートウィンドウへのアクセス"""
        return self._root