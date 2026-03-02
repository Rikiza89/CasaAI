"""
CasaAI ログ基盤モジュール
構造化ログをファイルとコンソールに出力する
"""
import logging
import logging.handlers
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class LogConfig:
    """ログ設定の値オブジェクト"""
    log_dir: str
    max_file_size_mb: int
    backup_count: int
    level: str


_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def setup_logger(name: str, config: LogConfig) -> logging.Logger:
    """
    名前付きロガーを設定して返す

    Args:
        name: ロガー名（モジュール識別用）
        config: ログ設定

    Returns:
        設定済みのLoggerインスタンス
    """
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    # 既にハンドラが設定済みなら再設定しない
    if logger.handlers:
        return logger

    level = getattr(logging, config.level.upper(), logging.INFO)
    logger.setLevel(level)

    # ファイルハンドラ（ローテーション付き）
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "casaai.log",
        maxBytes=config.max_file_size_mb * 1024 * 1024,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(_FORMATTER)
    logger.addHandler(file_handler)

    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(_FORMATTER)
    logger.addHandler(console_handler)

    return logger