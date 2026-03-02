"""
CasaAI システムモニタリングモジュール
CPU、RAM、GPU使用率をリアルタイムで取得する
"""
import subprocess
from dataclasses import dataclass, field
from typing import Protocol

import psutil


@dataclass(frozen=True)
class SystemMetrics:
    """システムリソース使用状況の値オブジェクト"""
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_name: str
    gpu_util_percent: float
    gpu_mem_used_mb: float
    gpu_mem_total_mb: float


class SystemMonitorProtocol(Protocol):
    """システムモニターの抽象インターフェース"""
    def get_metrics(self) -> SystemMetrics: ...


class SystemMonitor:
    """
    psutilとnvidia-smiを使用してシステムメトリクスを収集する

    GPU情報はnvidia-smiのCSV出力をパースして取得。
    nvidia-smiが利用不可の場合はデフォルト値を返す。
    """

    def get_metrics(self) -> SystemMetrics:
        """
        現在のシステムメトリクスを取得

        Returns:
            CPU、RAM、GPU使用率を含むSystemMetrics
        """
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        ram_percent = mem.percent
        ram_used = mem.used / (1024 ** 3)
        ram_total = mem.total / (1024 ** 3)

        gpu_name, gpu_util, gpu_mem_used, gpu_mem_total = self._query_gpu()

        return SystemMetrics(
            cpu_percent=cpu,
            ram_percent=ram_percent,
            ram_used_gb=round(ram_used, 1),
            ram_total_gb=round(ram_total, 1),
            gpu_name=gpu_name,
            gpu_util_percent=gpu_util,
            gpu_mem_used_mb=gpu_mem_used,
            gpu_mem_total_mb=gpu_mem_total,
        )

    def _query_gpu(self) -> tuple[str, float, float, float]:
        """
        nvidia-smiからGPU情報を取得

        Returns:
            (GPU名, 使用率%, VRAM使用MB, VRAM総量MB)
            取得失敗時はデフォルト値
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode != 0:
                return ("N/A", 0.0, 0.0, 0.0)

            parts = result.stdout.strip().split(",")
            if len(parts) >= 4:
                return (
                    parts[0].strip(),
                    float(parts[1].strip()),
                    float(parts[2].strip()),
                    float(parts[3].strip()),
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return ("N/A", 0.0, 0.0, 0.0)