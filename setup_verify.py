"""
CasaAI セットアップ検証スクリプト
全依存関係とハードウェアの状態を確認する
"""
import sys
import subprocess
import importlib
import shutil
from pathlib import Path


def check_python_version() -> bool:
    """Pythonバージョンが3.11以上であることを確認"""
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 11
    status = "OK" if ok else "FAIL"
    print(f"[{status}] Python version: {v.major}.{v.minor}.{v.micro}")
    return ok


def check_cuda() -> bool:
    """CUDA環境の存在を確認（nvccまたはnvidia-smiのCUDAバージョンで判定）"""
    nvcc = shutil.which("nvcc")
    if nvcc:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        print(f"[OK] CUDA Toolkit: {result.stdout.strip().splitlines()[-1]}")
        return True

    # nvccが無くてもドライバのCUDAランタイムで動作可能
    # nvidia-smiからCUDAバージョンを取得して確認
    smi = shutil.which("nvidia-smi")
    if smi:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        driver = result.stdout.strip()
        print(f"[WARN] CUDA Toolkit (nvcc) not in PATH, but driver {driver} found")
        print("       faster-whisper bundles its own CUDA runtime — this is OK")
        print("       Install CUDA Toolkit only if you need nvcc for compilation")
        return True

    print("[FAIL] No CUDA environment detected (no nvcc, no nvidia-smi)")
    return False


def check_nvidia_smi() -> bool:
    """nvidia-smiでGPU状態を確認"""
    smi = shutil.which("nvidia-smi")
    if smi:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        info = result.stdout.strip()
        print(f"[OK] GPU: {info}")
        return True
    print("[FAIL] nvidia-smi not found")
    return False


def check_module(name: str, import_name: str | None = None) -> bool:
    """Pythonモジュールのインポートを確認"""
    mod = import_name or name
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "unknown")
        print(f"[OK] {name}: {ver}")
        return True
    except ImportError as e:
        print(f"[FAIL] {name}: {e}")
        return False


def check_faster_whisper_gpu() -> bool:
    """faster-whisperがCUDAデバイスを使用可能か確認（RTX 50XX対応）"""
    try:
        from faster_whisper import WhisperModel
        # RTX 50XX (Blackwell) ではint8系が非対応、float16を明示指定
        model = WhisperModel("tiny", device="cuda", compute_type="float16")
        del model
        print("[OK] faster-whisper CUDA acceleration available (float16)")
        return True
    except Exception as e:
        err_str = str(e)
        if "CUBLAS_STATUS_NOT_SUPPORTED" in err_str:
            print("[FAIL] faster-whisper CUDA: cuBLAS not supported error")
            print("       RTX 50XX GPUs require float16/float32/bfloat16 compute_type")
            print("       Ensure settings.json has compute_type=float16 (NOT int8)")
        else:
            print(f"[FAIL] faster-whisper CUDA: {e}")
        return False


def check_ollama() -> bool:
    """Ollamaサーバーの応答を確認"""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"[OK] Ollama running. Models: {models}")
            return True
        print(f"[FAIL] Ollama responded with status {resp.status_code}")
        return False
    except Exception as e:
        print(f"[FAIL] Ollama not reachable: {e}")
        return False


def check_piper(settings_path: Path) -> bool:
    """Piper TTS実行ファイルとボイスモデルの存在を確認"""
    import json
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        exe = Path(cfg["tts"]["piper_executable"])
        voice = Path(cfg["tts"]["voice_model"])
        ok = True
        if exe.exists():
            print(f"[OK] Piper executable: {exe}")
        else:
            print(f"[FAIL] Piper executable not found: {exe}")
            ok = False
        if voice.exists():
            print(f"[OK] Piper voice model: {voice}")
        else:
            print(f"[FAIL] Piper voice model not found: {voice}")
            ok = False
        return ok
    except Exception as e:
        print(f"[FAIL] Piper check error: {e}")
        return False


def check_audio_devices() -> bool:
    """オーディオ入出力デバイスの存在を確認"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        inp = sd.query_devices(kind="input")
        out = sd.query_devices(kind="output")
        print(f"[OK] Input device: {inp['name']}")
        print(f"[OK] Output device: {out['name']}")
        return True
    except Exception as e:
        print(f"[FAIL] Audio devices: {e}")
        return False


def main() -> None:
    """全検証を実行し、結果サマリーを表示"""
    print("=" * 60)
    print("CasaAI 🇮🇹 Setup Verification")
    print("=" * 60)

    project_root = Path(__file__).parent
    settings_path = project_root / "config" / "settings.json"

    checks: list[tuple[str, bool]] = []

    checks.append(("Python version", check_python_version()))
    checks.append(("CUDA toolkit", check_cuda()))
    checks.append(("NVIDIA GPU", check_nvidia_smi()))
    checks.append(("numpy", check_module("numpy")))
    checks.append(("sounddevice", check_module("sounddevice")))
    checks.append(("faster-whisper", check_module("faster_whisper")))
    checks.append(("requests", check_module("requests")))
    checks.append(("psutil", check_module("psutil")))
    checks.append(("scipy", check_module("scipy")))
    checks.append(("faster-whisper GPU", check_faster_whisper_gpu()))
    checks.append(("Ollama server", check_ollama()))
    checks.append(("Piper TTS", check_piper(settings_path)))
    checks.append(("Audio devices", check_audio_devices()))

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    print(f"Results: {passed}/{total} checks passed")

    failed = [(name, ok) for name, ok in checks if not ok]
    if failed:
        print("\nFailed checks:")
        for name, _ in failed:
            print(f"  ✗ {name}")
        print("\nFix the above issues before running CasaAI.")
    else:
        print("\n✓ All checks passed. Run: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()