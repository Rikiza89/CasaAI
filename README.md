# CasaAI 🇮🇹

**Offline AI Voice Assistant for Windows 11 — Speaks Italian, Understands Japanese, English & Italian**

CasaAI is a fully offline, GPU-accelerated voice assistant that runs entirely on your local machine. It listens for a wake word, transcribes speech in multiple languages, and always responds in spoken Italian.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia&logoColor=white)
![Ollama](https://img.shields.io/badge/LLM-Ollama-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078D6?logo=windows&logoColor=white)

---

## ✨ Features

- **100% Offline** — No cloud APIs, no internet required after setup
- **GPU Accelerated** — STT runs on CUDA via faster-whisper (CTranslate2)
- **Multilingual Input** — Accepts spoken Japanese 🇯🇵, English 🇬🇧, and Italian 🇮🇹
- **Italian Output** — Always responds in spoken Italian with Piper TTS
- **Translation** — Automatically translates Japanese/English input to Italian
- **Grammar Correction** — Gently corrects Italian grammar mistakes
- **Wake Word Activation** — Say "Casa" to activate (fuzzy matching for accuracy)
- **Conversation Memory** — Remembers last 10 exchanges (JSON persistence)
- **Dark Fullscreen UI** — Tkinter-based status dashboard with system monitoring
- **Family Safe** — Content filtering built into the system prompt
- **RTX 50XX Ready** — Includes Blackwell architecture compatibility fixes

## 🏗️ Architecture

```
Wake Word → Record Audio → STT (GPU) → Language Detection → LLM (Ollama) 
→ Italian Response → TTS (Piper) → Speaker Output → Memory Save → UI Update → Loop
```

```
CasaAI/
├── config/
│   └── settings.json          # Externalized configuration
├── core/                      # Domain logic layer
│   ├── audio_manager.py       # Microphone recording & speaker playback
│   ├── wake_word.py           # Wake word detection with fuzzy matching
│   ├── stt_engine.py          # faster-whisper GPU speech-to-text
│   ├── language_detector.py   # IT/JA/EN language normalization
│   ├── llm_client.py          # Ollama REST API client
│   ├── tts_engine.py          # Piper offline text-to-speech
│   └── memory_manager.py      # JSON conversation persistence
├── infrastructure/            # Infrastructure layer
│   ├── logger.py              # Structured rotating file logger
│   └── system_monitor.py      # CPU/RAM/GPU metrics (psutil + nvidia-smi)
├── ui/                        # Interface layer
│   └── main_window.py         # Fullscreen dark Tkinter UI
├── main.py                    # Application orchestrator
├── setup_verify.py            # Pre-flight hardware & dependency checker
└── requirements.txt           # Python dependencies
```

**Design Principles:** Clean Architecture, dependency injection, typed configuration, Protocol-based abstractions, explicit error handling, no global mutable state.

## 📋 Requirements (tested with)

| Component | Requirement |
|-----------|-------------|
| OS | Windows 11 |
| Python | 3.11.x |
| GPU | NVIDIA with 8GB+ VRAM |
| GPU Driver | 570.xx+ (for RTX 5050) |
| Ollama | Latest version |
| Piper | v2023.11.14-2 (Windows binary) |
| Audio | Microphone + Speakers |
| Disk | ~15GB free space |

## 🚀 Installation

### 1. Clone Repository

```powershell
git clone https://github.com/Rikiza89/CasaAI.git
cd CasaAI
```

### 2. Create Python Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
install requirements will require time because of cuda libraries.

### 3. Install Ollama & Pull Model

Download [Ollama for Windows](https://ollama.com/download/windows), then:

```powershell
ollama pull qwen2.5:7b-instruct-q4_K_M
```

### 4. Install Piper TTS

```powershell
# Create directories
New-Item -ItemType Directory -Force -Path "piper\voices"

# Download Piper Windows binary
Invoke-WebRequest -Uri "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip" -OutFile "piper\piper.zip"
Expand-Archive -Path "piper\piper.zip" -DestinationPath "piper" -Force
Remove-Item "piper\piper.zip"

# Download Italian voice
Invoke-WebRequest -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx" -OutFile "piper\voices\it_IT-riccardo-x_low.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx.json" -OutFile "piper\voices\it_IT-riccardo-x_low.onnx.json"
```

### 5. Create Runtime Directories

```powershell
New-Item -ItemType Directory -Force -Path "logs"
New-Item -ItemType Directory -Force -Path "memory"
```

### 6. Configure Paths

Edit `config/settings.json` and update the Piper paths to match your installation:

```json
"tts": {
    "piper_executable": "C:\\CasaAI\\piper\\piper\\piper.exe",
    "voice_model": "C:\\CasaAI\\piper\\voices\\it_IT-riccardo-x_low.onnx"
}
```

### 7. Verify Setup

```bash
python setup_verify.py
```

## ▶️ Usage

```bash
# Make sure Ollama is running
ollama serve

# In another terminal
cd CasaAI
.venv\Scripts\activate
python main.py
```

### Interaction Flow

1. The fullscreen UI appears with status "Waiting for wake word..."
2. Say **"Casa"** to activate
3. Speak your message in Italian, Japanese, or English
4. CasaAI responds in spoken Italian
5. Press **ESC** or **F4** to exit

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `ESC` | Exit application |
| `F4` | Exit application |

## ⚠️ RTX 50XX (Blackwell) Notes

RTX 5050/5060/5070/5080/5090 GPUs have a known CTranslate2/cuBLAS incompatibility with `int8` compute types. CasaAI is pre-configured with `float16` which works correctly on these GPUs.

**Do NOT change** `compute_type` to `int8`, `int8_float16`, `int8_float32`, `int8_bfloat16`, or `auto` in `settings.json`.

## 🔧 Configuration

All configuration is in `config/settings.json`. Key settings:

| Setting | Description | Default |
|---------|-------------|---------|
| `stt.model_size` | Whisper model size | `"small"` |
| `stt.compute_type` | GPU compute type | `"float16"` |
| `llm.model` | Ollama model name | `"qwen2.5:7b-instruct-q4_K_M"` |
| `llm.max_tokens` | Max response length | `300` |
| `wake_word.keyword` | Wake word | `"casa"` |
| `audio.silence_threshold` | Mic sensitivity | `0.015` |
| `memory.max_entries` | Conversation history size | `10` |

### Reducing VRAM Usage

If you experience GPU memory issues:

- Change `stt.model_size` to `"tiny"` or `"base"`
- Use a smaller LLM: `qwen2.5:3b-instruct-q4_K_M`

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| `cublas64_12.dll not found` | Run `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` |
| `cuBLAS_STATUS_NOT_SUPPORTED` | Set `compute_type` to `"float16"` (RTX 50XX issue) |
| Ollama not responding | Run `ollama serve` in a separate terminal |
| Microphone not detected | Check `audio.input_device` in settings.json |
| Wake word not recognized | Try speaking "Casa" clearly, or lower `energy_threshold` |
| TTS encoding error | Already fixed — Piper receives UTF-8 bytes |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 + CUDA) |
| LLM | [Ollama](https://ollama.com/) + Qwen 2.5 7B Instruct |
| TTS | [Piper](https://github.com/rhasspy/piper) (Italian voice) |
| UI | Tkinter (fullscreen dark theme) |
| Audio | sounddevice + numpy |
| Monitoring | psutil + nvidia-smi |

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

- [Anthropic Claude](https://claude.ai) — Architecture design and code generation
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — GPU-accelerated speech recognition
- [Piper](https://github.com/rhasspy/piper) — High-quality offline TTS
- [Ollama](https://ollama.com/) — Local LLM inference
- [Qwen](https://github.com/QwenLM/Qwen2.5) — Multilingual language model
