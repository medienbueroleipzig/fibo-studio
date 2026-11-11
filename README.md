# 🎨 FIBO Studio

> **Universal High-Performance Creative Interface for BRIA FIBO 2.0**  
> *Native Apple Silicon MLX & NVIDIA CUDA Acceleration • Multi-Reference Image Editing • Batch / Bulk Processing • 1-Step Diffusion Upscaling • Vision AI*

[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20%7C%20M2%20%7C%20M3%20%7C%20M4-black?style=for-the-badge&logo=apple)](https://www.apple.com)
[![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA%20CUDA-RTX%2030%20%7C%2040%20%7C%2050%20Series-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![MLX](https://img.shields.io/badge/Engine-Apple%20MLX-orange?style=for-the-badge)](https://github.com/ml-explore/mlx)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![Gradio](https://img.shields.io/badge/UI-Gradio%20Blocks-critical?style=for-the-badge&logo=gradio)](https://gradio.app)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)](LICENSE)

---

## 💡 Why FIBO Studio? (The Antidote to Node-Spaghetti & Framework Bloat)

Modern AI image generation has become cluttered with overwhelming complexity:
* **ComfyUI** requires building and maintaining 60-node spaghetti graphs. Workflows break constantly when custom nodes or community packages update, and wiring reference images to latent conditioning requires deep technical knowledge.
* **WanGP, Pinokio & Automatic1111** drag in 40+ GB dependency mazes, conflicting virtual environments, and fragile setups just to run a quick image-to-image edit.

**FIBO Studio was created out of personal necessity to solve this exact frustration.**

Instead of spending hours untangling node wires, configuring tensor routing, or debugging Python package conflicts, FIBO Studio provides a **focused, lightning-fast, dedicated creative workspace built exclusively for BRIA FIBO 2.0**:
* **1-Click Iterative Workflow:** Directly send generated results back to Reference Image 1 with one click (*"Ergebnis als Referenz 1 übernehmen"*) to build multi-turn artistic progressions seamlessly.
* **Distraction-Free Professional UI:** Clean, ultra-crisp 0px dark theme designed for creators, digital artists, retouchers, and photographers.
* **Native Platform Optimization:** No bloated emulation layers. Runs natively on **Apple Silicon Metal (MLX / MPS)** on macOS, and on **NVIDIA CUDA (SageAttention / FlashAttention)** on Linux and Windows.

---

## 🚀 Key Features

### 1. ✏️ Multi-Reference Image Editing (FIBO 2 Turbo)
* **4-Step Turbo Diffusion:** Generates studio-grade editorial modifications in just 4 inference steps.
* **Up to 5 Reference Slots:** Combine subjects, style cues, backgrounds, and specific items with natural prompt tags (`<image_1>`, `<image_2>`, etc.).
* **Inpainting & Masking:** Draw surgical localized edit masks directly on the canvas or upload existing masks.
* **Persistent Custom Presets / Modifiers:** Save your go-to style transformations (e.g. *Comic2Photo*, *1970s Kodachrome*, *Cyberpunk Neon*) permanently with instant 1-click prompt application.

### 2. 📦 Production Batch & Bulk Processing (Stapelverarbeitung)
Designed specifically for real-world projects such as transforming an entire comic book into photographic realism, batch photo-retouching, or mass e-commerce product restyling:
* **Folder Scanning & Drag-and-Drop:** Pick an entire local directory or drop dozens of images at once.
* **Interactive Thumbnail Deck:** Live thumbnail preview with individual remove buttons (`✕`) to prune the queue before processing.
* **Smart EXIF Auto-Transpose:** Never suffers from sideways or upside-down smartphone/DSLR photos. Automatically corrects EXIF orientation flags (`Orientation: 6/8`) so reference images always match the canvas upright.
* **Adaptive Dynamic Aspect Ratios:** Select *"Original Aspect Ratio"* to have each image in the batch automatically generated at its native aspect ratio (normalized to ~1 Megapixel on a strict 16-pixel grid).
* **Abortable Queue:** Hit **"🛑 Stop"** at any point to cancel the running batch immediately.

### 3. 🎨 Text-to-Image Generation
* **Turbo (4 Steps) & Base/Dev (30 Steps):** Choose between lightning-fast draft iteration and studio precision.
* **VLM Prompt Expansion:** Automatically enrich brief ideas with descriptive photography prompt engineering via Apple MLX.
* **Preset & Custom Aspect Ratios:** 1:1 Square, 16:9 Landscape, 9:16 Portrait, 21:9 Ultrawide, and custom dimensions.

### 4. ✨ Multi-Tier Post-Processing & Upscaling
Integrated directly beneath generation outputs:
* **Lanczos Instant Resizer (0.01s):** Instant 1.5x / 2.0x / 4.0x scaling with adaptive **Unsharp Mask** edge restoration. Consumes 0 MB VRAM.
* **SeedVR2 (Native Apple MLX Diffusion):** 1-step diffusion super-resolution (up to 4K) specifically compiled for Apple Silicon.
* **Real-ESRGAN:** Neural texture restoration for fabrics, architectural lines, and photographic detail.
* **👤 Face Refiner:** Restores crisp eyes, lips, skin pores, and facial micro-contrast.

### 5. 👁️ Vision AI & WD-Tagger Studio
* **MLX Vision Language Model (Qwen2-VL 4-Bit):** Ultra-fast scene analysis and natural language captioning running at **~110 tokens/sec** on Apple MLX.
* **Deep Danbooru / WD-Tagger (SwinV2):** GPU-accelerated tag classification in **<0.4 seconds** on Apple MPS with confidence threshold filtering.
* **Direct Transfer:** Inject generated captions or detected tags directly into your Edit or T2I prompts.

### 6. 🔍 Multi-Standard PNG Metadata & Smart History Recovery
* **Universal Standards Embedded:** Writes `parameters` (A1111/Civitai), `prompt` (ComfyUI graph), `fibo_meta` (JSON), EXIF `UserComment`, Adobe XMP, and `.metadata.json` sidecars.
* **History Hash Recovery:** Even if you paste an image via macOS clipboard (`Cmd+V`) where browsers strip EXIF headers, FIBO Studio matches the structural thumbnail hash against local outputs to recover 100% of the original prompt, seed, and parameters.

---

## ⚡ Hardware Architecture & Performance Tiers

FIBO Studio dynamically detects your hardware and loads the optimal backend:

```
                      ┌─────────────────────────────────┐
                      │    FIBO Studio Universal App    │
                      └────────────────┬────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────────────┐             ┌───────────────────────────────────┐
│     macOS (Apple Silicon)     │             │      Linux & Windows (NVIDIA)     │
├───────────────────────────────┤             ├───────────────────────────────────┤
│ • Native Apple MLX (mflux)    │             │ • PyTorch 2.x + CUDA 12.x / cuDNN │
│ • Metal Performance Shaders   │             │ • SageAttention 2 / FlashAttention│
│ • Unified Memory Zero-Copy    │             │ • Diffusers / bfloat16 / fp16     │
│ • Quantized 8-Bit / 4-Bit     │             │ • RTX 30 / 40 / 50 Series Support │
└───────────────────────────────┘             └───────────────────────────────────┘
```

### Apple Silicon: Understanding the Memory Bandwidth Bottleneck

Diffusion Transformer models (like FIBO 2.0 / FLUX architectures) are heavily **memory-bandwidth bound**. In every denoising step, the entire model weights (several gigabytes) must be streamed from Unified Memory into the GPU compute cores.

Therefore, **generation speed on Apple Silicon scales directly with your chip's memory bandwidth**, rather than just GPU core count:

| Apple Silicon Tier | Memory Bandwidth | Unified Memory (VRAM) | Generation Speed (4 Steps, 1024×1024) | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **M1 / M2 / M3 / M4 (Base)** | 100 – 150 GB/s | 16 GB – 24 GB | ~45 – 65 seconds | Entry-level studio. **8-bit quantization** required. |
| **M1 / M2 / M3 / M4 Pro** | 150 – 200 GB/s | 18 GB – 36 GB | ~25 – 35 seconds | **Creative sweet spot** for designers and illustrators. |
| **M1 / M2 / M3 / M4 Max** | 300 – 400+ GB/s | 36 GB – 128 GB | **~12 – 18 seconds** | **Professional high-speed workhorse.** Ideal for heavy batch editing. |
| **M1 / M2 Ultra** | **800+ GB/s** | 64 GB – 192 GB | **~6 – 9 seconds** | **Ultimate workstation speed.** Outperforms many dedicated desktop GPUs. |

> [!IMPORTANT]
> **Memory Sizing Notice:**
> * **8 GB Macs:** Not supported. FIBO 2 weights and context tensors cannot fit into 8 GB Unified Memory without extreme system swapping.
> * **16 GB Macs:** Fully supported using **8-bit quantization** (`q8`). Leave browser tabs modest during generation.
> * **32 GB+ Macs:** Can run 8-bit or unquantized weights effortlessly with zero swap pressure.

---

## 💻 OS-Specific Installation Guides

### Universal 1-Click Setup (Recommended)

FIBO Studio includes an intelligent installer that detects your operating system and hardware accelerator, installing only the required packages:

```bash
git clone https://github.com/medienbueroleipzig/fibo-studio.git
cd fibo-studio

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

python installer/setup.py
```

---

### Manual Setup: macOS (Apple Silicon M1/M2/M3/M4)

```bash
# 1. Clone & enter repository
git clone https://github.com/medienbueroleipzig/fibo-studio.git
cd fibo-studio

# 2. Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install native Apple MLX & MPS stack
pip install -r installer/requirements-mlx.txt

# 4. Launch
./run.sh
# or
./scripts/start_mac.sh
```

---

### Manual Setup: Linux (Ubuntu/Debian + NVIDIA CUDA)

```bash
# 1. Clone & enter repository
git clone https://github.com/medienbueroleipzig/fibo-studio.git
cd fibo-studio

# 2. Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch with CUDA 12.8 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install CUDA requirements
pip install -r installer/requirements-cuda.txt

# 5. (Optional, Recommended for RTX 40/50) Install SageAttention 2:
pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/sageattention-2.1.1-cp310-cp310-linux_x86_64.whl

# 6. Launch
./scripts/start_linux.sh
```

---

### Manual Setup: Windows 11 (NVIDIA CUDA)

In PowerShell:
```powershell
# 1. Clone & enter repository
git clone https://github.com/medienbueroleipzig/fibo-studio.git
cd fibo-studio

# 2. Virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install CUDA requirements
pip install -r installer/requirements-cuda.txt

# 5. Launch
python app.py
```

Open your browser at: **`http://127.0.0.1:7865`**

---

## 💾 External SSD HuggingFace Cache (Optional, Recommended)

BRIA FIBO 2.0 models require approximately 20–40 GB of storage. To save space on your internal boot drive, store all Hugging Face snapshots on an external SSD:

```bash
# Point HF cache to external drive (e.g. in run.sh or ~/.zshrc)
export HF_HOME="/Volumes/YourSSD/ModelBackUp/Hugginface"
export HUGGINGFACE_HUB_CACHE="/Volumes/YourSSD/ModelBackUp/Hugginface/hub"
```
FIBO Studio automatically scans mounted external SSDs on startup.

---

## 🔑 Hugging Face Gated Model Access

BRIA FIBO 2.0 is a gated model released by BRIA AI.
1. Accept the model terms on Hugging Face: [briaai/Fibo-2.0-edit-turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo).
2. Enter your Hugging Face User Access Token directly in the **"Hugging Face Authentifizierung"** dropdown in the Web UI.

---

## 📂 Project Structure

```text
fibo-studio/
├── app.py                     # Main Gradio application with Dark Compact Theme
├── run.sh                     # Launch script with Apple Silicon environment checks
├── requirements.txt           # Standard requirements
├── user_modifiers.json        # Persistent user style presets (Comic2Photo, etc.)
├── installer/
│   ├── setup.py               # Cross-platform hardware auto-detection installer
│   ├── requirements-mlx.txt   # Native Apple Silicon MLX & MPS dependencies
│   ├── requirements-cuda.txt  # NVIDIA CUDA PyTorch & Diffusers dependencies
│   └── requirements-common.txt# Shared UI & utility dependencies
├── core/
│   ├── engine.py              # Universal execution engine (MLX & CUDA dispatch)
│   ├── backends/
│   │   ├── mlx_backend.py     # Native Apple MLX / mflux implementation
│   │   └── cuda_backend.py    # Native NVIDIA CUDA / SageAttention implementation
│   ├── postprocessing.py      # Lanczos, SeedVR2 (MLX), Real-ESRGAN, Face Refiner
│   ├── vision.py              # Qwen2-VL (MLX) captioning & WD-Tagger (MPS)
│   ├── metadata.py            # Universal metadata reader/writer & History-hash recovery
│   ├── resolution.py          # Aspect ratio presets, EXIF auto-transposition, 16px raster
│   └── utils.py               # Memory cleanup and Mac ._* file sanitization
├── scripts/
│   ├── start_mac.sh           # macOS launch script
│   ├── start_linux.sh         # Linux CUDA launch script
│   └── tunnel_mac_to_linux.sh # Remote SSH port forwarding helper
└── outputs/
    ├── edit/                  # Auto-saved multi-reference edit images & sidecars
    ├── txt2img/               # Auto-saved text-to-image outputs & sidecars
    └── upscaled/              # Auto-saved postprocessed and super-resolution images
```

---

## 🤝 Acknowledgments

* **BRIA AI** for releasing the revolutionary [FIBO 2.0](https://bria.ai) model series.
* **Apple Machine Learning Research** for [MLX](https://github.com/ml-explore/mlx).
* **Filip Strand** for [MFlux](https://github.com/filipstrand/mflux).
* **MonsterMMORPG / SageAttention** for next-generation attention kernels.
* **SmilingWolf** for the WD-Tagger anime and illustration vision models.

---

## 📄 License

This software is released under the **Apache 2.0 License**.  
*BRIA FIBO model weights are governed by their respective Hugging Face community licenses.*
