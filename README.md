# 🎨 FIBO Studio

> **High-Performance Native Apple Silicon MLX Interface for BRIA FIBO 2.0**  
> *Zero CUDA/PyTorch overhead • Native Metal Acceleration • Multi-Reference Image Editing • Diffusion Upscaling • Vision AI*

[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20%7C%20M2%20%7C%20M3%20%7C%20M4-black?style=for-the-badge&logo=apple)](https://www.apple.com)
[![MLX](https://img.shields.io/badge/Engine-Apple%20MLX-orange?style=for-the-badge)](https://github.com/ml-explore/mlx)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![Gradio](https://img.shields.io/badge/UI-Gradio%20Blocks-critical?style=for-the-badge&logo=gradio)](https://gradio.app)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)](LICENSE)

---

## 🌟 Overview

**FIBO Studio** is a dedicated, production-ready GUI application tailored specifically for **Apple Silicon Macs** (M1/M2/M3/M4). It provides a full-featured creative suite for **BRIA FIBO 2.0** (both 4-step Turbo Edit and 30-step Base/Dev models), running directly on Apple's **Metal Performance Shaders (MPS)** and the **Apple MLX** framework.

Unlike generic Stable Diffusion / FLUX web UIs designed around NVIDIA CUDA, FIBO Studio takes full advantage of Apple's **Unified Memory Architecture**, eliminating memory duplication and PCIe bandwidth bottlenecks.

---

## 🚀 Key Features

### 1. ✏️ Multi-Reference Image Editing (FIBO 2 Turbo)
* **4-Step Turbo Diffusion:** Generates high-fidelity modifications in just 4 inference steps.
* **Up to 5 Reference Slots:** Combine subjects, styles, backgrounds, and objects with prompt syntax (`<image_1>`, `<image_2>`, etc.).
* **Inpainting & Masking:** Draw directly or upload binary masks for surgical localized edits.
* **Iterative Loop:** 1-click **"Ergebnis als Referenz 1 übernehmen"** to chain edits into complex transformations.

### 2. 🎨 Text-to-Image Generation
* **Turbo (4 Steps) & Base/Dev (30 Steps):** Choose between lightning-fast drafts and studio-grade precision.
* **VLM Prompt Expansion:** Automatically expands short prompts with descriptive prompt engineering via Apple MLX.
* **Smart Aspect Ratios:** 1:1 Square, 16:9 Landscape, 9:16 Portrait, 21:9 Ultrawide, and custom resolutions rounded to 16px.

### 3. ✨ Multi-Tier Post-Processing & Upscaling
Integrated directly beneath generation outputs:
* **Lanczos Instant Resizer (0.05s):** Instant 1.5x / 2.0x / 4.0x scaling on CPU with adaptive **Unsharp Mask** detail enhancement. Consumes 0 MB VRAM.
* **SeedVR2 (Native Apple MLX Diffusion):** 1-step diffusion super-resolution (up to 4K) specifically compiled for Apple Silicon.
* **Real-ESRGAN:** Neural texture restoration for fine lines, fabrics, and photographic detail.
* **👤 Face Refiner:** Restores crisp eyes, lips, facial hair, and skin micro-contrast.

### 4. 👁️ Vision AI & WD-Tagger Studio
* **MLX Vision Language Model (Qwen2-VL 4-Bit):** Ultra-fast scene analysis and captioning running at **~110 tokens/second** on Apple MLX.
* **Deep Danbooru / WD-Tagger (SwinV2):** GPU-accelerated tag classification in **<0.4 seconds** on Apple MPS with confidence threshold filtering.
* **Direct Prompt Injection:** Transfer generated captions or detected tags directly into Edit or Text-to-Image prompts with a single click.

### 5. 🔍 Multi-Standard PNG Metadata & Smart History Recovery
* **Universal Embedding:** Automatically writes metadata into:
  * `parameters` chunk (Automatic1111 / WebUI / Civitai compatible)
  * `prompt` chunk (ComfyUI compatible)
  * `fibo_meta` chunk (JSON)
  * EXIF `UserComment` & Adobe XMP / IPTC
  * Companion `.metadata.json` sidecar
* **History Hash Recovery:** Even if you paste an image via macOS clipboard (`Cmd+V`) where browsers strip EXIF headers, FIBO Studio matches the structural thumbnail hash against local outputs to recover 100% of the original prompt and generation parameters.

### 6. 🎨 High-Tech Compact Dark Theme
* Built with the **Compact Theme** palette (`#0b0f19` dark canvas, glowing `#fe7c00` / `#d0670c` orange accents).
* **0px Strict Radius:** Crisp, sharp, professional studio edges across all buttons, dropdowns, cards, and image frames.
* **Typography:** Modern styling using Google Fonts *Gabarito* and *Roboto Mono*.

---

## 🛠️ System Requirements

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **Mac Model** | Apple Silicon M1 (16 GB) | Apple Silicon M2/M3/M4 Pro/Max (32 GB - 64 GB+) |
| **macOS** | macOS 14 Sonoma | macOS 15 Sequoia |
| **Python** | Python 3.11 or 3.12 | Python 3.12 |
| **Storage** | ~20 GB free space | External SSD (Thunderbolt / USB 3.2 Gen 2) for model cache |

---

## 📦 Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/medienbueroleipzig/fibo-studio.git
cd fibo-studio
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch FIBO Studio
```bash
./run.sh
```
Or directly:
```bash
python app.py
```
Open your browser at: **`http://127.0.0.1:7865`**

---

## 💾 External SSD HuggingFace Cache (Optional, Recommended)

BRIA FIBO 2.0 models require approximately 20–40 GB of storage. To save space on your internal Macintosh HD, you can store all Hugging Face snapshots on an external SSD:

```bash
# Example: Point HF cache to external drive
export HF_HOME="/Volumes/YourSSD/ModelBackUp/Hugginface"
export HUGGINGFACE_HUB_CACHE="/Volumes/YourSSD/ModelBackUp/Hugginface/hub"
```
FIBO Studio automatically detects external drives and symlinks `~/.cache/huggingface/hub` to keep your internal drive clean!

---

## 🔑 Hugging Face Gated Model Access

BRIA FIBO 2.0 is a gated model. To use it:
1. Accept the model terms on Hugging Face: [briaai/Fibo-2.0-edit-turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo).
2. Enter your Hugging Face User Access Token directly in the **"Hugging Face Authentifizierung"** dropdown in the Web UI.

---

## 📂 Project Architecture

```text
fibo_studio/
├── app.py                     # Main Gradio application with Dark Compact Theme
├── run.sh                     # Launch script with Apple Silicon environment checks
├── requirements.txt           # Python dependencies
├── core/
│   ├── engine.py              # MLX execution engine for FIBO 2 (Edit & Text-to-Image)
│   ├── postprocessing.py      # Lanczos, SeedVR2 (MLX), Real-ESRGAN, Face Refiner
│   ├── vision.py              # Qwen2-VL (MLX) captioning & WD-Tagger (MPS)
│   ├── metadata.py            # Universal metadata reader/writer & History-hash recovery
│   ├── resolution.py          # Aspect ratio presets and coordinate math
│   └── utils.py               # Memory cleanup and Mac ._* file sanitization
└── outputs/
    ├── edit/                  # Auto-saved multi-reference edit images & sidecars
    ├── txt2img/               # Auto-saved text-to-image outputs & sidecars
    └── upscaled/              # Auto-saved postprocessed and super-resolution images
```

---

## 🤝 Acknowledgments

* **BRIA AI** for releasing [FIBO 2.0](https://bria.ai).
* **Apple Machine Learning Research** for [MLX](https://github.com/ml-explore/mlx).
* **Filip Strand** for [MFlux](https://github.com/filipstrand/mflux).
* **SmilingWolf** for the WD-Tagger models.
* **DeepKyu** for the compact theme design inspiration.

---

## 📄 License

This software is released under the **Apache 2.0 License**.  
*BRIA FIBO weights are governed by their respective Hugging Face community licenses.*
