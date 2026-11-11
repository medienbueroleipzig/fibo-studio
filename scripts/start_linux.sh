#!/bin/bash
# ==============================================================================
# FIBO Studio - Linux NVIDIA CUDA & SageAttention 2 Runner
# Optimized for Arch Linux / CachyOS & NVIDIA GeForce RTX 50-Series (Blackwell)
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

echo "========================================================"
echo "🚀 Starting Universal FIBO Studio (Linux CUDA Mode)..."
echo "========================================================"

# 1. Activate local standalone .venv
if [ -d "$DIR/.venv" ]; then
    echo "⚡ Activating local FIBO Studio standalone environment (.venv)..."
    source "$DIR/.venv/bin/activate"
elif [ -d "/home/linx/pinokio/api/wan.git/app/env" ]; then
    echo "⚡ Activating fallback CUDA environment..."
    source /home/linx/pinokio/api/wan.git/app/env/bin/activate
fi

# 2. Check local checkpoints
if [ -d "$DIR/ckpts" ]; then
    echo "📦 Detected local checkpoints at $DIR/ckpts"
    export FIBO_CKPTS_DIR="$DIR/ckpts"
elif [ -d "/home/linx/pinokio/api/wan.git/app/ckpts" ]; then
    echo "📦 Detected local checkpoints cache at /home/linx/pinokio/api/wan.git/app/ckpts"
    export FIBO_CKPTS_DIR="/home/linx/pinokio/api/wan.git/app/ckpts"
fi

export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT="7865"

# Clean Mac metadata
find "$DIR" -type f -name "._*" -delete 2>/dev/null || true

python app.py

