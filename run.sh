#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Clean up macOS metadata files
find . -type f -name "._*" -delete 2>/dev/null || true

# Configure Hugging Face Cache directory to external SSD if available
EXTERNAL_HF="/Volumes/Extreme SSD/ModelBackUp/Hugginface"
if [ -d "$EXTERNAL_HF" ]; then
    export HF_HOME="$EXTERNAL_HF"
    export HUGGINGFACE_HUB_CACHE="$EXTERNAL_HF/hub"
    echo "💾 Verwende externe SSD als HuggingFace-Cache: $EXTERNAL_HF"
fi

PORT=7865
EXISTING_PIDS=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -n "$EXISTING_PIDS" ]; then
    echo "⚠️  Port $PORT ist bereits belegt (PID: $EXISTING_PIDS). Beende alten Prozess..."
    kill -9 $EXISTING_PIDS 2>/dev/null || true
    sleep 1
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
else
    echo "Virtual environment .venv not found. Please create it first."
    exit 1
fi

echo "=============================================="
echo "🚀 Starting FIBO Studio (Apple Silicon MLX)..."
echo "🌐 URL: http://127.0.0.1:7865"
echo "=============================================="

exec python app.py
