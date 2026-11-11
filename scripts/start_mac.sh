#!/bin/bash
# ==============================================================================
# FIBO Studio - macOS Apple Silicon Runner
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

echo "========================================================"
echo "🍏 Starting FIBO Studio (Apple Silicon MLX Metal)..."
echo "🌐 URL: http://127.0.0.1:7865"
echo "========================================================"

if [ -f "$DIR/.venv/bin/activate" ]; then
    source "$DIR/.venv/bin/activate"
fi

# Clean Mac metadata
find "$DIR" -type f -name "._*" -delete 2>/dev/null || true

python app.py

