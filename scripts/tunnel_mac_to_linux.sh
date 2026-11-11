#!/bin/bash
# ==============================================================================
# FIBO Studio - Mac to Linux SSH Tunnel & Auto-Launcher
# Connects to your CachyOS / Linux RTX 5070 Ti host and forwards port 7865
# ==============================================================================

LINUX_HOST="192.168.178.77"
LINUX_USER="linx"
PORT="7865"

echo "========================================================"
echo "🚀 Connecting FIBO Studio Tunnel to Linux (RTX 5070 Ti)..."
echo "🌐 Local URL: http://127.0.0.1:${PORT}"
echo "🖥️  Remote Host: ${LINUX_USER}@${LINUX_HOST}"
echo "========================================================"

# Auto-open browser after 2 seconds
(sleep 2 && open "http://127.0.0.1:${PORT}") &

# Forward local port 7865 to remote localhost 7865
ssh -N -L ${PORT}:127.0.0.1:${PORT} ${LINUX_USER}@${LINUX_HOST}

