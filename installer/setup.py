#!/usr/bin/env python3
"""Cross-platform installer for FIBO Studio.
Detects OS and hardware accelerators and installs the exact right requirements.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd_list):
    print(f"\n[RUN] {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, check=False)
    if res.returncode != 0:
        print(f"[WARNING] Command exited with code {res.returncode}")


def main():
    print("=" * 60)
    print("🚀 FIBO Studio - Cross-Platform Installer")
    print("=" * 60)
    
    plat = platform.system().lower()
    is_arm_mac = plat == "darwin" and platform.machine() == "arm64"
    
    req_dir = Path(__file__).parent
    
    if is_arm_mac:
        print("🍏 Detected: Apple Silicon macOS (arm64)")
        print("Installing MLX & Metal acceleration stack...")
        run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req_dir / "requirements-mlx.txt")])
    elif plat in ("linux", "windows"):
        print(f"🐧/🪟 Detected: {platform.system()}")
        print("Installing CUDA AI stack...")
        run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req_dir / "requirements-cuda.txt")])
        print("\nTip: To install PyTorch 2.7 with CUDA 12.8 support, run:")
        print(f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
        print("\nTo install SageAttention 2 on Linux:")
        print(f"{sys.executable} -m pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/sageattention-2.1.1-cp310-cp310-linux_x86_64.whl")
    else:
        print(f"Detected generic platform: {plat}")
        run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req_dir / "requirements-common.txt")])

    print("\n✅ Setup complete! You can now start FIBO Studio:")
    if is_arm_mac:
        print("   ./scripts/start_mac.sh")
    else:
        print("   ./scripts/start_linux.sh")


if __name__ == "__main__":
    main()

