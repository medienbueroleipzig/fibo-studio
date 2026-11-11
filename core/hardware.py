"""Hardware auto-detection and capabilities provider for FIBO Studio.
Dynamically detects NVIDIA CUDA (SageAttention 2 / FlashAttention), Apple Silicon MLX, or CPU fallback.
"""

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class HardwareInfo:
    platform_name: str          # "darwin", "linux", "windows"
    backend: str                # "cuda", "mlx", "cpu"
    device_name: str            # e.g. "NVIDIA GeForce RTX 5070 Ti"
    vram_gb: float              # Total VRAM or Unified Memory in GB
    has_sageattention: bool     # True if SageAttention 2/1 is installed
    has_flashattention: bool    # True if FlashAttention is installed
    compute_capability: Tuple[int, int]  # e.g. (12, 0) for Blackwell
    cuda_version: Optional[str] = None
    torch_version: Optional[str] = None

    @property
    def is_cuda(self) -> bool:
        return self.backend == "cuda"

    @property
    def is_mlx(self) -> bool:
        return self.backend == "mlx"

    @property
    def is_blackwell(self) -> bool:
        return self.compute_capability[0] >= 12

    def get_recommended_precision(self) -> str:
        if self.is_cuda:
            return "bf16"
        elif self.is_mlx:
            return "8-Bit"
        return "float32"

    def get_status_badge(self) -> str:
        """Returns a formatted Markdown/HTML status badge for the UI."""
        if self.is_cuda:
            sage_text = " • ⚡ SageAttention 2" if self.has_sageattention else (" • ⚡ FlashAttn" if self.has_flashattention else "")
            return (
                f"<div style='display:inline-flex; align-items:center; gap:8px; padding:4px 10px; background:#111827; "
                f"border:1px solid #fe7c00; border-radius:0px; font-family:monospace; font-size:0.8rem; color:#f8fafc;'>"
                f"<span style='color:#fe7c00; font-weight:bold;'>🚀 NVIDIA CUDA</span> "
                f"<span>{self.device_name} ({self.vram_gb:.1f} GB VRAM{sage_text})</span>"
                f"</div>"
            )
        elif self.is_mlx:
            return (
                f"<div style='display:inline-flex; align-items:center; gap:8px; padding:4px 10px; background:#111827; "
                f"border:1px solid #10b981; border-radius:0px; font-family:monospace; font-size:0.8rem; color:#f8fafc;'>"
                f"<span style='color:#10b981; font-weight:bold;'>🍏 Apple Silicon</span> "
                f"<span>{self.device_name} ({self.vram_gb:.1f} GB Unified Memory • MLX Metal)</span>"
                f"</div>"
            )
        else:
            return (
                f"<div style='display:inline-flex; align-items:center; gap:8px; padding:4px 10px; background:#111827; "
                f"border:1px solid #64748b; border-radius:0px; font-family:monospace; font-size:0.8rem; color:#94a3b8;'>"
                f"<span>⚙️ CPU Mode ({self.device_name})</span>"
                f"</div>"
            )


_CACHED_HW_INFO: Optional[HardwareInfo] = None


def detect_hardware() -> HardwareInfo:
    """Detects available hardware accelerators and returns a cached HardwareInfo object."""
    global _CACHED_HW_INFO
    if _CACHED_HW_INFO is not None:
        return _CACHED_HW_INFO

    plat = platform.system().lower()
    
    # 1. Check NVIDIA CUDA via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024**3)
            cap = torch.cuda.get_device_capability(0)
            cuda_ver = torch.version.cuda
            torch_ver = torch.__version__
            
            has_sage = False
            try:
                import sageattention
                has_sage = True
            except Exception:
                pass
                
            has_flash = False
            try:
                import flash_attn
                has_flash = True
            except Exception:
                pass

            _CACHED_HW_INFO = HardwareInfo(
                platform_name=plat,
                backend="cuda",
                device_name=dev_name,
                vram_gb=vram_gb,
                has_sageattention=has_sage,
                has_flashattention=has_flash,
                compute_capability=cap,
                cuda_version=cuda_ver,
                torch_version=torch_ver,
            )
            return _CACHED_HW_INFO
    except Exception:
        pass

    # 2. Check Apple Silicon MLX
    if plat == "darwin" and platform.machine() == "arm64":
        try:
            import mlx.core as mx
            
            # Approximate unified memory
            total_ram_gb = 16.0
            try:
                import psutil
                total_ram_gb = psutil.virtual_memory().total / (1024**3)
            except Exception:
                pass

            _CACHED_HW_INFO = HardwareInfo(
                platform_name="darwin",
                backend="mlx",
                device_name="Apple Silicon Metal",
                vram_gb=total_ram_gb,
                has_sageattention=False,
                has_flashattention=False,
                compute_capability=(0, 0),
            )
            return _CACHED_HW_INFO
        except Exception:
            pass

    # 3. CPU Fallback
    _CACHED_HW_INFO = HardwareInfo(
        platform_name=plat,
        backend="cpu",
        device_name=platform.processor() or "Generic CPU",
        vram_gb=0.0,
        has_sageattention=False,
        has_flashattention=False,
        compute_capability=(0, 0),
    )
    return _CACHED_HW_INFO


def locate_checkpoints() -> Dict[str, Optional[Path]]:
    """Scans local locations for pre-downloaded FIBO checkpoints (fibo_studio/ckpts or symlinked caches)."""
    candidates = [
        Path(__file__).resolve().parent.parent / "ckpts",
        Path.home() / ".cache" / "fibo_ckpts",
        Path("/home/linx/fibo_studio/ckpts"),
        Path("/home/linx/pinokio/api/wan.git/app/ckpts"),
        Path("/Volumes/Cachy-Files/fibo_studio/ckpts"),
        Path("/Volumes/Cachy-Files/pinokio/api/wan.git/app/ckpts"),
    ]
    
    found: Dict[str, Optional[Path]] = {
        "transformer_turbo": None,
        "text_encoder": None,
        "vae": None,
    }
    
    for base in candidates:
        if not base.exists():
            continue
        
        turbo = base / "Fibo_2.0_Edit_Turbo_bf16.safetensors"
        if turbo.exists() and found["transformer_turbo"] is None:
            found["transformer_turbo"] = turbo
            
        te = base / "Qwen3-VL-4B-Instruct"
        if te.exists() and found["text_encoder"] is None:
            found["text_encoder"] = te
            
        vae = base / "fibo2" / "flux2_vae_fp16.safetensors"
        if vae.exists() and found["vae"] is None:
            found["vae"] = vae

    return found


locate_wan_checkpoints = locate_checkpoints

