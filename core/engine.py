"""Universal Execution Engine for FIBO 2.0 (Dispatching to CUDA or MLX)."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

from .hardware import detect_hardware, HardwareInfo
from .resolution import round_to_16
from .backends.base import BaseFiboEngine
from .backends.mlx_backend import MLXFiboEngine, get_output_dir
from .backends.cuda_backend import CudaFiboEngine


class FiboEngine:
    """Universal dispatcher that delegates to the active hardware backend (CUDA or MLX)."""

    def __init__(self):
        self.hw: HardwareInfo = detect_hardware()
        self._backend: BaseFiboEngine = self._create_backend()

    def _create_backend(self) -> BaseFiboEngine:
        if self.hw.is_cuda:
            return CudaFiboEngine()
        elif self.hw.is_mlx:
            return MLXFiboEngine()
        else:
            # Fallback to MLX or CPU
            return MLXFiboEngine()

    @property
    def backend_name(self) -> str:
        return self.hw.backend

    def run_edit(
        self,
        images: List[Image.Image],
        prompt: str,
        mask: Optional[Image.Image] = None,
        model_name: str = "fibo2-edit-turbo",
        quantize: int = 8,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: int = 4,
        guidance: float = 1.0,
        seed: int = 42,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
    ) -> Tuple[Union[Image.Image, str], float, str]:
        """Universal edit inference runner returning (result_image, elapsed_seconds, info_text)."""
        w = width if width is not None else 1024
        h = height if height is not None else 1024
        prec = str(quantize) if self.hw.is_mlx else ("bf16" if self.hw.is_cuda else "fp32")
        
        result_img, meta = self._backend.generate_edit(
            ref_images=images,
            prompt=prompt,
            steps=steps,
            guidance=guidance,
            seed=seed,
            width=w,
            height=h,
            precision=prec,
            mask_image=mask,
            lora_path=lora_path,
            lora_scale=lora_scale,
        )

        elapsed = meta.get("elapsed_seconds", 0.0)
        saved_path = meta.get("saved_path", "")
        backend_used = meta.get("backend", self.hw.backend).upper()
        
        info = (
            f"⚡ **Generation Time:** {elapsed:.2f}s &nbsp;|&nbsp; "
            f"**Backend:** {backend_used} &nbsp;|&nbsp; "
            f"**Resolution:** {result_img.width}×{result_img.height} &nbsp;|&nbsp; "
            f"**Steps:** {steps} &nbsp;|&nbsp; "
            f"**Seed:** `{seed}`<br>"
            f"📦 **Model:** {model_name} &nbsp;|&nbsp; "
            f"**Saved:** `{Path(saved_path).name}`"
        )
        output_target = str(saved_path) if saved_path and Path(saved_path).exists() else result_img
        return output_target, elapsed, info

    def run_text_to_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        model_name: str = "fibo2",
        quantize: int = 8,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        guidance: float = 5.0,
        seed: int = 42,
        expand_prompt: bool = False,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
    ) -> Tuple[Union[Image.Image, str], float, str]:
        """Universal text-to-image runner returning (result_image, elapsed_seconds, info_text)."""
        w = round_to_16(width)
        h = round_to_16(height)
        prec = str(quantize) if self.hw.is_mlx else ("bf16" if self.hw.is_cuda else "fp32")

        # If expand_prompt requested, format or enhance if prompt enhancer available
        final_prompt = prompt.strip()
        
        result_img, meta = self._backend.generate_t2i(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            model_name=model_name,
            steps=steps,
            guidance=guidance,
            seed=seed,
            width=w,
            height=h,
            precision=prec,
            lora_path=lora_path,
            lora_scale=lora_scale,
        )

        elapsed = meta.get("elapsed_seconds", 0.0)
        saved_path = meta.get("saved_path", "")
        backend_used = meta.get("backend", self.hw.backend).upper()
        
        info = (
            f"⚡ **Generation Time:** {elapsed:.2f}s &nbsp;|&nbsp; "
            f"**Backend:** {backend_used} &nbsp;|&nbsp; "
            f"**Resolution:** {result_img.width}×{result_img.height} &nbsp;|&nbsp; "
            f"**Steps:** {steps} &nbsp;|&nbsp; "
            f"**Seed:** `{seed}`<br>"
            f"📦 **Model:** {model_name} &nbsp;|&nbsp; "
            f"**Saved:** `{Path(saved_path).name}`"
        )
        output_target = str(saved_path) if saved_path and Path(saved_path).exists() else result_img
        return output_target, elapsed, info

    run_t2i = run_text_to_image

    def abort(self) -> None:
        """Immediately halts any running image generation across active backend."""
        print("[Engine] Abort requested by user.")
        if hasattr(self._backend, "abort"):
            self._backend.abort()


GLOBAL_ENGINE = FiboEngine()
