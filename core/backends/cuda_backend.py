"""NVIDIA CUDA Backend for FIBO Studio with SageAttention 2 & PyTorch acceleration."""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from .base import BaseFiboEngine
from ..metadata import embed_full_metadata
from ..hardware import locate_checkpoints

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


def get_output_dir(subfolder: str = "edit") -> Path:
    base_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / subfolder
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


class CudaFiboEngine(BaseFiboEngine):
    """High-performance PyTorch CUDA engine with SageAttention 2 and MMGP acceleration."""

    def __init__(self):
        self._pipeline = None
        self._cached_pipeline_key: Optional[str] = None
        self._setup_wan_path()

    def _setup_wan_path(self):
        candidates = [
            Path("/home/linx/pinokio/api/wan.git/app"),
            Path("/Volumes/Cachy-Files/pinokio/api/wan.git/app"),
            Path(__file__).resolve().parent.parent.parent / "wan_app",
        ]
        for p in candidates:
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
                break

    def get_supported_models(self) -> List[str]:
        return ["fibo2_edit_turbo", "fibo2_dev"]

    def get_supported_precisions(self) -> List[Tuple[str, str]]:
        return [
            ("bfloat16 (Fastest / Recommended for RTX 5070 Ti)", "bf16"),
            ("float16 (Standard)", "fp16"),
        ]

    def _load_pipeline(self, model_type: str = "fibo2_edit_turbo", precision: str = "bf16"):
        cache_key = f"{model_type}_{precision}"
        if self._pipeline is not None and self._cached_pipeline_key == cache_key:
            return self._pipeline

        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._setup_wan_path()
        from mmgp import offload
        offload.shared_state["_attention"] = "sage2"

        from shared.utils import files_locator as fl
        from models.fibo2.fibo2_handler import family_handler

        # Configure checkpoint search paths (local ckpts symlink first, then system caches)
        local_ckpts = Path(__file__).resolve().parent.parent.parent / "ckpts"
        search_roots = [
            str(local_ckpts),
            "/home/linx/fibo_studio/ckpts",
            "/home/linx/pinokio/api/wan.git/app/ckpts",
            "/Volumes/Cachy-Files/fibo_studio/ckpts",
            "/Volumes/Cachy-Files/pinokio/api/wan.git/app/ckpts",
        ]
        fl.set_checkpoints_paths([p for p in search_roots if Path(p).exists()])

        model_def = family_handler.query_model_def(model_type, {})
        model_def["architecture"] = model_type

        # Locate model weights
        if model_type == "fibo2_edit_turbo":
            model_file = str(fl.locate_file("Fibo_2.0_Edit_Turbo_bf16.safetensors"))
        else:
            model_file = str(fl.locate_file("Fibo_2.0_Dev_int8_convrot.safetensors"))

        # Locate Qwen3-VL-4B weights
        te_files = [
            str(fl.locate_file("Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors")),
            str(fl.locate_file("Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_vision_bf16.safetensors")),
        ]

        print(f"[CUDA Backend] Loading {model_type} from {model_file}...")
        pipeline, _ = family_handler.load_model(
            model_filename=model_file,
            model_type=model_type,
            base_model_type=model_type,
            model_def=model_def,
            text_encoder_filename=te_files,
        )

        self._pipeline = pipeline
        self._cached_pipeline_key = cache_key
        print(f"[CUDA Backend] Model {model_type} successfully loaded on NVIDIA GeForce RTX 5070 Ti!")
        return self._pipeline

    def generate_edit(
        self,
        ref_images: List[Image.Image],
        prompt: str,
        steps: int = 4,
        guidance: float = 1.0,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        precision: str = "bf16",
        mask_image: Optional[Image.Image] = None,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        progress_callback: Optional[Any] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        if not ref_images:
            raise ValueError("At least one reference image is required for editing.")

        actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
        start_time = time.time()

        pipe = self._load_pipeline(model_type="fibo2_edit_turbo", precision=precision)

        from shared.utils.utils import convert_tensor_to_image

        masks_arg = [mask_image] if mask_image is not None else None

        with torch.inference_mode():
            raw_output = pipe.generate(
                input_prompt=prompt,
                input_ref_images=ref_images[:4],
                sampling_steps=steps,
                guide_scale=guidance,
                seed=actual_seed,
                width=width,
                height=height,
                video_prompt_type="KI",
                input_masks=masks_arg,
            )

        if isinstance(raw_output, torch.Tensor):
            result_pil = convert_tensor_to_image(raw_output)
        elif isinstance(raw_output, list) and len(raw_output) > 0 and isinstance(raw_output[0], Image.Image):
            result_pil = raw_output[0]
        elif isinstance(raw_output, Image.Image):
            result_pil = raw_output
        else:
            result_pil = convert_tensor_to_image(raw_output)

        elapsed = time.time() - start_time

        # Save output
        out_dir = get_output_dir("edit")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved_filename = f"edit_cuda_{timestamp}_s{actual_seed}.png"
        saved_path = out_dir / saved_filename
        try:
            result_pil.save(saved_path)
            embed_full_metadata(
                saved_path,
                prompt=prompt,
                model_name="fibo2-edit-turbo-cuda",
                steps=steps,
                guidance=guidance,
                seed=actual_seed,
                width=result_pil.width,
                height=result_pil.height,
                num_references=len(ref_images),
            )
        except Exception:
            pass

        meta = {
            "elapsed_seconds": elapsed,
            "seed": actual_seed,
            "width": result_pil.width,
            "height": result_pil.height,
            "steps": steps,
            "saved_path": str(saved_path),
            "backend": "cuda",
        }
        return result_pil, meta

    def generate_t2i(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 30,
        guidance: float = 5.0,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        precision: str = "bf16",
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        progress_callback: Optional[Any] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
        start_time = time.time()

        pipe = self._load_pipeline(model_type="fibo2_dev", precision=precision)
        from shared.utils.utils import convert_tensor_to_image

        with torch.inference_mode():
            raw_output = pipe.generate(
                input_prompt=prompt,
                n_prompt=negative_prompt,
                sampling_steps=steps,
                guide_scale=guidance,
                seed=actual_seed,
                width=width,
                height=height,
            )

        if isinstance(raw_output, torch.Tensor):
            result_pil = convert_tensor_to_image(raw_output)
        elif isinstance(raw_output, list) and len(raw_output) > 0 and isinstance(raw_output[0], Image.Image):
            result_pil = raw_output[0]
        else:
            result_pil = raw_output

        elapsed = time.time() - start_time
        out_dir = get_output_dir("t2i")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved_filename = f"t2i_cuda_{timestamp}_s{actual_seed}.png"
        saved_path = out_dir / saved_filename
        try:
            result_pil.save(saved_path)
        except Exception:
            pass

        meta = {
            "elapsed_seconds": elapsed,
            "seed": actual_seed,
            "width": result_pil.width,
            "height": result_pil.height,
            "steps": steps,
            "saved_path": str(saved_path),
            "backend": "cuda",
        }
        return result_pil, meta

    def abort(self) -> None:
        if self._pipeline is not None and hasattr(self._pipeline, "abort"):
            self._pipeline.abort()
