"""Apple Silicon MLX Backend for FIBO Studio using mflux and native Metal acceleration."""

import gc
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from .base import BaseFiboEngine
from ..metadata import embed_full_metadata
from ..utils import cleanup_mac_files

try:
    import mlx.core as mx
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.common.resolution.config_resolution import ConfigResolution
    from mflux.models.fibo2.variants import Fibo2, Fibo2Edit
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def get_output_dir(subfolder: str = "edit") -> Path:
    base_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / subfolder
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


class AbortableCallback:
    """In-loop callback that raises KeyboardInterrupt when abort is requested."""
    def __init__(self):
        self.should_abort = False

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        if self.should_abort:
            print(f"[MLX Backend] Generation cancelled by user at step {t}!")
            raise KeyboardInterrupt("Generation stopped by user.")


class MLXFiboEngine(BaseFiboEngine):
    """Engine backend executing on Apple Silicon via MLX / mflux."""

    def __init__(self):
        self._cached_model = None
        self._cached_model_key: Optional[str] = None
        self._abort_cb = AbortableCallback()

    def abort(self) -> None:
        """Signals active generation loop to terminate immediately."""
        print("[MLX Backend] Abort triggered.")
        self._abort_cb.should_abort = True

    def get_supported_models(self) -> List[str]:
        return ["fibo2-edit-turbo", "fibo2", "fibo-lite"]

    def get_supported_precisions(self) -> List[Tuple[str, str]]:
        return [
            ("8-Bit (Optimal Quality & Speed)", "8"),
            ("4-Bit (Ultra-Low Memory)", "4"),
        ]

    def _get_model(
        self,
        model_name: str,
        quantize: int,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        is_edit: bool = True,
    ):
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX / mflux is not installed or available on this system.")

        cache_key = f"{model_name}_edit{is_edit}_q{quantize}_lora_{lora_path}_{lora_scale}"
        if self._cached_model is not None and self._cached_model_key == cache_key:
            return self._cached_model

        if self._cached_model is not None:
            del self._cached_model
            self._cached_model = None
            gc.collect()
            mx.clear_cache()

        model_config = ConfigResolution.resolve_restricted(
            model_name,
            "fibo2" if not is_edit else "fibo2-edit-turbo",
            extra_keys=("fibo2-edit-turbo", "fibo2", "fibo-lite"),
        )

        lora_kwargs = {}
        if lora_path and os.path.exists(lora_path.strip()):
            clean_path = lora_path.strip()
            lora_kwargs = {
                "lora_paths": [Path(clean_path)],
                "lora_scales": [float(lora_scale)],
            }

        if is_edit:
            model = Fibo2Edit(
                model_config=model_config,
                quantize=quantize,
                **lora_kwargs,
            )
        else:
            model = Fibo2(
                model_config=model_config,
                quantize=quantize,
                **lora_kwargs,
            )

        try:
            model.callbacks.register(self._abort_cb)
        except Exception:
            pass

        self._cached_model = model
        self._cached_model_key = cache_key
        return model

    def generate_edit(
        self,
        ref_images: List[Image.Image],
        prompt: str,
        steps: int = 4,
        guidance: float = 1.0,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        precision: str = "8",
        mask_image: Optional[Image.Image] = None,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        progress_callback: Optional[Any] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        if not ref_images:
            raise ValueError("At least one reference image is required for editing.")

        self._abort_cb.should_abort = False
        from ..resolution import sanitize_reference_image, sanitize_mask_image, round_to_16

        clean_refs = [sanitize_reference_image(img) for img in ref_images[:5]]
        clean_mask = None
        if mask_image is not None and clean_refs:
            clean_mask = sanitize_mask_image(mask_image, clean_refs[0].size)

        gen_width = round_to_16(width)
        gen_height = round_to_16(height)

        actual_seed = seed if seed is not None else 42
        quant_int = int(precision) if precision.isdigit() else 8
        model = self._get_model(
            model_name="fibo2-edit-turbo",
            quantize=quant_int,
            lora_path=lora_path,
            lora_scale=lora_scale,
            is_edit=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = []
            for idx, img in enumerate(clean_refs):
                path = Path(tmpdir) / f"ref_{idx}.png"
                img.convert("RGB").save(path)
                image_paths.append(path)

            mask_path = None
            if clean_mask is not None and len(image_paths) == 1:
                mask_file = Path(tmpdir) / "mask.png"
                clean_mask.convert("L").save(mask_file)
                mask_path = mask_file

            start_time = time.time()
            try:
                generated = model.generate_image(
                    seed=actual_seed,
                    prompt=prompt,
                    image_paths=image_paths,
                    mask_path=mask_path,
                    width=gen_width,
                    height=gen_height,
                    guidance=guidance,
                    num_inference_steps=steps,
                )
            except Exception as e:
                if self._abort_cb.should_abort or "stop" in str(e).lower() or isinstance(e, KeyboardInterrupt):
                    raise RuntimeError("Generierung vom Benutzer abgebrochen (Stop).") from None
                raise e
            elapsed = time.time() - start_time
            cleanup_mac_files(tmpdir)

            # Extract actual PIL Image
            if hasattr(generated, "image") and isinstance(generated.image, Image.Image):
                result_pil = generated.image
            elif hasattr(generated, "to_pil"):
                result_pil = generated.to_pil()
            else:
                result_pil = generated

            # Save persistently
            out_dir = get_output_dir("edit")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            saved_filename = f"edit_{timestamp}_s{actual_seed}.png"
            saved_path = out_dir / saved_filename
            try:
                if hasattr(generated, "save"):
                    generated.save(path=saved_path)
                else:
                    result_pil.save(saved_path)

                embed_full_metadata(
                    saved_path,
                    prompt=prompt,
                    model_name="fibo2-edit-turbo",
                    steps=steps,
                    guidance=guidance,
                    seed=actual_seed,
                    quantize=quant_int,
                    width=result_pil.width,
                    height=result_pil.height,
                    num_references=len(image_paths),
                )
            except Exception as e:
                print(f"[MLX Backend] Save metadata error: {e}")

        meta = {
            "elapsed_seconds": elapsed,
            "seed": actual_seed,
            "width": result_pil.width,
            "height": result_pil.height,
            "steps": steps,
            "saved_path": str(saved_path),
            "backend": "mlx",
        }
        return result_pil, meta

    def generate_t2i(
        self,
        prompt: str,
        negative_prompt: str = "",
        model_name: str = "fibo2",
        steps: int = 30,
        guidance: float = 5.0,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        precision: str = "8",
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        progress_callback: Optional[Any] = None,
        **kwargs,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        self._abort_cb.should_abort = False
        from ..resolution import round_to_16

        gen_width = round_to_16(width)
        gen_height = round_to_16(height)

        actual_seed = seed if seed is not None else 42
        quant_int = int(precision) if precision.isdigit() else 8
        model = self._get_model(
            model_name=model_name or "fibo2",
            quantize=quant_int,
            lora_path=lora_path,
            lora_scale=lora_scale,
            is_edit=False,
        )

        start_time = time.time()
        try:
            generated = model.generate_image(
                seed=actual_seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=gen_width,
                height=gen_height,
                guidance=guidance,
                num_inference_steps=steps,
            )
        except Exception as e:
            if self._abort_cb.should_abort or "stop" in str(e).lower() or isinstance(e, KeyboardInterrupt):
                raise RuntimeError("Generierung vom Benutzer abgebrochen (Stop).") from None
            raise e
        elapsed = time.time() - start_time

        # Extract actual PIL Image
        if hasattr(generated, "image") and isinstance(generated.image, Image.Image):
            result_pil = generated.image
        elif hasattr(generated, "to_pil"):
            result_pil = generated.to_pil()
        else:
            result_pil = generated

        out_dir = get_output_dir("t2i")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved_filename = f"t2i_{timestamp}_s{actual_seed}.png"
        saved_path = out_dir / saved_filename
        try:
            if hasattr(generated, "save"):
                generated.save(path=saved_path)
            else:
                result_pil.save(saved_path)

            embed_full_metadata(
                saved_path,
                prompt=prompt,
                model_name="fibo2",
                steps=steps,
                guidance=guidance,
                seed=actual_seed,
                quantize=quant_int,
                width=result_pil.width,
                height=result_pil.height,
            )
        except Exception as e:
            print(f"[MLX Backend] Save metadata error: {e}")
            pass

        meta = {
            "elapsed_seconds": elapsed,
            "seed": actual_seed,
            "width": result_pil.width,
            "height": result_pil.height,
            "steps": steps,
            "saved_path": str(saved_path),
            "backend": "mlx",
        }
        return result_pil, meta

