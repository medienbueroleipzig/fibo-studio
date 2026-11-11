"""Abstract Base Engine Interface for FIBO Studio Backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image


class BaseFiboEngine(ABC):
    """Abstract interface that every hardware backend (CUDA, MLX, CPU) must implement."""

    @abstractmethod
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
        """Executes multi-reference image editing (FIBO 2 Turbo)."""
        pass

    @abstractmethod
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
        """Executes text-to-image generation (FIBO 2 Base/Dev)."""
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Returns list of supported model variant names."""
        pass

    @abstractmethod
    def get_supported_precisions(self) -> List[Tuple[str, str]]:
        """Returns list of (label, value) choices for precision/quantization."""
        pass

    def abort(self) -> None:
        """Interrupts ongoing generation if supported."""
        pass

