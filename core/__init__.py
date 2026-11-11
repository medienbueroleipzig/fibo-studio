"""FIBO Studio Core Module."""

from .engine import FiboEngine, GLOBAL_ENGINE, get_output_dir
from .hardware import detect_hardware, locate_wan_checkpoints, HardwareInfo
from .i18n import t, TRANSLATIONS
from .metadata import embed_full_metadata, extract_image_metadata, format_metadata_display
from .postprocessing import run_postprocessing
from .resolution import ASPECT_RATIOS, resolve_dimensions, round_to_16, sanitize_reference_image, sanitize_mask_image
from .utils import cleanup_mac_files
from .vision import describe_image_vlm, tag_image_wd

__all__ = [
    "FiboEngine",
    "GLOBAL_ENGINE",
    "get_output_dir",
    "detect_hardware",
    "locate_wan_checkpoints",
    "HardwareInfo",
    "t",
    "TRANSLATIONS",
    "embed_full_metadata",
    "extract_image_metadata",
    "format_metadata_display",
    "run_postprocessing",
    "tag_image_wd",
    "describe_image_vlm",
    "ASPECT_RATIOS",
    "resolve_dimensions",
    "round_to_16",
    "sanitize_reference_image",
    "sanitize_mask_image",
    "cleanup_mac_files",
]
