"""Resolution presets and dimension resolver for FIBO Studio."""

from typing import Optional, Tuple
from PIL import Image

# Common aspect ratios with standard generation resolutions (divisible by 16)
ASPECT_RATIOS = {
    "Original Aspect Ratio": (None, None),
    "1:1 Square (1024 x 1024)": (1024, 1024),
    "16:9 Landscape (1344 x 768)": (1344, 768),
    "9:16 Portrait (768 x 1344)": (768, 1344),
    "4:3 Standard (1152 x 864)": (1152, 864),
    "3:4 Standard Portrait (864 x 1152)": (864, 1152),
    "21:9 Ultrawide (1536 x 640)": (1536, 640),
    "Custom Dimension": (-1, -1),
}


def round_to_16(value: int) -> int:
    """Rounds a dimension value to the nearest multiple of 16."""
    return max(64, round(value / 16) * 16)


def resolve_dimensions(
    preset: str,
    custom_width: int,
    custom_height: int,
    reference_image: Optional[Image.Image] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Calculates final generation width and height based on user choice or source image.
    """
    if preset == "Custom Dimension":
        return round_to_16(custom_width), round_to_16(custom_height)

    target_w, target_h = ASPECT_RATIOS.get(preset, (1024, 1024))

    if preset == "Original Aspect Ratio" and reference_image is not None:
        orig_w, orig_h = reference_image.size
        # Scale while keeping total pixels roughly around 1024x1024 (~1MP)
        scale = (1024 * 1024 / (orig_w * orig_h)) ** 0.5
        target_w = round_to_16(int(orig_w * scale))
        target_h = round_to_16(int(orig_h * scale))
        return target_w, target_h

    if target_w is None or target_h is None:
        return 1024, 1024

    return target_w, target_h


def sanitize_reference_image(image: Image.Image, max_dim: int = 1536) -> Image.Image:
    """
    Sanitizes a reference image for FIBO / Flux VAE and Transformer:
    1. Corrects EXIF orientation (e.g. smartphone/camera portrait photos with orientation tags).
    2. Ensures standard RGB mode.
    3. Bounds very large images while preserving exact aspect ratio.
    4. Guarantees width and height are strictly divisible by 16 (required by Flux2 VAE 8x downsample + 2x2 patchify).
    """
    from PIL import ImageOps
    image = ImageOps.exif_transpose(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    
    # Scale down if maximum dimension exceeds max_dim
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        w = int(w * scale)
        h = int(h * scale)

    # Strictly round to multiple of 16
    new_w = round_to_16(w)
    new_h = round_to_16(h)

    if (new_w, new_h) != image.size:
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return image


def sanitize_mask_image(mask: Optional[Image.Image], target_size: Tuple[int, int]) -> Optional[Image.Image]:
    """Ensures mask matches target image size and is converted to grayscale L."""
    if mask is None:
        return None
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    return mask.convert("L")
