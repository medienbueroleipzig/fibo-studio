"""Vision AI and Tagging module for FIBO Studio on Apple Silicon.
Supports:
1. WD-Tagger (SmilingWolf SwinV2/ViT Danbooru & General Image Tagging on Apple MPS)
2. Qwen2-VL / VLM (Apple Silicon MLX Vision Language Model for natural language image descriptions)
"""

import csv
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)

# Global model caches so models stay in memory between runs
_CACHED_TAGGER_MODEL = None
_CACHED_TAGGER_PROCESSOR = None
_CACHED_TAGGER_TAGS: List[str] = []
_CACHED_TAGGER_CATS: List[str] = []

_CACHED_VLM_MODEL = None
_CACHED_VLM_PROCESSOR = None


def get_cache_root() -> str:
    """Returns the external SSD HuggingFace cache directory if present."""
    ext_hf = "/Volumes/Extreme SSD/ModelBackUp/Hugginface/hub"
    if os.path.exists(ext_hf):
        return ext_hf
    hf_home = os.environ.get("HF_HOME")
    if hf_home and os.path.exists(hf_home):
        return str(Path(hf_home) / "hub")
    return str(Path.home() / ".cache" / "huggingface" / "hub")


# =========================================================================
# 1. WD-TAGGER (SmilingWolf SwinV2 / ViT on Apple MPS)
# =========================================================================
def load_wd_tagger(model_id: str = "SmilingWolf/wd-swinv2-tagger-v3"):
    """Loads and caches the WD-Tagger model on Apple Silicon MPS or CPU."""
    global _CACHED_TAGGER_MODEL, _CACHED_TAGGER_PROCESSOR, _CACHED_TAGGER_TAGS, _CACHED_TAGGER_CATS
    if _CACHED_TAGGER_MODEL is not None:
        return _CACHED_TAGGER_MODEL, _CACHED_TAGGER_PROCESSOR, _CACHED_TAGGER_TAGS, _CACHED_TAGGER_CATS

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    cache_dir = get_cache_root()
    logger.info(f"Loading WD-Tagger {model_id} from {cache_dir}...")

    processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForImageClassification.from_pretrained(model_id, cache_dir=cache_dir)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Load tag dictionary (selected_tags.csv)
    csv_file = hf_hub_download(repo_id=model_id, filename="selected_tags.csv", cache_dir=cache_dir)
    tags, cats = [], []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if row and len(row) > 2:
                tags.append(row[1])
                cats.append(row[2])

    _CACHED_TAGGER_MODEL = model
    _CACHED_TAGGER_PROCESSOR = processor
    _CACHED_TAGGER_TAGS = tags
    _CACHED_TAGGER_CATS = cats
    return model, processor, tags, cats


def tag_image_wd(
    image_input: Union[str, Path, Image.Image],
    threshold: float = 0.35,
    model_id: str = "SmilingWolf/wd-swinv2-tagger-v3",
    include_characters: bool = True,
) -> Tuple[str, str, List[Tuple[str, float]]]:
    """
    Extracts tags using WD-Tagger on Apple Silicon GPU.
    Returns:
        - comma_separated_tags (str)
        - markdown_table (str)
        - raw_tag_list (List[Tuple[str, float]])
    """
    if image_input is None:
        return "", "*Kein Bild geladen.*", []

    start_time = time.time()
    import torch

    model, processor, tags, cats = load_wd_tagger(model_id)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    from PIL import ImageOps

    if isinstance(image_input, (str, Path)):
        with Image.open(str(image_input)) as raw_img:
            img = ImageOps.exif_transpose(raw_img).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = ImageOps.exif_transpose(image_input).convert("RGB")
    else:
        return "", "Ungültiges Bild.", []

    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits[0]).cpu().numpy()

    detected: List[Tuple[str, float, str]] = []
    for i, p in enumerate(probs):
        if p >= threshold:
            category = cats[i] if i < len(cats) else "0"
            # Category 0 = general tag, Category 4 = character
            if category == "0" or (include_characters and category == "4"):
                name = tags[i].replace("_", " ")
                detected.append((name, float(p), "Character" if category == "4" else "General"))

    # Sort descending by confidence
    detected.sort(key=lambda x: x[1], reverse=True)
    elapsed = time.time() - start_time

    tag_names = [t[0] for t in detected]
    comma_tags = ", ".join(tag_names)

    # Build markdown summary
    top_rows = "\n".join([f"| **{t[0]}** | `{t[1]*100:.1f}%` | {t[2]} |" for t in detected[:20]])
    md_output = f"""**🏷️ {len(detected)} Tags gefunden** ({elapsed:.2f}s auf Apple {device.upper()}):

| Tag | Konfidenz | Typ |
| :--- | :--- | :--- |
{top_rows}
"""
    raw_list = [(t[0], t[1]) for t in detected]
    return comma_tags, md_output, raw_list


# =========================================================================
# 2. VISION LANGUAGE MODEL (MLX Native Qwen2-VL)
# =========================================================================
def load_mlx_vlm(model_path: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit"):
    """Loads and caches the MLX Vision Language Model."""
    global _CACHED_VLM_MODEL, _CACHED_VLM_PROCESSOR
    if _CACHED_VLM_MODEL is not None:
        return _CACHED_VLM_MODEL, _CACHED_VLM_PROCESSOR

    from mlx_vlm import load

    logger.info(f"Loading MLX VLM {model_path}...")
    model, processor = load(model_path)
    _CACHED_VLM_MODEL = model
    _CACHED_VLM_PROCESSOR = processor
    return model, processor


def describe_image_vlm(
    image_input: Union[str, Path, Image.Image],
    prompt_mode: str = "Detaillierte Szenenbeschreibung",
    custom_question: Optional[str] = None,
    max_tokens: int = 160,
) -> Tuple[str, str]:
    """
    Analyzes the image using MLX-VLM at ~110 tokens/sec on Apple Silicon.
    Returns: (generated_description, info_markdown)
    """
    if image_input is None:
        return "", "*Kein Bild geladen.*"

    import tempfile
    from mlx_vlm import generate

    start_time = time.time()
    model, processor = load_mlx_vlm("mlx-community/Qwen2-VL-2B-Instruct-4bit")

    # Format instruction prompt
    if custom_question and custom_question.strip():
        prompt = custom_question.strip()
    elif "Kleidung & Personen" in prompt_mode:
        prompt = "Describe the people, their clothing, hairstyles, facial expressions, and poses in detail."
    elif "Kompakt" in prompt_mode:
        prompt = "Provide a concise single-sentence caption describing the main subject and action."
    else:
        prompt = "Describe this image in rich visual detail for an AI image generation prompt, focusing on subject, style, lighting, and composition."

    # MLX VLM image input
    temp_path = None
    if isinstance(image_input, (str, Path)) and Path(image_input).exists():
        in_path = str(image_input)
    else:
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        if isinstance(image_input, Image.Image):
            image_input.save(temp_path)
        in_path = temp_path

    try:
        result = generate(
            model,
            processor,
            prompt=prompt,
            image=in_path,
            max_tokens=max_tokens,
            verbose=False,
        )
        elapsed = time.time() - start_time
        clean_text = result.text.strip() if hasattr(result, "text") else str(result).strip()
        info_md = f"⚡ Generiert in **{elapsed:.2f}s** mit nativer Apple Silicon MLX GPU ({len(clean_text.split())} Wörter)"
        return clean_text, info_md
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

