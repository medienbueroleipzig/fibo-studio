"""Metadata extraction, embedding, and inspection for generated images (PNG/JPEG/WEBP).
Supports MFlux, FIBO 2.0, ComfyUI, Automatic1111/WebUI, Fooocus, and standard EXIF/XMP.
Includes automatic local outputs history matching if clipboard/browser drops strip metadata.
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import piexif
from PIL import Image, PngImagePlugin
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)


def compute_thumbnail_hash(img: Image.Image) -> str:
    """Computes a fast structural thumbnail hash to recognize images even if metadata is stripped."""
    try:
        small = img.resize((32, 32)).convert("L")
        return hashlib.md5(small.tobytes()).hexdigest()
    except Exception:
        return ""


def embed_full_metadata(
    file_path: Union[str, Path],
    prompt: str,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    model: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> bool:
    """
    Embeds metadata into a saved PNG file across ALL major standards:
    1. Automatic1111 / WebUI / Civitai ('parameters' chunk)
    2. ComfyUI ('prompt' chunk)
    3. MFlux / Camera EXIF (UserComment JSON)
    4. Adobe XMP (XML:com.adobe.xmp)
    """
    effective_model = model or model_name
    path_obj = Path(file_path)
    if not path_obj.exists() or path_obj.suffix.lower() != ".png":
        return False

    try:
        img = Image.open(path_obj)
        existing_info = getattr(img, "info", {}) or {}
        existing_exif = existing_info.get("exif")
        existing_xmp = existing_info.get("XML:com.adobe.xmp")
        existing_iptc = existing_info.get("IPTC")

        w, h = img.size if (not width or not height) else (width, height)

        # 1. Prepare A1111 'parameters' text chunk
        # Format:
        # <Prompt>
        # Steps: 4, Sampler: MLX, CFG scale: 1.0, Seed: 12345, Size: 1024x1024, Model: fibo2-edit-turbo
        param_parts = []
        if steps is not None:
            param_parts.append(f"Steps: {steps}")
        param_parts.append("Sampler: MLX")
        if guidance is not None:
            param_parts.append(f"CFG scale: {guidance}")
        if seed is not None:
            param_parts.append(f"Seed: {seed}")
        param_parts.append(f"Size: {w}x{h}")
        if effective_model:
            param_parts.append(f"Model: {effective_model}")

        param_string = f"{prompt}\n" + ", ".join(param_parts)

        # 2. Build full JSON payload
        meta_dict = {
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "model": effective_model,
            "width": w,
            "height": h,
        }
        if extra_metadata:
            meta_dict.update(extra_metadata)
        meta_json_str = json.dumps(meta_dict, ensure_ascii=False)

        # 3. Construct PngInfo
        pnginfo = PngImagePlugin.PngInfo()

        # Copy any existing tags
        for k, v in existing_info.items():
            if k not in ["parameters", "prompt", "fibo_meta", "Comment", "XML:com.adobe.xmp", "IPTC", "exif"]:
                pnginfo.add_text(k, str(v))

        # Add universal standard tags
        pnginfo.add_text("parameters", param_string)
        pnginfo.add_text("prompt", prompt)
        pnginfo.add_text("fibo_meta", meta_json_str)
        pnginfo.add_text("Comment", prompt)

        if existing_xmp:
            pnginfo.add_text("XML:com.adobe.xmp", existing_xmp)
        if existing_iptc:
            pnginfo.add_text("IPTC", existing_iptc)

        # 4. Prepare EXIF UserComment if not already present
        exif_to_save = existing_exif
        if not exif_to_save:
            try:
                user_comment_bytes = b"ASCII\x00\x00\x00" + meta_json_str.encode("utf-8")
                exif_dict = {"Exif": {piexif.ExifIFD.UserComment: user_comment_bytes}}
                exif_to_save = piexif.dump(exif_dict)
            except Exception:
                exif_to_save = None

        if exif_to_save and isinstance(exif_to_save, (bytes, bytearray)):
            img.save(path_obj, pnginfo=pnginfo, exif=exif_to_save)
        else:
            img.save(path_obj, pnginfo=pnginfo)

        return True
    except Exception as e:
        logger.warning(f"Could not embed extended PNG metadata to {path_obj}: {e}")
        return False


def _find_metadata_in_local_history(img: Image.Image) -> Optional[Dict[str, Any]]:
    """
    Fallback mechanism:
    If the image was pasted from the OS clipboard or dragged from a browser that stripped metadata,
    we compare its structural thumbnail hash with recently generated images in outputs/ to recover metadata.
    """
    try:
        current_hash = compute_thumbnail_hash(img)
        if not current_hash:
            return None

        # Look in outputs/edit and outputs/txt2img
        base_dir = Path(__file__).resolve().parent.parent / "outputs"
        candidate_dirs = [base_dir / "edit", base_dir / "txt2img"]

        for c_dir in candidate_dirs:
            if not c_dir.exists():
                continue
            # Check recent PNG files (sorted by modification time descending)
            png_files = sorted(c_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]
            for p in png_files:
                try:
                    with Image.open(p) as ref_img:
                        if ref_img.size == img.size:
                            ref_hash = compute_thumbnail_hash(ref_img)
                            if ref_hash == current_hash:
                                # Matched! Extract metadata from the original saved file or sidecar
                                sidecar = p.with_suffix(".metadata.json")
                                if sidecar.exists():
                                    with open(sidecar, "r", encoding="utf-8") as f:
                                        data = json.load(f)
                                        return {
                                            "prompt": data.get("prompt", ""),
                                            "seed": data.get("seed"),
                                            "steps": data.get("steps"),
                                            "guidance": data.get("guidance"),
                                            "model": data.get("model"),
                                            "width": data.get("width", img.size[0]),
                                            "height": data.get("height", img.size[1]),
                                            "raw_info": data,
                                            "source_type": f"Lokale Historie ({p.name})",
                                        }
                                # If no sidecar, extract from the file directly
                                meta = extract_image_metadata(p, check_history=False)
                                if meta.get("prompt"):
                                    meta["source_type"] = f"Lokale Historie ({p.name})"
                                    return meta
                except Exception:
                    continue
    except Exception as e:
        logger.debug(f"History search failed: {e}")
    return None


def extract_image_metadata(
    image_input: Union[str, Path, Image.Image],
    check_history: bool = True,
) -> Dict[str, Any]:
    """
    Extracts embedded generation metadata from a PIL Image or file path.
    Checks:
        1. Sidecar .metadata.json / .json (if path provided)
        2. PNG info text chunks: 'parameters' (A1111/Fooocus), 'prompt' (ComfyUI), 'fibo_meta', 'Comment'
        3. EXIF UserComment (MFlux/FIBO JSON, standard EXIF)
        4. Adobe XMP packet (XML:com.adobe.xmp)
        5. Fallback: Local outputs history match (if clipboard or browser stripped metadata)
    """
    result: Dict[str, Any] = {
        "prompt": "",
        "seed": None,
        "steps": None,
        "guidance": None,
        "model": None,
        "width": None,
        "height": None,
        "raw_info": {},
        "source_type": "Unknown",
    }

    try:
        img: Optional[Image.Image] = None
        source_path: Optional[Path] = None

        if isinstance(image_input, (str, Path)):
            source_path = Path(image_input)
            if source_path.exists():
                # 0. Check for sidecar JSON directly next to the image
                sidecar_meta = source_path.with_suffix(".metadata.json")
                if not sidecar_meta.exists():
                    sidecar_meta = source_path.with_suffix(".json")
                if sidecar_meta.exists():
                    try:
                        with open(sidecar_meta, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if "prompt" in data:
                                result["prompt"] = str(data["prompt"])
                                result["seed"] = data.get("seed")
                                result["steps"] = data.get("steps")
                                result["guidance"] = data.get("guidance")
                                result["model"] = data.get("model")
                                result["width"] = data.get("width")
                                result["height"] = data.get("height")
                                result["raw_info"] = data
                                result["source_type"] = "Sidecar JSON (.metadata.json)"
                                return result
                    except Exception:
                        pass
                img = Image.open(source_path)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            return result

        if img is None:
            return result

        result["width"], result["height"] = img.size
        png_info = getattr(img, "info", {}) or {}

        # -------------------------------------------------------------
        # 1. Check PNG Info 'parameters' (A1111 / WebUI / Fooocus)
        # -------------------------------------------------------------
        if "parameters" in png_info:
            params_str = str(png_info["parameters"])
            result["source_type"] = "A1111 / Fooocus (PNG Info)"
            result["raw_info"]["parameters"] = params_str

            prompt_part = params_str
            if "Negative prompt:" in prompt_part:
                prompt_part = prompt_part.split("Negative prompt:")[0]
            elif "Steps:" in prompt_part:
                prompt_part = prompt_part.split("Steps:")[0]
            result["prompt"] = prompt_part.strip()

            seed_m = re.search(r"Seed:\s*(\d+)", params_str)
            steps_m = re.search(r"Steps:\s*(\d+)", params_str)
            cfg_m = re.search(r"CFG scale:\s*([\d\.]+)", params_str)
            model_m = re.search(r"Model:\s*([^,\n]+)", params_str)

            if seed_m:
                result["seed"] = int(seed_m.group(1))
            if steps_m:
                result["steps"] = int(steps_m.group(1))
            if cfg_m:
                result["guidance"] = float(cfg_m.group(1))
            if model_m:
                result["model"] = model_m.group(1).strip()
            return result

        # -------------------------------------------------------------
        # 2. Check PNG Info 'fibo_meta'
        # -------------------------------------------------------------
        if "fibo_meta" in png_info:
            try:
                data = json.loads(str(png_info["fibo_meta"]))
                result["prompt"] = str(data.get("prompt", ""))
                result["seed"] = data.get("seed")
                result["steps"] = data.get("steps")
                result["guidance"] = data.get("guidance")
                result["model"] = data.get("model")
                result["raw_info"] = data
                result["source_type"] = "FIBO Studio (PNG Info)"
                return result
            except Exception:
                pass

        # -------------------------------------------------------------
        # 3. Check PNG Info 'prompt' (ComfyUI graph or plain prompt)
        # -------------------------------------------------------------
        if "prompt" in png_info:
            prompt_content = str(png_info["prompt"])
            if prompt_content.strip().startswith("{"):
                try:
                    graph = json.loads(prompt_content)
                    result["source_type"] = "ComfyUI (Graph)"
                    result["raw_info"] = graph
                    for _, node in graph.items():
                        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
                            text = node.get("inputs", {}).get("text", "")
                            if text and not any(neg in text.lower() for neg in ["bad", "blurry", "ugly", "low quality"]):
                                result["prompt"] = text
                                break
                        elif isinstance(node, dict) and "KSampler" in node.get("class_type", ""):
                            inputs = node.get("inputs", {})
                            if "seed" in inputs:
                                result["seed"] = int(inputs["seed"])
                            if "steps" in inputs:
                                result["steps"] = int(inputs["steps"])
                            if "cfg" in inputs:
                                result["guidance"] = float(inputs["cfg"])

                    if result["prompt"]:
                        return result
                except Exception as e:
                    logger.debug(f"Could not parse ComfyUI prompt JSON: {e}")
            else:
                result["prompt"] = prompt_content.strip()
                result["source_type"] = "PNG Info (prompt)"
                return result

        # -------------------------------------------------------------
        # 4. Check MFlux / FIBO EXIF UserComment (tag 0x9286 / 37510)
        # -------------------------------------------------------------
        exif_bytes = png_info.get("exif")
        if exif_bytes:
            try:
                if isinstance(exif_bytes, str):
                    exif_bytes = exif_bytes.encode("latin-1", errors="ignore")
                if isinstance(exif_bytes, (bytes, bytearray)):
                    exif_dict = piexif.load(exif_bytes)
                    user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
                    if user_comment:
                        if isinstance(user_comment, (bytes, bytearray)):
                            if user_comment.startswith(b"ASCII\x00\x00\x00"):
                                user_comment = user_comment[8:]
                            user_comment = user_comment.decode("utf-8", errors="replace")

                        if isinstance(user_comment, str) and user_comment.strip().startswith("{"):
                            data = json.loads(user_comment)
                            result["raw_info"] = data
                            result["source_type"] = "MFLUX / FIBO (EXIF)"

                            p_val = data.get("prompt", "")
                            if isinstance(p_val, str) and p_val.strip().startswith("{"):
                                try:
                                    sub = json.loads(p_val)
                                    result["prompt"] = (
                                        sub.get("edit_instruction")
                                        or sub.get("caption")
                                        or sub.get("short_description")
                                        or p_val
                                    )
                                except Exception:
                                    result["prompt"] = p_val
                            else:
                                result["prompt"] = str(p_val)

                            result["seed"] = data.get("seed")
                            result["steps"] = data.get("steps") or data.get("num_inference_steps")
                            result["guidance"] = data.get("guidance")
                            result["model"] = data.get("model_config") or data.get("model")
                            return result
            except Exception as e:
                logger.debug(f"Could not parse EXIF UserComment: {e}")

        # -------------------------------------------------------------
        # 5. Check XMP packet (XML:com.adobe.xmp)
        # -------------------------------------------------------------
        xmp_str = png_info.get("XML:com.adobe.xmp")
        if xmp_str and isinstance(xmp_str, str):
            result["source_type"] = "XMP Metadata"
            desc_m = re.search(r"<dc:description>.*?<rdf:li[^>]*>(.*?)</rdf:li>", xmp_str, re.DOTALL)
            if desc_m:
                result["prompt"] = desc_m.group(1).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()

            seed_m = re.search(r"<mflux:seed>(\d+)</mflux:seed>", xmp_str)
            steps_m = re.search(r"<mflux:steps>(\d+)</mflux:steps>", xmp_str)
            cfg_m = re.search(r"<mflux:guidance>([\d\.]+)</mflux:guidance>", xmp_str)
            model_m = re.search(r"<mflux:model>([^<]+)</mflux:model>", xmp_str)

            if seed_m:
                result["seed"] = int(seed_m.group(1))
            if steps_m:
                result["steps"] = int(steps_m.group(1))
            if cfg_m:
                result["guidance"] = float(cfg_m.group(1))
            if model_m:
                result["model"] = model_m.group(1)

            if result["prompt"]:
                return result

        # -------------------------------------------------------------
        # 6. Standard PIL Exif Tags Fallback
        # -------------------------------------------------------------
        try:
            exif = img.getexif()
            if exif:
                for tag_id, val in exif.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    if tag_name in ("UserComment", "ImageDescription", 37510, 270) and val:
                        val_str = str(val).strip()
                        if val_str.startswith("ASCII"):
                            val_str = val_str[5:].strip()
                        if val_str.startswith("{"):
                            try:
                                d = json.loads(val_str)
                                result["prompt"] = d.get("prompt", val_str)
                                result["seed"] = d.get("seed")
                                result["steps"] = d.get("steps")
                                result["guidance"] = d.get("guidance")
                                result["source_type"] = "EXIF Tag"
                                return result
                            except Exception:
                                pass
                        result["prompt"] = val_str
                        result["source_type"] = "EXIF Description"
                        return result
        except Exception:
            pass

        # -------------------------------------------------------------
        # 7. Check PNG 'Comment' or 'Description'
        # -------------------------------------------------------------
        for comment_key in ["Comment", "Description", "description", "comment"]:
            if comment_key in png_info:
                val = str(png_info[comment_key]).strip()
                if val:
                    result["prompt"] = val
                    result["source_type"] = f"PNG Chunk ({comment_key})"
                    return result

        # -------------------------------------------------------------
        # 8. Fallback: Search local outputs history by structural hash
        # (Crucial when browser paste/drop strips EXIF & PNG text chunks)
        # -------------------------------------------------------------
        if check_history and not result["prompt"]:
            history_result = _find_metadata_in_local_history(img)
            if history_result:
                return history_result

    except Exception as e:
        logger.warning(f"Error extracting image metadata: {e}")

    return result


def format_metadata_display(meta: Dict[str, Any]) -> str:
    """Formats the extracted metadata dictionary into a pretty markdown string."""
    prompt = meta.get("prompt") or "*Kein Prompt im Bild gefunden*"
    source = meta.get("source_type") or "Unbekannt"
    seed = meta.get("seed")
    steps = meta.get("steps")
    guidance = meta.get("guidance")
    model = meta.get("model")
    w = meta.get("width")
    h = meta.get("height")

    details = []
    if seed is not None:
        details.append(f"**Seed:** `{seed}`")
    if steps is not None:
        details.append(f"**Steps:** `{steps}`")
    if guidance is not None:
        details.append(f"**Guidance:** `{guidance}`")
    if model:
        details.append(f"**Modell:** `{model}`")
    if w and h:
        details.append(f"**Auflösung:** `{w}x{h}`")

    detail_str = " | ".join(details) if details else "*Keine weiteren Parameter gefunden*"

    return f"""### 🔍 Gefundene Bild-Metadaten ({source})
**Prompt:**
> {prompt}

**Parameter:**
{detail_str}
"""
