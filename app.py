"""FIBO Studio - Lightweight Gradio WebUI for BRIA FIBO 2.0 on Apple Silicon MLX."""

import json
import os
import platform
import random
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
from PIL import Image, ImageOps

from core import (
    ASPECT_RATIOS,
    GLOBAL_ENGINE,
    HardwareInfo,
    cleanup_mac_files,
    describe_image_vlm,
    detect_hardware,
    extract_image_metadata,
    format_metadata_display,
    get_output_dir,
    locate_wan_checkpoints,
    resolve_dimensions,
    run_postprocessing,
    t,
    tag_image_wd,
)


def get_hub_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home and os.path.exists(hf_home):
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def check_model_ready(repo_id: str) -> Tuple[bool, str]:
    """Checks whether all components of the specified model are downloaded and ready."""
    hub_name = "models--" + repo_id.replace("/", "--")
    hub_path = get_hub_root() / hub_name
    if not hub_path.exists():
        return False, "❌ Nicht auf SSD vorhanden"

    snaps = [d for d in (hub_path / "snapshots").iterdir() if d.is_dir()] if (hub_path / "snapshots").exists() else []
    if not snaps:
        return False, "⚠️ Noch nicht initialisiert"

    snap = snaps[0]
    trans = list((snap / "transformer").glob("*.safetensors")) if (snap / "transformer").exists() else []
    text_enc = list((snap / "text_encoder").glob("*.safetensors")) if (snap / "text_encoder").exists() else []
    vae = list((snap / "vae").glob("*.safetensors")) if (snap / "vae").exists() else []

    blobs = hub_path / "blobs"
    incomplete = list(blobs.glob("*.incomplete")) if blobs.exists() else []

    if len(trans) >= 2 and len(text_enc) >= 1 and len(vae) >= 1:
        return True, "✅ Vollständig installiert & einsatzbereit"
    elif incomplete:
        inc_size = sum(f.stat().st_size for f in incomplete) / (1024**3)
        return False, f"⏳ Download läuft noch ({inc_size:.1f} GB unvollständig)..."
    else:
        return False, f"⚠️ Unvollständig (Transformer: {len(trans)}/2 Shards vorhanden)"


def get_models_overview_markdown() -> str:
    """Generates a clean status overview of downloaded models."""
    hw = detect_hardware()
    if hw.is_cuda:
        ckpts = locate_wan_checkpoints()
        t_path = ckpts.get("transformer_turbo")
        te_path = ckpts.get("text_encoder")
        vae_path = ckpts.get("vae")
        
        status_turbo = f"✅ Ready (`{t_path.name}`)" if t_path else "❌ Not found in WanGP ckpts"
        status_te = f"✅ Ready (`{te_path.name}`)" if te_path else "❌ Not found"
        status_vae = f"✅ Ready (`{vae_path.name}`)" if vae_path else "❌ Not found"
        
        sage_info = "⚡ SageAttention 2 Active" if hw.has_sageattention else ("⚡ FlashAttention Active" if hw.has_flashattention else "PyTorch SDPA")
        
        return f"""
| Component | Status | Location / Details |
| :--- | :--- | :--- |
| **GPU Acceleration** | **{hw.device_name}** | {hw.vram_gb:.1f} GB VRAM • {sage_info} |
| **FIBO 2 Edit Turbo** | {status_turbo} | `{t_path if t_path else 'Missing'}` |
| **Qwen3-VL-4B Encoder** | {status_te} | `{te_path if te_path else 'Missing'}` |
| **Flux2 VAE** | {status_vae} | `{vae_path if vae_path else 'Missing'}` |
"""

    ready_turbo, status_turbo = check_model_ready("briaai/Fibo-2.0-edit-turbo")
    ready_dev, status_dev = check_model_ready("briaai/Fibo-2.0-dev")
    hub_display = str(get_hub_root())
    return f"""
| Model | Function | Status on Disk |
| :--- | :--- | :--- |
| **FIBO 2 Turbo** | Image Edit & Ultra-Fast T2I (4 Steps) | **{status_turbo}** |
| **FIBO 2 Base/Dev** | Text-to-Image Precision (30 Steps) | **{status_dev}** |

*Active Storage Location:* `{hub_display}`
"""


def download_model_clean(model_choice: str, progress=gr.Progress(track_tqdm=False)):
    """Downloads the selected model with clean user progress reporting."""
    is_turbo = "Turbo" in model_choice
    repo_name = "briaai/Fibo-2.0-edit-turbo" if is_turbo else "briaai/Fibo-2.0-dev"
    internal_name = "fibo2-edit-turbo" if is_turbo else "fibo2"

    ready, msg = check_model_ready(repo_name)
    if ready:
        return f"✅ {model_choice} ist bereits vollständig auf deiner SSD installiert!", get_models_overview_markdown()

    progress(0.1, desc=f"Starte Download von {repo_name}...")
    try:
        progress(0.3, desc=f"Lade Modellgewichte ({repo_name})... Dies kann je nach Verbindung einige Minuten dauern.")
        GLOBAL_ENGINE.get_model(model_name=internal_name, quantize=8)
        progress(1.0, desc="Fertig!")
        return f"✅ {model_choice} erfolgreich heruntergeladen und bereit!", get_models_overview_markdown()
    except Exception as e:
        return f"❌ Download fehlgeschlagen: {str(e)}", get_models_overview_markdown()


def get_hf_status():
    """Checks current Hugging Face authentication status."""
    import huggingface_hub
    try:
        user = huggingface_hub.whoami()
        return f"✅ Angemeldet bei Hugging Face als: **{user.get('name', 'User')}**"
    except Exception:
        return "⚠️ Nicht bei Hugging Face angemeldet. Da FIBO 2.0 ein 'Gated Model' von Bria AI ist, wird ein Hugging Face Token benötigt."


def login_hf_token(token: str):
    """Logs in and persists the Hugging Face token."""
    if not token or not token.strip():
        return "❌ Bitte gib ein gültiges Hugging Face Token ein."
    import huggingface_hub
    try:
        clean_token = token.strip()
        huggingface_hub.login(token=clean_token)
        user = huggingface_hub.whoami(token=clean_token)
        return f"✅ Erfolgreich angemeldet als: **{user.get('name', 'User')}**! Du kannst die Modelle jetzt laden."
    except Exception as e:
        return f"❌ Anmeldung fehlgeschlagen: {str(e)}"


def handle_png_info(
    img_input: Union[str, Path, Image.Image, None]
) -> Tuple[str, str, Optional[int], Optional[int], Optional[float]]:
    """Inspects any image and returns formatted Markdown plus individual fields."""
    if img_input is None:
        return "*Kein Bild geladen.*", "", None, None, None
    meta = extract_image_metadata(img_input)
    display_md = format_metadata_display(meta)
    prompt = meta.get("prompt", "")
    seed = meta.get("seed")
    steps = meta.get("steps")
    guidance = meta.get("guidance")
    return display_md, prompt, seed, steps, guidance


def load_prompt_from_image(img_input: Union[str, Path, Image.Image, None]) -> str:
    """Quick extractor returning only the prompt text."""
    if img_input is None:
        return ""
    meta = extract_image_metadata(img_input)
    return meta.get("prompt", "")


def load_all_params_from_image(
    img_input: Union[str, Path, Image.Image, None], cur_steps: int, cur_guidance: float, cur_seed: int
) -> Tuple[str, int, float, int]:
    """Extracts all generation parameters from an image, falling back to current values."""
    if img_input is None:
        return "", cur_steps, cur_guidance, cur_seed
    meta = extract_image_metadata(img_input)
    prompt = meta.get("prompt", "")
    steps = meta.get("steps") if meta.get("steps") is not None else cur_steps
    guidance = meta.get("guidance") if meta.get("guidance") is not None else cur_guidance
    seed = meta.get("seed") if meta.get("seed") is not None else cur_seed
    return prompt, int(steps), float(guidance), int(seed)


def open_outputs_in_finder() -> str:
    """Opens the persistent outputs folder in macOS Finder."""
    try:
        out_root = str(get_output_dir("edit").parent)
        subprocess.run(["open", out_root], check=False)
        return f"📂 Ausgabeordner geöffnet: `{out_root}`"
    except Exception as e:
        return f"⚠️ Fehler beim Öffnen: {e}"


# ==========================================
# USER CUSTOM MODIFIERS PRESETS
# ==========================================
MODIFIERS_FILE = Path(__file__).resolve().parent / "user_modifiers.json"

DEFAULT_MODIFIERS = {
    "Vintage 1970s Kodachrome": "1970s vintage 35mm photograph, authentic Kodachrome color science, warm nostalgic tones, subtle film grain",
    "Cyberpunk Neon Mood": "vibrant neon cyberpunk lighting, cyan and magenta glow, dark moody reflections, atmospheric haze",
    "Cinematic Golden Hour": "shot during golden hour, warm low-angle sunlight, long soft shadows, natural lens flare, rich skin tones",
    "Black & White Noir Film": "dramatic film noir black and white, deep rich blacks, high dynamic contrast, chiaroscuro lighting",
    "Anamorphic Lens Flare": "anamorphic widescreen lens, horizontal blue streak flare, soft oval bokeh, high-end cinema look",
    "Hyper-Realistic Skin Detail": "ultra-realistic macro skin texture, fine pores, subsurface scattering, authentic natural imperfections",
}


def load_user_modifiers() -> Dict[str, str]:
    """Loads custom user modifiers from JSON file."""
    if not MODIFIERS_FILE.exists():
        try:
            with open(MODIFIERS_FILE, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_MODIFIERS, f, indent=2, ensure_ascii=False)
            return DEFAULT_MODIFIERS.copy()
        except Exception as e:
            print(f"[Modifiers] Could not create default modifiers file: {e}")
            return DEFAULT_MODIFIERS.copy()
    try:
        with open(MODIFIERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and data:
                return data
    except Exception as e:
        print(f"[Modifiers] Error reading modifiers file: {e}")
    return DEFAULT_MODIFIERS.copy()


def save_user_modifier(name: str, text: str):
    """Saves a new user modifier and updates dropdown with reliable feedback."""
    clean_name = (name or "").strip()
    clean_text = (text or "").strip()
    mods = load_user_modifiers()

    if not clean_name:
        gr.Warning("Bitte einen Namen für den neuen Modifier angeben.")
        return (
            gr.update(choices=list(mods.keys())),
            gr.update(),
            gr.update(),
            gr.update(visible=True),
        )
    if not clean_text:
        gr.Warning("Bitte einen Prompt-Text für den Modifier angeben.")
        return (
            gr.update(choices=list(mods.keys())),
            gr.update(),
            gr.update(),
            gr.update(visible=True),
        )

    mods[clean_name] = clean_text
    try:
        with open(MODIFIERS_FILE, "w", encoding="utf-8") as f:
            json.dump(mods, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        gr.Info(f"✅ Modifier '{clean_name}' dauerhaft gespeichert!")
    except Exception as e:
        gr.Error(f"Fehler beim Speichern des Modifiers: {e}")

    return (
        gr.update(choices=list(mods.keys()), value=clean_name),
        gr.update(value=""),  # clear name field
        gr.update(value=""),  # clear text field
        gr.update(visible=False),  # close add box
    )


def delete_user_modifier(selected_name: Optional[str]):
    """Deletes the selected user modifier."""
    mods = load_user_modifiers()
    if not selected_name:
        gr.Warning("Kein Modifier zum Löschen ausgewählt.")
        return gr.update(choices=list(mods.keys()))

    if selected_name in mods:
        del mods[selected_name]
        try:
            with open(MODIFIERS_FILE, "w", encoding="utf-8") as f:
                json.dump(mods, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            gr.Info(f"🗑️ Modifier '{selected_name}' gelöscht.")
        except Exception as e:
            gr.Error(f"Fehler beim Löschen: {e}")
    else:
        gr.Warning(f"Modifier '{selected_name}' nicht gefunden.")

    new_choices = list(mods.keys())
    new_val = new_choices[0] if new_choices else None
    return gr.update(choices=new_choices, value=new_val)


def apply_user_modifier(current_prompt: str, selected_name: Optional[str]) -> str:
    """Appends selected modifier text to prompt."""
    if not selected_name:
        gr.Warning("Bitte zuerst einen Modifier aus dem Menü auswählen.")
        return current_prompt or ""
    mods = load_user_modifiers()
    mod_text = mods.get(selected_name, "").strip()
    if not mod_text:
        return current_prompt or ""

    base = (current_prompt or "").strip()
    if not base:
        return mod_text
    if mod_text.lower() in base.lower():
        gr.Info("Dieser Modifier ist bereits im Prompt enthalten.")
        return base
    if base.endswith(","):
        return f"{base} {mod_text}"
    return f"{base}, {mod_text}"


def handle_postprocessing_run(
    img_path: Optional[str],
    method: str,
    scale: float,
    sharpness: float,
    s_res: int,
    s_soft: float,
    face_refine: bool,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], str]:
    """Unified handler for postprocessing & upscaling."""
    if not img_path:
        raise gr.Error("Bitte generiere oder lade zuerst ein Bild.")
    progress(0.2, desc=f"Wende {method} an...")
    try:
        out_path, info = run_postprocessing(
            image_input=img_path,
            method=method,
            scale=float(scale),
            sharpness=float(sharpness),
            seedvr2_resolution=int(s_res),
            seedvr2_softness=float(s_soft),
            face_refine=face_refine,
            face_fidelity=0.7,
        )
        progress(1.0, desc="Fertig!")
        return out_path, info
    except Exception as e:
        raise gr.Error(f"Post-Processing Fehler: {e}")


def handle_vlm_run(
    img_path: Optional[str],
    prompt_mode: str,
    custom_q: str,
    progress=gr.Progress(),
) -> Tuple[str, str]:
    """Runs Apple Silicon MLX VLM image description."""
    if not img_path:
        raise gr.Error("Bitte lade ein Bild für die Vision AI Bilderkennung hoch.")
    progress(0.3, desc="Analysiere Bildinhalt mit nativer Apple MLX GPU...")
    try:
        desc, info = describe_image_vlm(img_path, prompt_mode=prompt_mode, custom_question=custom_q)
        progress(1.0, desc="Fertig!")
        return desc, info
    except Exception as e:
        raise gr.Error(f"Vision AI Fehler: {e}")


def handle_tagger_run(
    img_path: Optional[str],
    threshold: float,
    include_chars: bool,
    progress=gr.Progress(),
) -> Tuple[str, str]:
    """Runs WD-Tagger on Apple Silicon GPU."""
    if not img_path:
        raise gr.Error("Bitte lade ein Bild für das Tagging hoch.")
    progress(0.3, desc="Extrahiere Tags mit WD-Tagger auf Apple GPU...")
    try:
        tags_str, md_table, _ = tag_image_wd(img_path, threshold=float(threshold), include_characters=include_chars)
        progress(1.0, desc="Fertig!")
        return tags_str, md_table
    except Exception as e:
        raise gr.Error(f"Tagging Fehler: {e}")


def handle_edit(
    img1: Optional[Image.Image],
    img2: Optional[Image.Image],
    img3: Optional[Image.Image],
    img4: Optional[Image.Image],
    img5: Optional[Image.Image],
    mask: Optional[Image.Image],
    prompt: str,
    aspect_preset: str,
    custom_w: int,
    custom_h: int,
    quantize: int,
    steps: int,
    guidance: float,
    seed: int,
    randomize_seed: bool,
    lora_path: str,
    lora_scale: float,
    progress=gr.Progress(track_tqdm=True),
):
    """Executes the FIBO 2 Edit Turbo pipeline."""
    if img1 is None:
        raise gr.Error("Bitte lade mindestens Reference Image 1 (Basis) hoch.")

    if not prompt or not prompt.strip():
        raise gr.Error("Bitte gib einen Edit-Prompt ein.")

    ready, status_msg = check_model_ready("briaai/Fibo-2.0-edit-turbo")
    if not ready:
        progress(0.05, desc=f"Modell noch nicht komplett lokal ({status_msg}). Lade fehlende Daten...")

    # Collect reference images in order
    ref_images = [img for img in [img1, img2, img3, img4, img5] if img is not None]

    # Resolve dimensions
    width, height = resolve_dimensions(
        preset=aspect_preset,
        custom_width=custom_w,
        custom_height=custom_h,
        reference_image=img1,
    )

    actual_seed = random.randint(0, 2**32 - 1) if randomize_seed else int(seed)

    progress(0.2, desc="Initialisiere FIBO 2 Turbo in Apple MLX...")
    try:
        progress(0.4, desc="Generiere Bild (4 Steps SFT)...")
        result_img, elapsed, info = GLOBAL_ENGINE.run_edit(
            images=ref_images,
            prompt=prompt.strip(),
            mask=mask,
            model_name="fibo2-edit-turbo",
            quantize=int(quantize),
            width=width,
            height=height,
            steps=int(steps),
            guidance=float(guidance),
            seed=actual_seed,
            lora_path=lora_path.strip() if lora_path else None,
            lora_scale=float(lora_scale),
        )
        progress(1.0, desc="Fertig!")
        return result_img, info, actual_seed
    except Exception as e:
        err_msg = str(e)
        if "abgebrochen" in err_msg.lower() or "stop" in err_msg.lower():
            gr.Warning("Inferenz abgebrochen.")
            return None, "*Inferenz vom Benutzer abgebrochen.*", actual_seed
        if "gated" in err_msg.lower() or "401" in err_msg or "restricted" in err_msg.lower():
            raise gr.Error(
                "Hugging Face Zugriff verweigert! "
                "Bitte stelle sicher, dass du auf https://huggingface.co/briaai/Fibo-2.0-edit-turbo "
                "die Bedingungen akzeptiert hast und dein HF-Token oben unter 'Hugging Face Authentifizierung' gespeichert hast."
            )
        raise gr.Error(f"Edit fehlgeschlagen: {err_msg}")


VALID_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".jfif"}


def sanitize_folder_path(raw: Optional[str]) -> str:
    """Cleans up folder paths, strips quotes/spaces, and auto-corrects duplicate path prefixes."""
    if not raw or not isinstance(raw, str):
        return ""
    clean = raw.strip().strip("'\"")
    clean = os.path.expanduser(clean)
    if os.path.isdir(clean):
        return str(Path(clean).resolve())

    # Detect accidental duplicate path prefixes (e.g. /Users/jaf/Desktop/Users/jaf/Desktop/stills)
    for marker in ["/Users/", "/Volumes/", "/System/", "/private/"]:
        idx = clean.rfind(marker)
        if idx > 0:
            candidate = clean[idx:]
            if os.path.isdir(candidate):
                return str(Path(candidate).resolve())

    return clean


def select_folder_via_finder(current_val: str = "") -> str:
    """Opens a native macOS folder selection dialog via AppleScript."""
    try:
        cmd = [
            "osascript",
            "-e",
            "try",
            "-e",
            'set chosen to POSIX path of (choose folder with prompt "Ordner mit Bildern für FIBO Studio auswählen:")',
            "-e",
            "return chosen",
            "-e",
            "on error",
            "-e",
            'return ""',
            "-e",
            "end try",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = res.stdout.strip()
        if out:
            clean = sanitize_folder_path(out)
            if os.path.isdir(clean):
                return clean
    except Exception as e:
        print(f"Error opening folder picker: {e}")
    return current_val


def scan_bulk_inputs(
    folder_str: Optional[str],
    uploaded_files: Optional[List[Any]],
    recursive: bool = False,
) -> Tuple[List[str], str]:
    """Scans a local directory and/or uploaded files for genuine images.

    Filters out non-images (HTML, TXT, DS_Store, etc.) safely without crashing.
    Returns (valid_image_paths, status_markdown).
    """
    valid_paths: List[str] = []
    skipped_items: List[Tuple[str, str]] = []
    seen = set()

    clean_dir = sanitize_folder_path(folder_str) if folder_str else ""

    # 1. Local folder scanning
    if clean_dir:
        p_dir = Path(clean_dir)
        if p_dir.is_dir():
            pattern = "**/*" if recursive else "*"
            for item in sorted(p_dir.glob(pattern)):
                if not item.is_file():
                    continue
                if item.name.startswith("._") or item.name.startswith("."):
                    continue
                if item.suffix.lower() not in VALID_IMG_EXTENSIONS:
                    skipped_items.append((item.name, f"Format '{item.suffix}' übersprungen (kein Bild)"))
                    continue
                try:
                    with Image.open(item) as test_img:
                        test_img.verify()
                    canonical = str(item.resolve())
                    if canonical not in seen:
                        seen.add(canonical)
                        valid_paths.append(canonical)
                except Exception as e:
                    skipped_items.append((item.name, f"Ungültiges/korruptes Bild ({e})"))
        else:
            skipped_items.append((clean_dir, "Ordnerpfad existiert nicht"))
    elif folder_str and folder_str.strip():
        # folder_str was provided but could not be resolved to an existing directory
        raw_val = folder_str.strip().strip("'\"")
        skipped_items.append((raw_val, "Ordnerpfad existiert nicht"))

    # 2. Uploaded files (drag & drop / multi-file)
    if uploaded_files:
        for f in uploaded_files:
            f_path_str = f if isinstance(f, str) else getattr(f, "name", str(f))
            item = Path(f_path_str)
            if not item.exists() or not item.is_file():
                continue
            if item.name.startswith("._") or item.name.startswith("."):
                continue
            if item.suffix.lower() not in VALID_IMG_EXTENSIONS:
                skipped_items.append((item.name, f"Format '{item.suffix}' übersprungen (z.B. HTML/Text)"))
                continue
            try:
                with Image.open(item) as test_img:
                    test_img.verify()
                canonical = str(item.resolve())
                if canonical not in seen:
                    seen.add(canonical)
                    valid_paths.append(canonical)
            except Exception as e:
                skipped_items.append((item.name, f"Ungültig ({e})"))

    if not valid_paths:
        if skipped_items:
            reasons = "<br>".join([f"• `{name}`: {reason}" for name, reason in skipped_items[:12]])
            if len(skipped_items) > 12:
                reasons += f"<br>... und {len(skipped_items)-12} weitere Dateien."
            return [], f"⚠️ **Keine gültigen Bilder gefunden!**<br>Übersprungene Dateien:<br>{reasons}"
        return [], "*Keine Dateien oder Ordner ausgewählt.*"

    skipped_note = ""
    if skipped_items:
        # Only mention skipped if they are non-images (not folder path miss)
        non_img_skipped = [s for s in skipped_items if s[1] != "Ordnerpfad existiert nicht"]
        if non_img_skipped:
            skipped_note = f" (ℹ️ *{len(non_img_skipped)} Nicht-Bild-Dateien automatisch gefiltert*)"

    source_info = f"aus `{clean_dir}`" if clean_dir else "aus Upload"
    status_md = f"✅ **{len(valid_paths)} gültige Bilder bereit** ({source_info}).{skipped_note}"
    return valid_paths, status_md


def render_bulk_thumbnails_html(file_paths: List[str]) -> str:
    """Renders a grid of thumbnail cards with a distinct [X] removal button on each image."""
    if not file_paths:
        return ""
    items_html = []
    for idx, p_str in enumerate(file_paths):
        p = Path(p_str)
        canonical = str(p.resolve())
        url_escaped_path = urllib.parse.quote(canonical)
        img_src = f"/gradio_api/file={url_escaped_path}"
        filename = p.name
        display_name = filename if len(filename) <= 14 else (filename[:9] + "…" + p.suffix)
        js_escaped_path = canonical.replace("\\", "\\\\").replace("'", "\\'")
        card = (
            f'<div class="bulk-thumb-item" id="bulk-thumb-{idx}">'
            f'<button type="button" class="bulk-thumb-remove-btn" title="Bild \'{filename}\' entfernen" onclick="removeBulkImage(\'{js_escaped_path}\', {idx})">✕</button>'
            f'<img class="bulk-thumb-img" src="{img_src}" alt="{filename}" loading="lazy" />'
            f'<span class="bulk-thumb-filename" title="{filename}">{display_name}</span>'
            f'</div>'
        )
        items_html.append(card)

    return f'<div class="bulk-thumb-grid">{"".join(items_html)}</div>'


def preview_bulk_inputs(
    folder_str: Optional[str],
    uploaded_files: Optional[List[Any]],
    recursive: bool = False,
) -> Tuple[List[str], str, str]:
    """Callback for checking and previewing inputs before running bulk edit."""
    paths, status_md = scan_bulk_inputs(folder_str, uploaded_files, recursive)
    grid_html = render_bulk_thumbnails_html(paths)
    return paths, grid_html, status_md


def handle_bulk_edit(
    active_paths: List[str],
    folder_str: Optional[str],
    uploaded_files: Optional[List[Any]],
    recursive: bool,
    img2: Optional[Image.Image],
    img3: Optional[Image.Image],
    img4: Optional[Image.Image],
    img5: Optional[Image.Image],
    mask: Optional[Image.Image],
    prompt: str,
    aspect_preset: str,
    custom_w: int,
    custom_h: int,
    quantize: int,
    steps: int,
    guidance: float,
    seed: int,
    randomize_seed: bool,
    lora_path: str,
    lora_scale: float,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[List[Tuple[str, str]], str]:
    """Executes FIBO 2 Turbo Edit sequentially across a batch of images.

    Each image is assigned to <image_1>, while retaining shared reference images
    (<image_2> - <image_5>) and prompt.
    """
    if not prompt or not prompt.strip():
        raise gr.Error("Bitte gib einen Edit-Prompt für die Stapelverarbeitung ein.")

    # Use active_paths (user-adjusted list) if non-empty, otherwise scan
    valid_paths = [p for p in active_paths if os.path.exists(p)] if active_paths else []
    if not valid_paths:
        valid_paths, _ = scan_bulk_inputs(folder_str, uploaded_files, recursive)

    if not valid_paths:
        raise gr.Error("Keine gültigen Bilddateien gefunden. Bitte wähle einen Ordner oder lade Bilddateien hoch.")

    ready, status_msg = check_model_ready("briaai/Fibo-2.0-edit-turbo")
    if not ready:
        progress(0.02, desc=f"Modell noch nicht komplett lokal ({status_msg}). Lade fehlende Daten...")

    total = len(valid_paths)
    results = []
    failed = []
    start_total = time.time()

    extra_refs = [img for img in [img2, img3, img4, img5] if img is not None]

    for idx, path_str in enumerate(valid_paths):
        p = Path(path_str)
        progress((idx) / total, desc=f"Stapelverarbeitung [{idx+1}/{total}]: {p.name}...")
        try:
            with Image.open(p) as raw_img:
                curr_img = ImageOps.exif_transpose(raw_img).convert("RGB")

            # Resolve dimensions - adapting to each image's native aspect ratio if 'Original' selected
            width, height = resolve_dimensions(
                preset=aspect_preset,
                custom_width=custom_w,
                custom_height=custom_h,
                reference_image=curr_img,
            )

            all_refs = [curr_img] + extra_refs
            actual_seed = random.randint(0, 2**32 - 1) if randomize_seed else int(seed) + idx

            out_img, elapsed, _ = GLOBAL_ENGINE.run_edit(
                images=all_refs,
                prompt=prompt.strip(),
                mask=mask,
                model_name="fibo2-edit-turbo",
                quantize=int(quantize),
                width=width,
                height=height,
                steps=int(steps),
                guidance=float(guidance),
                seed=actual_seed,
                lora_path=lora_path.strip() if lora_path else None,
                lora_scale=float(lora_scale),
            )
            saved_path_str = str(out_img)
            results.append((saved_path_str, f"{p.name} ({elapsed:.1f}s)"))
        except Exception as e:
            err_msg = str(e)
            if "abgebrochen" in err_msg.lower() or "stop" in err_msg.lower() or isinstance(e, KeyboardInterrupt):
                gr.Warning("Stapelverarbeitung vom Benutzer abgebrochen.")
                failed.append((p.name, "Vom Benutzer abgebrochen"))
                break
            failed.append((p.name, err_msg))

    total_time = time.time() - start_total
    avg_time = total_time / len(results) if results else 0
    out_dir = str(get_output_dir("edit"))

    summary_lines = [
        "### 🎉 Stapelverarbeitung abgeschlossen!",
        f"* **Erfolgreich verarbeitet:** {len(results)} von {total} Bildern",
        f"* **Gesamtdauer:** {total_time:.1f}s (Ø {avg_time:.2f}s pro Bild)",
        f"* **Speicherort:** `{out_dir}`",
    ]
    if failed:
        summary_lines.append(f"<br>⚠️ **{len(failed)} Fehler aufgetreten:**")
        for fn, err in failed[:8]:
            summary_lines.append(f"• `{fn}`: {err}")

    progress(1.0, desc="Stapelverarbeitung beendet!")
    return results, "\n".join(summary_lines)


def handle_txt2img(
    prompt: str,
    model_choice: str,
    expand_prompt: bool,
    aspect_preset: str,
    custom_w: int,
    custom_h: int,
    quantize: int,
    steps: int,
    guidance: float,
    seed: int,
    randomize_seed: bool,
    lora_path: str,
    lora_scale: float,
    progress=gr.Progress(track_tqdm=True),
):
    """Executes the FIBO 2 Text-to-Image pipeline."""
    if not prompt or not prompt.strip():
        raise gr.Error("Bitte gib einen Text-Prompt ein.")

    model_name = "fibo2-edit-turbo" if "Turbo" in model_choice else "fibo2"
    repo_id = "briaai/Fibo-2.0-edit-turbo" if "Turbo" in model_choice else "briaai/Fibo-2.0-dev"

    ready, status_msg = check_model_ready(repo_id)
    if not ready:
        progress(0.05, desc=f"Modell noch nicht komplett lokal ({status_msg}). Lade fehlende Daten...")

    # Resolve dimensions
    width, height = resolve_dimensions(
        preset=aspect_preset if aspect_preset != "Original Aspect Ratio" else "1:1 Square (1024 x 1024)",
        custom_width=custom_w,
        custom_height=custom_h,
    )

    actual_seed = random.randint(0, 2**32 - 1) if randomize_seed else int(seed)

    progress(0.2, desc=f"Initialisiere {model_name} in Apple MLX...")
    try:
        progress(0.4, desc=f"Generiere Bild ({steps} Steps)...")
        result_img, elapsed, info = GLOBAL_ENGINE.run_text_to_image(
            prompt=prompt.strip(),
            model_name=model_name,
            quantize=int(quantize),
            width=width,
            height=height,
            steps=int(steps),
            guidance=float(guidance),
            seed=actual_seed,
            expand_prompt=bool(expand_prompt),
            lora_path=lora_path.strip() if lora_path else None,
            lora_scale=float(lora_scale),
        )
        progress(1.0, desc="Fertig!")
        return result_img, info, actual_seed
    except Exception as e:
        err_msg = str(e)
        if "abgebrochen" in err_msg.lower() or "stop" in err_msg.lower():
            gr.Warning("Inferenz abgebrochen.")
            return None, "*Inferenz vom Benutzer abgebrochen.*", actual_seed
        if "gated" in err_msg.lower() or "401" in err_msg or "restricted" in err_msg.lower():
            raise gr.Error(
                f"Hugging Face Zugriff verweigert! Bitte stelle sicher, dass du auf https://huggingface.co/{repo_id} die Bedingungen akzeptiert hast."
            )
        raise gr.Error(f"Generierung fehlgeschlagen: {err_msg}")


def toggle_custom_dims(preset: str):
    """Show/hide custom dimension sliders based on dropdown."""
    is_custom = preset == "Custom Dimension"
    return gr.update(visible=is_custom)


def update_t2i_defaults(model_choice: str):
    """Automatically adjust default steps and guidance when model is changed."""
    if "Turbo" in model_choice:
        return gr.update(value=4, maximum=20), gr.update(value=1.0)
    else:
        return gr.update(value=30, maximum=60), gr.update(value=3.5)


# UI Theme & Styling (Compact Dark Theme with Orange Studio Accent & 0px Sharp Radius)
def load_app_theme():
    try:
        theme = gr.Theme.from_hub("deepkyu/compact-theme")
        return theme
    except Exception:
        return gr.themes.Default(
            primary_hue=gr.themes.colors.orange,
            secondary_hue=gr.themes.colors.neutral,
            neutral_hue=gr.themes.colors.neutral,
            font=[gr.themes.GoogleFont("Gabarito"), "sans-serif"],
            font_mono=[gr.themes.GoogleFont("Roboto Mono"), "monospace"],
            radius_size=gr.themes.sizes.radius_none,
        )

app_theme = load_app_theme()

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Gabarito:wght@400;500;600;700;800&family=Roboto+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-dark: #0b0f19;
    --bg-card: #111625;
    --bg-input: #080c14;
    --border-color: #1e293b;
    --orange-primary: #fe7c00;
    --orange-hover: #d0670c;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
}

/* Force dark mode canvas and Gabarito typography */
html, body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #f8fafc !important;
    font-family: 'Gabarito', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* STRICT ZERO RADIUS: No rounded corners on any element */
*, *::before, *::after,
.gr-button, button, input, select, textarea,
.gr-box, .gr-panel, .gr-form, .block,
.tabs, .tab-nav, .tabitem,
.gr-image, img, .image-frame,
.accordion, .accordion-header,
.slider, .radio, .checkbox,
.dropdown, span, div {
    border-radius: 0px !important;
}

/* Orange Studio Accent Styling */
button.primary, .primary-btn, button[variant="primary"] {
    background: #fe7c00 !important;
    background-color: #fe7c00 !important;
    border: 1px solid #d0670c !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    transition: background 0.15s ease, transform 0.05s ease !important;
}

button.primary:hover, button[variant="primary"]:hover {
    background: #d0670c !important;
    background-color: #d0670c !important;
    border-color: #fe7c00 !important;
}

button.secondary, button[variant="secondary"] {
    background: #172033 !important;
    border: 1px solid #2d3b55 !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

button.secondary:hover, button[variant="secondary"]:hover {
    background: #1e2c47 !important;
    border-color: #fe7c00 !important;
    color: #fe7c00 !important;
}

/* Sharp Card & Block Borders */
.block, .form, .panel, .gr-panel {
    background: #111625 !important;
    border: 1px solid #1e293b !important;
}

/* Input Fields */
input, textarea, select, .gr-input {
    background-color: #080c14 !important;
    border: 1px solid #1e293b !important;
    color: #f8fafc !important;
    font-family: 'Gabarito', sans-serif !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #fe7c00 !important;
    box-shadow: 0 0 0 1px #fe7c00 !important;
}

/* Tabs */
.tabs > .tab-nav > button {
    font-family: 'Gabarito', sans-serif !important;
    font-weight: 600 !important;
    border-bottom: 2px solid transparent !important;
    color: #94a3b8 !important;
}

.tabs > .tab-nav > button.selected {
    color: #fe7c00 !important;
    border-bottom: 2px solid #fe7c00 !important;
    background: rgba(254, 124, 0, 0.08) !important;
}

/* Monospace areas */
.info-box, pre, code, .font-mono {
    font-family: 'Roboto Mono', monospace !important;
    background: #080c14 !important;
    border: 1px solid #1e293b !important;
    color: #cbd5e1 !important;
}
.main-container { max-width: 1440px; margin: 0 auto; }
.generate-btn { font-size: 1.15rem !important; font-weight: 700 !important; padding: 14px !important; }
.status-box { background: rgba(254, 124, 0, 0.05); border: 1px solid rgba(254, 124, 0, 0.3); padding: 12px 16px; margin-bottom: 12px; }

/* Prompt Shortcuts Toolbar */
.prompt-shortcut-toolbar {
    display: flex;
    flex-wrap: nowrap;
    align-items: center;
    gap: 6px;
    margin-top: 4px;
    margin-bottom: 6px;
    padding: 6px 8px;
    background: #0e1422;
    border: 1px solid #1e293b;
    border-radius: 0px !important;
    overflow-x: auto;
}

.shortcut-toolbar-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-right: 6px;
    white-space: nowrap;
    user-select: none;
}

.shortcut-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    min-width: 32px;
    padding: 0;
    margin: 0;
    background: #172033;
    border: 1px solid #2d3b55;
    color: #94a3b8;
    cursor: pointer;
    border-radius: 0px !important;
    transition: all 0.15s ease-in-out;
}

.shortcut-btn svg {
    width: 16px;
    height: 16px;
    fill: currentColor;
    pointer-events: none;
}

.shortcut-btn:hover {
    background: #1e2c47;
    border-color: #fe7c00;
    color: #fe7c00;
    transform: translateY(-1px);
}

.shortcut-btn.active {
    background: #fe7c00 !important;
    background-color: #fe7c00 !important;
    border-color: #d0670c !important;
    color: #ffffff !important;
    box-shadow: 0 0 8px rgba(254, 124, 0, 0.5) !important;
}

#hidden_bulk_bridge_row {
    display: none !important;
}

/* Bulk Thumbnail Preview Grid with Remove [X] */
.bulk-thumb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(105px, 1fr));
    gap: 8px;
    max-height: 290px;
    overflow-y: auto;
    padding: 8px;
    background: #080c14;
    border: 1px solid #1e293b;
    border-radius: 0px !important;
    margin-top: 6px;
    margin-bottom: 8px;
}

.bulk-thumb-item {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    background: #111625;
    border: 1px solid #1e293b;
    border-radius: 0px !important;
    padding: 3px;
    transition: border-color 0.15s ease, transform 0.1s ease;
}

.bulk-thumb-item:hover {
    border-color: #fe7c00;
}

.bulk-thumb-img {
    width: 100%;
    height: 70px;
    object-fit: cover;
    background: #000;
    display: block;
    border-radius: 0px !important;
}

.bulk-thumb-remove-btn {
    position: absolute;
    top: 2px;
    right: 2px;
    width: 22px;
    height: 22px;
    background: rgba(220, 38, 38, 0.88) !important;
    color: #ffffff !important;
    border: 1px solid #ef4444 !important;
    font-size: 13px !important;
    font-weight: 800 !important;
    line-height: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    border-radius: 0px !important;
    z-index: 10;
    transition: all 0.15s ease !important;
    padding: 0 !important;
    margin: 0 !important;
}

.bulk-thumb-remove-btn:hover {
    background: #ef4444 !important;
    transform: scale(1.18);
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.85);
}

.bulk-thumb-filename {
    font-size: 0.65rem;
    color: #94a3b8;
    margin-top: 3px;
    width: 100%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: center;
    font-family: 'Roboto Mono', monospace;
}

/* Custom Modifiers Dropdown Bar & Creation Box */
.custom-modifier-bar {
    align-items: flex-end !important;
    gap: 8px !important;
    margin-top: 2px !important;
    margin-bottom: 6px !important;
    background: #090e18 !important;
    border: 1px solid #1e293b !important;
    border-radius: 0px !important;
    padding: 8px 10px !important;
}

.custom-mod-add-box {
    background: #0f172a !important;
    border: 1px solid #fe7c00 !important;
    border-radius: 0px !important;
    padding: 10px 12px !important;
    margin-bottom: 8px !important;
}

/* Floating Navigation Arrows for Gallery Fullscreen / Preview */
.gallery-float-nav {
    position: absolute !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: 48px !important;
    height: 48px !important;
    border-radius: 50% !important;
    background: rgba(11, 15, 25, 0.85) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1.5px solid #fe7c00 !important;
    color: #fe7c00 !important;
    font-size: 24px !important;
    font-weight: 900 !important;
    line-height: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    z-index: 999999 !important;
    user-select: none !important;
    transition: all 0.15s ease-in-out !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6) !important;
}

.gallery-float-nav:hover {
    background: #fe7c00 !important;
    color: #0b0f19 !important;
    transform: translateY(-50%) scale(1.12) !important;
    box-shadow: 0 0 18px rgba(254, 124, 0, 0.8) !important;
}

.gallery-float-nav.prev {
    left: 18px !important;
}

.gallery-float-nav.next {
    right: 18px !important;
}
"""

HEAD_SCRIPTS = r"""
<script>
window.togglePromptDirective = function(btn, directive) {
    const dir = directive || btn.getAttribute('data-directive');
    if (!dir) return;
    const container = document.getElementById('edit_prompt');
    const textarea = container ? container.querySelector('textarea') : document.querySelector('#edit_prompt textarea');
    if (!textarea) return;

    const isActive = btn.classList.contains('active');
    let currentVal = textarea.value.trim();

    if (!isActive) {
        btn.classList.add('active');
        if (currentVal.length > 0) {
            if (currentVal.endsWith(',')) {
                textarea.value = currentVal + ' ' + dir;
            } else {
                textarea.value = currentVal + ', ' + dir;
            }
        } else {
            textarea.value = dir;
        }
    } else {
        btn.classList.remove('active');
        const esc = dir.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const reg = new RegExp('(\\s*,\\s*' + esc + ')|(' + esc + '\\s*,?\\s*)', 'gi');
        let newVal = textarea.value.replace(reg, '');
        newVal = newVal.replace(/^\s*,\s*/, '').replace(/\s*,\s*$/, '').trim();
        textarea.value = newVal;
    }

    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    textarea.dispatchEvent(new Event('change', { bubbles: true }));
};

document.addEventListener('input', function(e) {
    if (e.target && e.target.tagName === 'TEXTAREA' && e.target.closest('#edit_prompt')) {
        const val = e.target.value.toLowerCase();
        document.querySelectorAll('.shortcut-btn').forEach(function(btn) {
            const dir = (btn.getAttribute('data-directive') || '').toLowerCase();
            if (dir && val.includes(dir)) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }
});

window.removeBulkImage = function(targetPath, idx) {
    const card = document.getElementById('bulk-thumb-' + idx);
    if (card) {
        card.style.opacity = '0.3';
        card.style.transform = 'scale(0.92)';
        card.style.pointerEvents = 'none';
    }
    const box = document.querySelector('#hidden_bulk_remove_path textarea') || document.querySelector('#hidden_bulk_remove_path input');
    const btnContainer = document.querySelector('#btn_trigger_bulk_remove');
    if (box && btnContainer) {
        const proto = box instanceof HTMLTextAreaElement ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
        const descriptor = Object.getOwnPropertyDescriptor(proto, 'value');
        if (descriptor && descriptor.set) {
            descriptor.set.call(box, targetPath);
        } else {
            box.value = targetPath;
        }
        box.dispatchEvent(new Event('input', { bubbles: true }));
        box.dispatchEvent(new Event('change', { bubbles: true }));
        const realBtn = btnContainer.tagName === 'BUTTON' ? btnContainer : btnContainer.querySelector('button');
        if (realBtn) {
            realBtn.click();
        } else {
            btnContainer.click();
        }
    }
};

/* Floating Navigation Arrows for Gallery Fullscreen / Preview */
(function() {
    function triggerGalleryNav(container, direction) {
        const code = direction === 'next' ? 'ArrowRight' : 'ArrowLeft';
        const key = code;
        const keyCode = direction === 'next' ? 39 : 37;
        const evt = new KeyboardEvent('keydown', {
            key: key,
            code: code,
            keyCode: keyCode,
            which: keyCode,
            bubbles: true,
            cancelable: true
        });
        window.dispatchEvent(evt);
        document.dispatchEvent(evt);

        const gallery = container.closest('.gr-gallery') || container.closest('.gallery') || document.querySelector('#bulk_gallery') || document;
        const thumbs = Array.from(gallery.querySelectorAll('.thumbnails button, button.thumbnail-item, button.thumbnail-small'));
        if (thumbs.length > 1) {
            let activeIdx = thumbs.findIndex(t => t.classList.contains('selected') || t.getAttribute('aria-selected') === 'true');
            if (activeIdx !== -1) {
                let nextIdx = direction === 'next' ? activeIdx + 1 : activeIdx - 1;
                if (nextIdx < 0) nextIdx = thumbs.length - 1;
                if (nextIdx >= thumbs.length) nextIdx = 0;
                thumbs[nextIdx].click();
            }
        }
    }

    function injectGalleryNav(container) {
        if (!container || container.querySelector('.gallery-float-nav.prev')) return;

        const prevBtn = document.createElement('div');
        prevBtn.className = 'gallery-float-nav prev';
        prevBtn.innerHTML = '&#10094;';
        prevBtn.title = 'Vorheriges Bild (Pfeiltaste Links)';
        prevBtn.onclick = function(e) {
            e.stopPropagation();
            e.preventDefault();
            triggerGalleryNav(container, 'prev');
        };

        const nextBtn = document.createElement('div');
        nextBtn.className = 'gallery-float-nav next';
        nextBtn.innerHTML = '&#10095;';
        nextBtn.title = 'Nächstes Bild (Pfeiltaste Rechts)';
        nextBtn.onclick = function(e) {
            e.stopPropagation();
            e.preventDefault();
            triggerGalleryNav(container, 'next');
        };

        const computedStyle = window.getComputedStyle(container);
        if (computedStyle.position === 'static') {
            container.style.position = 'relative';
        }

        container.appendChild(prevBtn);
        container.appendChild(nextBtn);
    }

    const observer = new MutationObserver(function() {
        const targets = document.querySelectorAll(
            '#bulk_gallery .preview, #bulk_gallery .detailed-view, .gr-gallery .preview, .gr-gallery .detailed-view, div[data-testid="detailed-image"], .image-preview'
        );
        targets.forEach(function(el) {
            injectGalleryNav(el);
        });
    });

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            observer.observe(document.body, { childList: true, subtree: true });
        });
    } else {
        observer.observe(document.body, { childList: true, subtree: true });
    }
})();
</script>
"""

EDIT_SHORTCUTS_TOOLBAR_HTML = """
<div class="prompt-shortcut-toolbar" role="toolbar" aria-label="Prompt Shortcuts Toolbar">
    <span class="shortcut-toolbar-label">MODIFIER:</span>
    <button type="button" class="shortcut-btn" title="Enhance Image: Enhance the image" data-directive="Enhance the image" onclick="togglePromptDirective(this)">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M7.657 6.247c.11-.33.576-.33.686 0l.645 1.937a2.89 2.89 0 0 0 1.829 1.828l1.936.645c.33.11.33.576 0 .686l-1.937.645a2.89 2.89 0 0 0-1.828 1.829l-.645 1.936a.361.361 0 0 1-.686 0l-.645-1.937a2.89 2.89 0 0 0-1.828-1.828l-1.937-.645a.361.361 0 0 1 0-.686l1.937-.645a2.89 2.89 0 0 0 1.828-1.828zM3.794 1.148a.217.217 0 0 1 .412 0l.387 1.162c.173.518.579.924 1.097 1.097l1.162.387a.217.217 0 0 1 0 .412l-1.162.387A2.17 2.17 0 0 0 4.593 5.69l-.387 1.162a.217.217 0 0 1-.412 0L3.407 5.69A2.17 2.17 0 0 0 2.31 4.593l-1.162-.387a.217.217 0 0 1 0-.412l1.162-.387A2.17 2.17 0 0 0 3.407 2.31zM10.863.099a.145.145 0 0 1 .274 0l.258.774c.115.346.386.617.732.732l.774.258a.145.145 0 0 1 0 .274l-.774.258a1.45 1.45 0 0 0-.732.732l-.258.774a.145.145 0 0 1-.274 0l-.258-.774a1.45 1.45 0 0 0-.732-.732L9.83 2.137a.145.145 0 0 1 0-.274l.774-.258c.346-.115.617-.386.732-.732z"/>
        </svg>
    </button>
    <button type="button" class="shortcut-btn" title="Sharpen & Detail: make it sharper, high texture detail" data-directive="make it sharper, high texture detail" onclick="togglePromptDirective(this)">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M8.5.5a.5.5 0 0 0-1 0v.518A7 7 0 0 0 1.018 7.5H.5a.5.5 0 0 0 0 1h.518A7 7 0 0 0 7.5 14.982v.518a.5.5 0 0 0 1 0v-.518A7 7 0 0 0 14.982 8.5h.518a.5.5 0 0 0 0-1h-.518A7 7 0 0 0 8.5 1.018zm-6.48 7A6 6 0 0 1 7.5 2.02v4.48h-5.48zm0 1h5.48v4.48A6 6 0 0 1 2.02 8.5zm6.48 4.48V8.5h4.48a6 6 0 0 1-4.48 4.48zm4.48-5.48H8.5V2.02a6 6 0 0 1 4.48 5.48z"/>
        </svg>
    </button>
    <button type="button" class="shortcut-btn" title="Hollywood Movie Color: make the color as in a modern hollywood movie, cinematic color grade" data-directive="make the color as in a modern hollywood movie, cinematic color grade" onclick="togglePromptDirective(this)">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M0 1a1 1 0 0 1 1-1h14a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H1a1 1 0 0 1-1-1zm4 0v6h8V1zm8 8H4v6h8zM1 1v2h2V1zm2 3H1v2h2zm-2 3v2h2V7zm2 3H1v2h2zm-2 3v2h2v-2zM15 1h-2v2h2zm-2 3v2h2V4zm2 3h-2v2h2zm-2 3v2h2v-2zm2 3h-2v2h2z"/>
        </svg>
    </button>
    <button type="button" class="shortcut-btn" title="Outpaint / Fill Space: fill the black space around the image with seamless environment" data-directive="fill the black space around the image with seamless environment" onclick="togglePromptDirective(this)">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path fill-rule="evenodd" d="M5.828 10.172a.5.5 0 0 0-.707 0l-4.096 4.096V11.5a.5.5 0 0 0-1 0v3.975a.5.5 0 0 0 .5.5H4.5a.5.5 0 0 0 0-1H1.732l4.096-4.096a.5.5 0 0 0 0-.707m4.344-4.344a.5.5 0 0 0 .707 0l4.096-4.096V4.5a.5.5 0 1 0 1 0V.525a.5.5 0 0 0-.5-.5H11.5a.5.5 0 0 0 0 1h2.768l-4.096 4.096a.5.5 0 0 0 0 .707"/>
        </svg>
    </button>
    <button type="button" class="shortcut-btn" title="Remove Watermarks & Text: remove all overlay text and watermarks" data-directive="remove all overlay text and watermarks" onclick="togglePromptDirective(this)">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M8.086 2.207a2 2 0 0 1 2.828 0l3.879 3.879a2 2 0 0 1 0 2.828l-5.5 5.5A2 2 0 0 1 7.879 15H5.12a2 2 0 0 1-1.414-.586l-2.5-2.5a2 2 0 0 1 0-2.828zm2.121.707a1 1 0 0 0-1.414 0L4.16 7.547l5.293 5.293 4.633-4.633a1 1 0 0 0 0-1.414zM8.746 13.547 3.453 8.254 1.914 9.793a1 1 0 0 0 0 1.414l2.5 2.5a1 1 0 0 0 .707.293H7.88a1 1 0 0 0 .707-.293z"/>
        </svg>
    </button>
    <button type="button" class="shortcut-btn" title="Clean Video UI: remove all video player UI, playback controls, timelines, overlays and on-screen instruments" data-directive="remove all video player UI, playback controls, timelines, overlays and on-screen instruments" onclick="togglePromptDirective(this)">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M2.5 5a.5.5 0 1 0 0-1 .5.5 0 0 0 0 1m2 0a.5.5 0 1 0 0-1 .5.5 0 0 0 0 1m2 0a.5.5 0 1 0 0-1 .5.5 0 0 0 0 1"/>
            <path d="M0 4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2zm1 10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V6H1zM14 5V4a1 1 0 0 0-1-1H2a1 1 0 0 0-1 1v1z"/>
            <path d="M10.854 8.146a.5.5 0 0 1 0 .708L9.707 10l1.147 1.146a.5.5 0 0 1-.708.708L9 10.707l-1.146 1.147a.5.5 0 0 1-.708-.708L8.293 10 7.146 8.854a.5.5 0 1 1 .708-.708L9 9.293l1.146-1.147a.5.5 0 0 1 .708 0"/>
        </svg>
    </button>
    <button type="button" class="shortcut-btn" title="Colourize: colourize the image with natural realistic vibrant colors" data-directive="colourize the image with natural realistic vibrant colors" onclick="togglePromptDirective(this)">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M8 5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3m4 3a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3M5.5 7a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m.5 6a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3"/>
            <path d="M16 8c0 3.15-1.866 2.585-3.58 2.585-.506 0-.877.169-1.077.348-.204.183-.343.46-.343.867 0 1.554-1.298 2.2-2.5 2.2a4.49 4.49 0 0 1-3.182-1.318C3.89 11.155 3 9.684 3 8a5 5 0 0 1 5-5c2.76 0 5 2.24 5 5 0 .285-.027.562-.078.83A6.2 6.2 0 0 0 16 8m-1.5 0c0-.18-.014-.356-.041-.527A4.004 4.004 0 0 0 8 4a4 4 0 0 0-4 4c0 1.346.72 2.53 1.818 3.182A3.5 3.5 0 0 0 8.5 12.2c.484 0 .9-.22.9-.6a1.54 1.54 0 0 1 .475-1.12c.382-.345.96-.58 1.625-.58 2.015 0 3-.702 3-1.9"/>
        </svg>
    </button>
    <button type="button" class="shortcut-btn" title="Depth of Field / Bokeh: shallow depth of field, soft blurry background, cinematic bokeh" data-directive="shallow depth of field, soft blurry background, cinematic bokeh" onclick="togglePromptDirective(this)">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M15 12a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h1.172a3 3 0 0 0 2.12-.879l.83-.828A1 1 0 0 1 6.827 3h2.344a1 1 0 0 1 .707.293l.828.828A3 3 0 0 0 12.828 5H14a1 1 0 0 1 1 1zM2 4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-1.172a2 2 0 0 1-1.414-.586l-.828-.828A2 2 0 0 0 9.172 2H6.828a2 2 0 0 0-1.414.586l-.828.828A2 2 0 0 1 3.172 4z"/>
            <path d="M8 11a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5m0 1a3.5 3.5 0 1 1 0-7 3.5 3.5 0 0 1 0 7M3 6.5a.5.5 0 1 1-1 0 .5.5 0 0 1 1 0"/>
        </svg>
    </button>
    <button type="button" class="shortcut-btn" title="Face Refinement: enhance facial details, highly detailed realistic eyes and skin" data-directive="enhance facial details, highly detailed realistic eyes and skin" onclick="togglePromptDirective(this)">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M1.5 1a.5.5 0 0 0-.5.5v3a.5.5 0 0 1-1 0v-3A1.5 1.5 0 0 1 1.5 0h3a.5.5 0 0 1 0 1zM11 .5a.5.5 0 0 1 .5-.5h3A1.5 1.5 0 0 1 16 1.5v3a.5.5 0 0 1-1 0v-3a.5.5 0 0 0-.5-.5h-3a.5.5 0 0 1-.5-.5M.5 11a.5.5 0 0 1 .5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 1 0 1h-3A1.5 1.5 0 0 1 0 14.5v-3a.5.5 0 0 1 .5-.5m15 0a.5.5 0 0 1 .5.5v3a1.5 1.5 0 0 1-1.5 1.5h-3a.5.5 0 0 1 0-1h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 1 .5-.5"/>
            <path d="M3 14s-1 0-1-1 1-4 6-4 6 3 6 4-1 1-1 1zm8-9a3 3 0 1 1-6 0 3 3 0 0 1 6 0"/>
        </svg>
    </button>
    <button type="button" class="shortcut-btn" title="Face Swap: Tausche Gesicht aus Referenzbild in &lt;image_1&gt;" data-directive="swap the face of the person in &lt;image_1&gt; with the face from &lt;image_2&gt;, preserve all original lighting, head pose and background of &lt;image_1&gt;" onclick="togglePromptDirective(this)">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path fill-rule="evenodd" d="M1 11.5a.5.5 0 0 0 .5.5h11.793l-3.147 3.146a.5.5 0 0 0 .708.708l4-4a.5.5 0 0 0 0-.708l-4-4a.5.5 0 0 0-.708.708L13.293 11H1.5a.5.5 0 0 0-.5.5m14-7a.5.5 0 0 0-.5-.5H2.707l3.147-3.146a.5.5 0 1 0-.708-.708l-4 4a.5.5 0 0 0 0 .708l4 4a.5.5 0 0 0 .708-.708L2.707 5H14.5a.5.5 0 0 0 .5-.5"/>
        </svg>
    </button>
    <button type="button" class="shortcut-btn" title="Person Swap: Ersetze Person komplett durch Referenz (Körper, Kleidung, Haare, Gesicht)" data-directive="replace the person in &lt;image_1&gt; with the person from &lt;image_2&gt;, including complete body, outfit, hairstyle and appearance while maintaining the scene context, pose and perspective of &lt;image_1&gt;" onclick="togglePromptDirective(this)">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M7 14s-1 0-1-1 1-4 5-4 5 3 5 4-1 1-1 1zm4-6a3 3 0 1 0 0-6 3 3 0 0 0 0 6m-5.784 6A2.24 2.24 0 0 1 5 13c0-1.355.68-2.75 1.936-3.72A6.3 6.3 0 0 0 5 9c-4 0-5 3-5 4s1 1 1 1zM4.5 8a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5"/>
        </svg>
    </button>
    <button type="button" class="shortcut-btn" title="Character Sheet: Model-Sheet (Frontal, 3/4, Profil, Cowboy-Shot) auf neutralem Hintergrund" data-directive="turn the character from reference into a cinematic multi-angle character model sheet, featuring frontal close-up portrait, three-quarter view, profile shot, and a medium cowboy shot, neutral solid studio backdrop, consistent facial features and clothing, high fidelity concept art" onclick="togglePromptDirective(this)">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M6 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6m-5 6s-1 0-1-1 1-4 6-4 6 3 6 4-1 1-1 1zM11 3.5a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 0 1h-4a.5.5 0 0 1-.5-.5m.5 2.5a.5.5 0 0 0 0 1h4a.5.5 0 0 0 0-1zm2 3a.5.5 0 0 0 0 1h2a.5.5 0 0 0 0-1zm0 3a.5.5 0 0 0 0 1h2a.5.5 0 0 0 0-1z"/>
        </svg>
    </button>
</div>
"""

with gr.Blocks(
    title="FIBO Studio - Universal Multi-Platform Engine",
    theme=app_theme,
    css=custom_css,
    head=HEAD_SCRIPTS,
    js="() => { document.body.classList.add('dark'); }",
    fill_width=True,
) as demo:
    hw = detect_hardware()
    gr.HTML(f"<style>{custom_css}</style>", visible=False)
    with gr.Row():
        with gr.Column(scale=8):
            gr.Markdown(
                """
                # 🎨 FIBO Studio
                **Universal High-Performance Interface for BRIA FIBO 2.0**
                *Auto-Hardware Acceleration • SageAttention 2 / MLX Metal • WanGP-style Workflows*
                """
            )
        with gr.Column(scale=4, min_width=250):
            gr.HTML(hw.get_status_badge())

    # ==========================================
    # MODELL-STATUS & SPEICHER-MANAGER
    # ==========================================
    with gr.Accordion("📦 Modell-Status & Download-Manager (Lokale Festplatte)", open=True):
        models_overview_md = gr.Markdown(value=get_models_overview_markdown)
        with gr.Row():
            dl_model_choice = gr.Dropdown(
                choices=["FIBO 2 Turbo (Edit / 4 Steps)", "FIBO 2 Base/Dev (30 Steps)"],
                value="FIBO 2 Turbo (Edit / 4 Steps)",
                label="Modell für Vorab-Download auswählen",
                scale=6,
            )
            btn_start_dl = gr.Button("⬇️ Modell jetzt herunterladen", variant="primary", scale=3)
            btn_refresh_status = gr.Button("🔄 Status aktualisieren", variant="secondary", scale=2)

        dl_status_result = gr.Markdown(value="*Status bereit.*")

        btn_refresh_status.click(
            fn=get_models_overview_markdown,
            inputs=[],
            outputs=[models_overview_md],
        )

        btn_start_dl.click(
            fn=download_model_clean,
            inputs=[dl_model_choice],
            outputs=[dl_status_result, models_overview_md],
        )

    with gr.Accordion("🔑 Hugging Face Authentifizierung", open=False):
        hf_status_md = gr.Markdown(value=get_hf_status)
        with gr.Row():
            hf_token_input = gr.Textbox(
                label="Hugging Face Access Token (Read)",
                placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxx",
                type="password",
                scale=8,
            )
            hf_login_btn = gr.Button("💾 Token speichern & Login", variant="primary", scale=2)

        hf_login_btn.click(
            fn=login_hf_token,
            inputs=[hf_token_input],
            outputs=[hf_status_md],
        )

    with gr.Tabs() as tabs:
        # ==========================================
        # TAB 1: IMAGE EDIT (FIBO 2 Turbo)
        # ==========================================
        with gr.Tab("✏️ Image Edit (FIBO 2 Turbo)", id="tab_edit"):
            with gr.Row():
                # LEFT COLUMN: INPUTS
                with gr.Column(scale=5):
                    ref1 = gr.Image(
                        label="Reference Image 1 (Base / Primary)",
                        type="pil",
                        sources=["upload", "clipboard"],
                    )

                    with gr.Accordion("📦 Stapelverarbeitung / Bulk Processing (Mehrere Bilder als <image_1> abarbeiten)", open=False):
                        gr.Markdown(
                            "💡 **Stapelverarbeitung (Batch Mode):** Gib einen lokalen Bilder-Ordnerpfad an ODER lade mehrere Dateien per Drag & Drop hoch. "
                            "Jedes Bild wird sequentiell als Basisbild (`<image_1>`) eingesetzt und mit denselben Einstellungen, zusätzlichen Referenzbildern (`<image_2>` – `<image_5>`) "
                            "und dem Prompt verarbeitet. Perfekt für Filmstills, Sequenzen oder Watermark-Entfernung. "
                            "*(Nicht-Bild-Dateien wie HTML oder Text werden automatisch erkannt und gefiltert.)*"
                        )
                        with gr.Row():
                            bulk_folder = gr.Textbox(
                                label="Lokaler Ordnerpfad mit Bildern",
                                placeholder="/Users/.../stills",
                                scale=8,
                            )
                            btn_browse_folder = gr.Button("📁 Ordner im Finder wählen...", variant="secondary", scale=4)
                        with gr.Row():
                            bulk_recursive = gr.Checkbox(label="Unterordner ebenfalls durchsuchen", value=False, scale=6)
                            btn_clear_bulk = gr.Button("🗑️ Auswahl leeren", variant="secondary", scale=3)
                        bulk_files = gr.File(
                            label="Oder Dateien per Drag & Drop hochladen (Mehrfachauswahl)",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        with gr.Row():
                            btn_scan_bulk = gr.Button("🔍 Bilder scannen & prüfen", variant="secondary", scale=4)
                            with gr.Column(scale=8):
                                bulk_scan_status = gr.Markdown(value="*Keine Dateien ausgewählt.*")

                        bulk_active_paths = gr.State(value=[])
                        with gr.Row(elem_id="hidden_bulk_bridge_row"):
                            hidden_bulk_remove_path = gr.Textbox(elem_id="hidden_bulk_remove_path")
                            btn_trigger_bulk_remove = gr.Button("Remove", elem_id="btn_trigger_bulk_remove")

                        bulk_preview_html = gr.HTML(
                            value="",
                            visible=False,
                            elem_id="bulk_preview_html_container",
                        )
                        with gr.Row():
                            btn_run_bulk = gr.Button(
                                "🚀 Stapelverarbeitung starten",
                                variant="primary",
                                scale=8,
                                elem_classes=["generate-btn"],
                            )
                            btn_stop_bulk = gr.Button("🛑 Stop", variant="stop", scale=2, elem_classes=["generate-btn"])

                    with gr.Accordion("Zusätzliche Referenzbilder (Slots 2 – 5)", open=False):
                        gr.Markdown(
                            "💡 *Referenzbilder können im Prompt als `<image_1>`, `<image_2>`, `<image_3>`, `<image_4>`, `<image_5>` angesprochen werden.*"
                        )
                        with gr.Row():
                            ref2 = gr.Image(label="Image 2 (<image_2>)", type="pil", sources=["upload", "clipboard"])
                            ref3 = gr.Image(label="Image 3 (<image_3>)", type="pil", sources=["upload", "clipboard"])
                        with gr.Row():
                            ref4 = gr.Image(label="Image 4 (<image_4>)", type="pil", sources=["upload", "clipboard"])
                            ref5 = gr.Image(label="Image 5 (<image_5>)", type="pil", sources=["upload", "clipboard"])

                    with gr.Accordion("Inpainting-Maske (Optional)", open=False):
                        edit_mask = gr.Image(
                            label="Mask Image (Weiß = Bearbeiten, Schwarz = Beibehalten)",
                            type="pil",
                            sources=["upload", "clipboard"],
                        )

                    gr.HTML(EDIT_SHORTCUTS_TOOLBAR_HTML)

                    initial_modifiers = load_user_modifiers()
                    initial_mod_names = list(initial_modifiers.keys())
                    with gr.Row(elem_classes=["custom-modifier-bar"]):
                        custom_mod_dropdown = gr.Dropdown(
                            choices=initial_mod_names,
                            value=initial_mod_names[0] if initial_mod_names else None,
                            label="Gespeicherte Modifier (Custom Presets)",
                            scale=6,
                            interactive=True,
                        )
                        btn_apply_mod = gr.Button("📋 Anwenden", variant="primary", scale=2)
                        btn_toggle_add_mod = gr.Button("➕ Neu", variant="secondary", scale=1)
                        btn_del_mod = gr.Button("🗑️", variant="stop", scale=1, min_width=45)

                    with gr.Row(visible=False, elem_classes=["custom-mod-add-box"]) as add_mod_box:
                        with gr.Column():
                            with gr.Row():
                                new_mod_name = gr.Textbox(
                                    label="Name des neuen Modifiers",
                                    placeholder="z.B. 1980s Retro Synthwave",
                                    scale=6,
                                )
                                btn_use_cur_prompt = gr.Button("⬇️ Aus aktuellem Edit Prompt", variant="secondary", scale=4)
                            new_mod_text = gr.Textbox(
                                label="Prompt / Text des Modifiers",
                                placeholder="Text, der an den Prompt angehängt werden soll...",
                                lines=2,
                            )
                            with gr.Row():
                                btn_save_mod = gr.Button("💾 Speichern", variant="primary", scale=5)
                                btn_cancel_add_mod = gr.Button("Abbrechen", variant="secondary", scale=5)

                    edit_prompt = gr.Textbox(
                        label="Edit Prompt",
                        placeholder="z.B. Make him wear sunglasses and a leather jacket in the style of <image_2>...",
                        lines=3,
                        elem_id="edit_prompt",
                    )

                    btn_apply_mod.click(
                        fn=apply_user_modifier,
                        inputs=[edit_prompt, custom_mod_dropdown],
                        outputs=[edit_prompt],
                    )
                    btn_toggle_add_mod.click(
                        fn=lambda: gr.update(visible=True),
                        inputs=[],
                        outputs=[add_mod_box],
                    )
                    btn_cancel_add_mod.click(
                        fn=lambda: gr.update(visible=False),
                        inputs=[],
                        outputs=[add_mod_box],
                    )
                    btn_use_cur_prompt.click(
                        fn=lambda p: p,
                        inputs=[edit_prompt],
                        outputs=[new_mod_text],
                    )
                    btn_save_mod.click(
                        fn=save_user_modifier,
                        inputs=[new_mod_name, new_mod_text],
                        outputs=[custom_mod_dropdown, new_mod_name, new_mod_text, add_mod_box],
                    )
                    btn_del_mod.click(
                        fn=delete_user_modifier,
                        inputs=[custom_mod_dropdown],
                        outputs=[custom_mod_dropdown],
                    )

                    with gr.Accordion("📥 Prompt aus beliebigem Bild auslesen & einfügen", open=False):
                        with gr.Row():
                            edit_meta_img = gr.Image(
                                label="Bild hier ablegen / einfügen (Drag & Drop)",
                                type="filepath",
                                sources=["upload", "clipboard"],
                                scale=7,
                            )
                            btn_extract_edit_prompt = gr.Button("📋 Prompt übernehmen", variant="secondary", scale=3)
                        btn_extract_edit_prompt.click(
                            fn=load_prompt_from_image,
                            inputs=[edit_meta_img],
                            outputs=[edit_prompt],
                        )
                        edit_meta_img.change(
                            fn=load_prompt_from_image,
                            inputs=[edit_meta_img],
                            outputs=[edit_prompt],
                        )

                    edit_aspect = gr.Dropdown(
                        label="Seitenverhältnis & Auflösung",
                        choices=list(ASPECT_RATIOS.keys()),
                        value="Original Aspect Ratio",
                    )

                    with gr.Row(visible=False) as edit_custom_box:
                        edit_cw = gr.Slider(minimum=256, maximum=2048, step=16, value=1024, label="Custom Width")
                        edit_ch = gr.Slider(minimum=256, maximum=2048, step=16, value=1024, label="Custom Height")

                    edit_aspect.change(toggle_custom_dims, inputs=[edit_aspect], outputs=[edit_custom_box])

                    with gr.Accordion("Erweiterte Einstellungen", open=False):
                        with gr.Row():
                            edit_quant = gr.Radio(
                                choices=[(8, 8), (4, 4)],
                                value=8,
                                label="Quantisierung (MLX)",
                                info="8-Bit: Beste Qualität & Speed | 4-Bit: Sehr sparsam",
                            )
                            edit_steps = gr.Slider(minimum=1, maximum=20, step=1, value=4, label="Inferenz-Steps")
                            edit_guidance = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=1.0, label="Guidance (CFG)")

                        with gr.Row():
                            edit_seed = gr.Number(value=42, label="Seed", precision=0)
                            edit_random_seed = gr.Checkbox(value=True, label="Zufälliger Seed")

                    with gr.Accordion("LoRA Modell (Optional)", open=False):
                        edit_lora_path = gr.Textbox(label="LoRA Dateipfad (.safetensors)", placeholder="/pfad/zu/lora.safetensors")
                        edit_lora_scale = gr.Slider(minimum=-2.0, maximum=2.0, step=0.05, value=1.0, label="LoRA Skalierung")

                    with gr.Row():
                        btn_run_edit = gr.Button("✨ Run Edit (4 Steps Turbo)", variant="primary", scale=8, elem_classes=["generate-btn"])
                        btn_stop_edit = gr.Button("🛑 Stop", variant="stop", scale=2, elem_classes=["generate-btn"])

                # RIGHT COLUMN: OUTPUTS
                with gr.Column(scale=5):
                    edit_output = gr.Image(label="Ergebnis", type="filepath", format="png")
                    edit_info = gr.Markdown(value="*Bereit zur Generierung.*", elem_classes=["info-box"])
                    with gr.Row():
                        btn_send_to_ref1 = gr.Button("🔄 Ergebnis als Referenzbild 1 übernehmen", variant="secondary")

                    with gr.Accordion("📦 Galerie der Stapelverarbeitung (Bulk Results)", open=True):
                        bulk_gallery = gr.Gallery(
                            label="Ergebnisse der Stapelverarbeitung",
                            show_label=False,
                            columns=3,
                            height="auto",
                            object_fit="contain",
                            elem_id="bulk_gallery",
                            preview=True,
                        )
                        bulk_summary_md = gr.Markdown(value="*Keine Stapelverarbeitung aktiv.*", elem_classes=["info-box"])
                        with gr.Row():
                            btn_bulk_open_finder = gr.Button("📂 Ausgabeordner im Mac-Finder öffnen", variant="secondary")

                    with gr.Accordion("✨ Post-Processing & Upscaling (Lanczos, SeedVR2, Real-ESRGAN, Face Refiner)", open=False):
                        with gr.Row():
                            edit_post_method = gr.Dropdown(
                                label="Upscale-Methode",
                                choices=[
                                    "Lanczos Resizer (Instant 0.05s)",
                                    "SeedVR2 (Natives MLX Diffusion)",
                                    "Real-ESRGAN (AI Texture)",
                                ],
                                value="Lanczos Resizer (Instant 0.05s)",
                            )
                            edit_post_scale = gr.Dropdown(
                                label="Faktor",
                                choices=[("1.5x", 1.5), ("2.0x", 2.0), ("4.0x", 4.0)],
                                value=2.0,
                            )
                        with gr.Row():
                            edit_post_sharpness = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Lanczos Detail-Schärfe")
                            edit_post_face = gr.Checkbox(label="👤 Face Refiner (Gesichtsdetails schärfen)", value=False)
                        with gr.Row(visible=False) as edit_seedvr2_box:
                            edit_seedvr2_res = gr.Dropdown(label="Zielauflösung (SeedVR2)", choices=[1536, 2048, 2560, 3840], value=2048)
                            edit_seedvr2_soft = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.0, label="Softness")

                        btn_run_edit_post = gr.Button("🚀 Bild jetzt hochskalieren & verfeinern", variant="secondary")

                    edit_post_method.change(
                        lambda m: gr.update(visible="SeedVR2" in m),
                        inputs=[edit_post_method],
                        outputs=[edit_seedvr2_box],
                    )

                    btn_run_edit_post.click(
                        fn=handle_postprocessing_run,
                        inputs=[
                            edit_output,
                            edit_post_method,
                            edit_post_scale,
                            edit_post_sharpness,
                            edit_seedvr2_res,
                            edit_seedvr2_soft,
                            edit_post_face,
                        ],
                        outputs=[edit_output, edit_info],
                    )

            btn_send_to_ref1.click(lambda img: img, inputs=[edit_output], outputs=[ref1])

            edit_event = btn_run_edit.click(
                fn=handle_edit,
                inputs=[
                    ref1,
                    ref2,
                    ref3,
                    ref4,
                    ref5,
                    edit_mask,
                    edit_prompt,
                    edit_aspect,
                    edit_cw,
                    edit_ch,
                    edit_quant,
                    edit_steps,
                    edit_guidance,
                    edit_seed,
                    edit_random_seed,
                    edit_lora_path,
                    edit_lora_scale,
                ],
                outputs=[edit_output, edit_info, edit_seed],
            )
            btn_stop_edit.click(
                fn=GLOBAL_ENGINE.abort,
                cancels=[edit_event],
                queue=False,
            )

            # Bulk Processing Events
            def on_pick_folder(cur_folder, files, rec):
                chosen = select_folder_via_finder(cur_folder)
                target = chosen if chosen else cur_folder
                target = sanitize_folder_path(target)
                paths, grid_html, status_md = preview_bulk_inputs(target, files, rec)
                return target, paths, gr.update(value=grid_html, visible=bool(paths)), status_md

            def on_scan_btn_click(folder, files, rec):
                clean_f = sanitize_folder_path(folder) if folder else ""
                paths, grid_html, status_md = preview_bulk_inputs(clean_f, files, rec)
                return clean_f, paths, gr.update(value=grid_html, visible=bool(paths)), status_md

            def on_bulk_inputs_change(folder, files, rec):
                clean_f = sanitize_folder_path(folder) if folder else ""
                paths, grid_html, status_md = preview_bulk_inputs(clean_f, files, rec)
                return paths, gr.update(value=grid_html, visible=bool(paths)), status_md

            def on_remove_bulk_item(target_path_str: str, current_paths: List[str]):
                if current_paths is None:
                    current_paths = []
                clean_target = str(Path(target_path_str.strip()).resolve()) if target_path_str else ""
                paths_copy = [p for p in current_paths if str(Path(p).resolve()) != clean_target and p != target_path_str]
                new_html = render_bulk_thumbnails_html(paths_copy)
                count = len(paths_copy)
                status_md = f"✅ **{count} gültige Bilder bereit** (manuell angepasst)." if count > 0 else "*Keine Bilder mehr in der Auswahl.*"
                return paths_copy, gr.update(value=new_html, visible=bool(paths_copy)), status_md

            def on_clear_bulk():
                return "", None, [], gr.update(value="", visible=False), "*Keine Dateien ausgewählt.*"

            btn_browse_folder.click(
                fn=on_pick_folder,
                inputs=[bulk_folder, bulk_files, bulk_recursive],
                outputs=[bulk_folder, bulk_active_paths, bulk_preview_html, bulk_scan_status],
            )

            btn_scan_bulk.click(
                fn=on_scan_btn_click,
                inputs=[bulk_folder, bulk_files, bulk_recursive],
                outputs=[bulk_folder, bulk_active_paths, bulk_preview_html, bulk_scan_status],
            )

            btn_clear_bulk.click(
                fn=on_clear_bulk,
                inputs=[],
                outputs=[bulk_folder, bulk_files, bulk_active_paths, bulk_preview_html, bulk_scan_status],
            )

            btn_trigger_bulk_remove.click(
                fn=on_remove_bulk_item,
                inputs=[hidden_bulk_remove_path, bulk_active_paths],
                outputs=[bulk_active_paths, bulk_preview_html, bulk_scan_status],
            )

            bulk_folder.change(
                fn=on_bulk_inputs_change,
                inputs=[bulk_folder, bulk_files, bulk_recursive],
                outputs=[bulk_active_paths, bulk_preview_html, bulk_scan_status],
            )

            bulk_files.change(
                fn=on_bulk_inputs_change,
                inputs=[bulk_folder, bulk_files, bulk_recursive],
                outputs=[bulk_active_paths, bulk_preview_html, bulk_scan_status],
            )

            bulk_event = btn_run_bulk.click(
                fn=handle_bulk_edit,
                inputs=[
                    bulk_active_paths,
                    bulk_folder,
                    bulk_files,
                    bulk_recursive,
                    ref2,
                    ref3,
                    ref4,
                    ref5,
                    edit_mask,
                    edit_prompt,
                    edit_aspect,
                    edit_cw,
                    edit_ch,
                    edit_quant,
                    edit_steps,
                    edit_guidance,
                    edit_seed,
                    edit_random_seed,
                    edit_lora_path,
                    edit_lora_scale,
                ],
                outputs=[bulk_gallery, bulk_summary_md],
            )
            btn_stop_bulk.click(
                fn=GLOBAL_ENGINE.abort,
                cancels=[bulk_event],
                queue=False,
            )

            btn_bulk_open_finder.click(
                fn=open_outputs_in_finder,
                inputs=[],
                outputs=[bulk_summary_md],
            )

        # ==========================================
        # TAB 2: TEXT TO IMAGE (FIBO 2)
        # ==========================================
        with gr.Tab("🖼️ Text to Image (FIBO 2)", id="tab_t2i"):
            with gr.Row():
                # LEFT COLUMN: INPUTS
                with gr.Column(scale=5):
                    t2i_model = gr.Dropdown(
                        label="Modell-Engine",
                        choices=[
                            "FIBO 2 Turbo (4 Steps, Ultra-Fast ~30s)",
                            "FIBO 2 Base/Dev (30 Steps, Präzisions-Modus)",
                        ],
                        value="FIBO 2 Turbo (4 Steps, Ultra-Fast ~30s)",
                        info="Turbo nutzt 4-Step SFT Gewichte für schnelle Generierungen.",
                    )

                    t2i_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="z.B. A futuristic glass greenhouse situated on a Martian crater, cinematic 8k...",
                        lines=4,
                    )

                    with gr.Accordion("📥 Prompt & Parameter aus Bild auslesen & einfügen", open=False):
                        with gr.Row():
                            t2i_meta_img = gr.Image(
                                label="Bild hier ablegen / einfügen (Drag & Drop)",
                                type="filepath",
                                sources=["upload", "clipboard"],
                                scale=6,
                            )
                            with gr.Column(scale=4):
                                btn_extract_t2i_prompt = gr.Button("📋 Nur Prompt übernehmen", variant="secondary")
                                btn_extract_t2i_all = gr.Button("⚡ Alle Parameter übernehmen", variant="primary")
                        btn_extract_t2i_prompt.click(
                            fn=load_prompt_from_image,
                            inputs=[t2i_meta_img],
                            outputs=[t2i_prompt],
                        )
                        t2i_meta_img.change(
                            fn=load_prompt_from_image,
                            inputs=[t2i_meta_img],
                            outputs=[t2i_prompt],
                        )
                    t2i_expand = gr.Checkbox(
                        label="✨ Prompt mit FIBO-VLM erweitern",
                        value=False,
                        info="Generiert automatisch detailliertere Beschreibungen vor der Diffusions-Berechnung.",
                    )

                    t2i_aspect = gr.Dropdown(
                        label="Seitenverhältnis & Auflösung",
                        choices=[k for k in ASPECT_RATIOS.keys() if k != "Original Aspect Ratio"],
                        value="1:1 Square (1024 x 1024)",
                    )

                    with gr.Row(visible=False) as t2i_custom_box:
                        t2i_cw = gr.Slider(minimum=256, maximum=2048, step=16, value=1024, label="Custom Width")
                        t2i_ch = gr.Slider(minimum=256, maximum=2048, step=16, value=1024, label="Custom Height")

                    t2i_aspect.change(toggle_custom_dims, inputs=[t2i_aspect], outputs=[t2i_custom_box])

                    with gr.Accordion("Erweiterte Einstellungen", open=False):
                        with gr.Row():
                            t2i_quant = gr.Radio(
                                choices=[(8, 8), (4, 4)],
                                value=8,
                                label="Quantisierung (MLX)",
                                info="8-Bit: Beste Qualität | 4-Bit: Sehr sparsam",
                            )
                            t2i_steps = gr.Slider(minimum=1, maximum=60, step=1, value=4, label="Inferenz-Steps")
                            t2i_guidance = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=1.0, label="Guidance (CFG)")

                        with gr.Row():
                            t2i_seed = gr.Number(value=42, label="Seed", precision=0)
                            t2i_random_seed = gr.Checkbox(value=True, label="Zufälliger Seed")

                    btn_extract_t2i_all.click(
                        fn=load_all_params_from_image,
                        inputs=[t2i_meta_img, t2i_steps, t2i_guidance, t2i_seed],
                        outputs=[t2i_prompt, t2i_steps, t2i_guidance, t2i_seed],
                    )

                    t2i_model.change(
                        update_t2i_defaults,
                        inputs=[t2i_model],
                        outputs=[t2i_steps, t2i_guidance],
                    )

                    with gr.Accordion("LoRA Modell (Optional)", open=False):
                        t2i_lora_path = gr.Textbox(label="LoRA Dateipfad (.safetensors)", placeholder="/pfad/zu/lora.safetensors")
                        t2i_lora_scale = gr.Slider(minimum=-2.0, maximum=2.0, step=0.05, value=1.0, label="LoRA Skalierung")

                    with gr.Row():
                        btn_run_t2i = gr.Button("🎨 Bild generieren", variant="primary", scale=8, elem_classes=["generate-btn"])
                        btn_stop_t2i = gr.Button("🛑 Stop", variant="stop", scale=2, elem_classes=["generate-btn"])

                # RIGHT COLUMN: OUTPUTS
                with gr.Column(scale=5):
                    t2i_output = gr.Image(label="Generiertes Bild", type="filepath", format="png")
                    t2i_info = gr.Markdown(value="*Bereit zur Generierung.*", elem_classes=["info-box"])
                    with gr.Row():
                        btn_send_t2i_to_edit = gr.Button("➡️ Bild in Edit-Tab übertragen", variant="secondary")

                    with gr.Accordion("✨ Post-Processing & Upscaling (Lanczos, SeedVR2, Real-ESRGAN, Face Refiner)", open=False):
                        with gr.Row():
                            t2i_post_method = gr.Dropdown(
                                label="Upscale-Methode",
                                choices=[
                                    "Lanczos Resizer (Instant 0.05s)",
                                    "SeedVR2 (Natives MLX Diffusion)",
                                    "Real-ESRGAN (AI Texture)",
                                ],
                                value="Lanczos Resizer (Instant 0.05s)",
                            )
                            t2i_post_scale = gr.Dropdown(
                                label="Faktor",
                                choices=[("1.5x", 1.5), ("2.0x", 2.0), ("4.0x", 4.0)],
                                value=2.0,
                            )
                        with gr.Row():
                            t2i_post_sharpness = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Lanczos Detail-Schärfe")
                            t2i_post_face = gr.Checkbox(label="👤 Face Refiner (Gesichtsdetails schärfen)", value=False)
                        with gr.Row(visible=False) as t2i_seedvr2_box:
                            t2i_seedvr2_res = gr.Dropdown(label="Zielauflösung (SeedVR2)", choices=[1536, 2048, 2560, 3840], value=2048)
                            t2i_seedvr2_soft = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.0, label="Softness")

                        btn_run_t2i_post = gr.Button("🚀 Bild jetzt hochskalieren & verfeinern", variant="secondary")

                    t2i_post_method.change(
                        lambda m: gr.update(visible="SeedVR2" in m),
                        inputs=[t2i_post_method],
                        outputs=[t2i_seedvr2_box],
                    )

                    btn_run_t2i_post.click(
                        fn=handle_postprocessing_run,
                        inputs=[
                            t2i_output,
                            t2i_post_method,
                            t2i_post_scale,
                            t2i_post_sharpness,
                            t2i_seedvr2_res,
                            t2i_seedvr2_soft,
                            t2i_post_face,
                        ],
                        outputs=[t2i_output, t2i_info],
                    )

            btn_send_t2i_to_edit.click(
                fn=lambda img: (img, gr.Tabs(selected="tab_edit")),
                inputs=[t2i_output],
                outputs=[ref1, tabs],
            )

            t2i_event = btn_run_t2i.click(
                fn=handle_txt2img,
                inputs=[
                    t2i_prompt,
                    t2i_model,
                    t2i_expand,
                    t2i_aspect,
                    t2i_cw,
                    t2i_ch,
                    t2i_quant,
                    t2i_steps,
                    t2i_guidance,
                    t2i_seed,
                    t2i_random_seed,
                    t2i_lora_path,
                    t2i_lora_scale,
                ],
                outputs=[t2i_output, t2i_info, t2i_seed],
            )
            btn_stop_t2i.click(
                fn=GLOBAL_ENGINE.abort,
                cancels=[t2i_event],
                queue=False,
            )

        # ==========================================
        # TAB 3: VISION AI, TAGGING & PNG INFO
        # ==========================================
        with gr.Tab("🔎 Vision AI, Tagging & PNG Info", id="tab_pnginfo"):
            gr.Markdown(
                "Lade ein beliebiges Bild hoch, um **eingebettete Diffusions-Metadaten** auszulesen, "
                "den Bildinhalt mit **Vision AI (Apple MLX GPU)** zu beschreiben oder **Tags mit WD-Tagger (Apple MPS)** zu extrahieren."
            )
            with gr.Row():
                with gr.Column(scale=5):
                    inspector_img = gr.Image(
                        label="Bild hier ablegen / einfügen (Drag & Drop oder Clipboard)",
                        type="filepath",
                        sources=["upload", "clipboard"],
                    )
                with gr.Column(scale=5):
                    inspector_details_md = gr.Markdown(value="*Kein Bild geladen.*", elem_classes=["info-box"])
                    inspector_prompt_box = gr.Textbox(label="Ausgelesener Prompt", lines=3)
                    with gr.Row():
                        inspector_seed = gr.Number(label="Seed", precision=0)
                        inspector_steps = gr.Number(label="Steps", precision=0)
                        inspector_cfg = gr.Number(label="Guidance (CFG)")
                    with gr.Row():
                        btn_send_inspector_to_edit = gr.Button("✏️ In 'Image Edit' übernehmen", variant="primary")
                        btn_send_inspector_to_t2i = gr.Button("🖼️ In 'Text to Image' übernehmen", variant="primary")
                        btn_send_inspector_to_ref1 = gr.Button("🔄 Als Referenz 1 in Edit", variant="secondary")

            inspector_img.change(
                fn=handle_png_info,
                inputs=[inspector_img],
                outputs=[
                    inspector_details_md,
                    inspector_prompt_box,
                    inspector_seed,
                    inspector_steps,
                    inspector_cfg,
                ],
            )

            btn_send_inspector_to_edit.click(
                fn=lambda p: (p, gr.Tabs(selected="tab_edit")),
                inputs=[inspector_prompt_box],
                outputs=[edit_prompt, tabs],
            )

            btn_send_inspector_to_t2i.click(
                fn=lambda p, s, st, c: (
                    p,
                    s if s is not None else 42,
                    st if st is not None else 4,
                    c if c is not None else 1.0,
                    gr.Tabs(selected="tab_t2i"),
                ),
                inputs=[inspector_prompt_box, inspector_seed, inspector_steps, inspector_cfg],
                outputs=[t2i_prompt, t2i_seed, t2i_steps, t2i_guidance, tabs],
            )

            btn_send_inspector_to_ref1.click(
                fn=lambda img: (img, gr.Tabs(selected="tab_edit")),
                inputs=[inspector_img],
                outputs=[ref1, tabs],
            )

            # --- VISION AI & TAGGING SECTIONS ---
            with gr.Accordion("👁️ Vision AI Bilderkennung & Captioning (Apple MLX GPU)", open=True):
                with gr.Row():
                    vlm_prompt_mode = gr.Dropdown(
                        label="Analyse-Modus",
                        choices=[
                            "Detaillierte Szenenbeschreibung (Ideal für KI-Prompts)",
                            "Fokus auf Kleidung & Personen (Ideal für Edit-Prompts)",
                            "Kompakter Bildinhalt",
                        ],
                        value="Detaillierte Szenenbeschreibung (Ideal für KI-Prompts)",
                        scale=6,
                    )
                    vlm_custom_q = gr.Textbox(
                        label="Spezifische Frage (Optional)",
                        placeholder="z.B. What style and artistic technique is used?",
                        scale=4,
                    )
                btn_run_vlm = gr.Button("✨ Bildinhalt analysieren (Qwen2-VL 4-Bit MLX)", variant="primary")
                vlm_output_box = gr.Textbox(label="Generierte Bildbeschreibung / Prompt", lines=3)
                vlm_info_txt = gr.Markdown(value="*Bereit.*")
                with gr.Row():
                    btn_send_vlm_to_edit = gr.Button("✏️ Als Edit-Prompt übernehmen", variant="secondary")
                    btn_send_vlm_to_t2i = gr.Button("🖼️ Als Text-to-Image Prompt übernehmen", variant="secondary")

                btn_run_vlm.click(
                    fn=handle_vlm_run,
                    inputs=[inspector_img, vlm_prompt_mode, vlm_custom_q],
                    outputs=[vlm_output_box, vlm_info_txt],
                )
                btn_send_vlm_to_edit.click(
                    fn=lambda p: (p, gr.Tabs(selected="tab_edit")),
                    inputs=[vlm_output_box],
                    outputs=[edit_prompt, tabs],
                )
                btn_send_vlm_to_t2i.click(
                    fn=lambda p: (p, gr.Tabs(selected="tab_t2i")),
                    inputs=[vlm_output_box],
                    outputs=[t2i_prompt, tabs],
                )

            with gr.Accordion("🏷️ WD-Tagger (Deep Danbooru / SwinV2 auf Apple MPS)", open=True):
                with gr.Row():
                    tag_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        step=0.05,
                        value=0.35,
                        label="Konfidenz-Schwellenwert",
                        scale=6,
                    )
                    tag_include_chars = gr.Checkbox(label="Charakter-Tags einbeziehen", value=True, scale=4)
                btn_run_tagger = gr.Button("🏷️ Tags extrahieren (Apple MPS GPU)", variant="primary")
                tag_result_box = gr.Textbox(label="Erkannte Tags (kommagetrennt)", lines=2)
                tag_table_md = gr.Markdown(value="*Noch keine Tags analysiert.*")
                with gr.Row():
                    btn_append_tags_to_edit = gr.Button("✏️ Tags an Edit-Prompt anhängen", variant="secondary")
                    btn_append_tags_to_t2i = gr.Button("🖼️ Tags an Text-to-Image Prompt anhängen", variant="secondary")

                btn_run_tagger.click(
                    fn=handle_tagger_run,
                    inputs=[inspector_img, tag_threshold, tag_include_chars],
                    outputs=[tag_result_box, tag_table_md],
                )
                btn_append_tags_to_edit.click(
                    fn=lambda cur, new: (f"{cur}, {new}".strip(", "), gr.Tabs(selected="tab_edit")),
                    inputs=[edit_prompt, tag_result_box],
                    outputs=[edit_prompt, tabs],
                )
                btn_append_tags_to_t2i.click(
                    fn=lambda cur, new: (f"{cur}, {new}".strip(", "), gr.Tabs(selected="tab_t2i")),
                    inputs=[t2i_prompt, tag_result_box],
                    outputs=[t2i_prompt, tabs],
                )

        # ==========================================
        # TAB 4: BILINGUAL GUIDE / ANLEITUNG
        # ==========================================
        with gr.Tab("📖 Guide / Anleitung", id="tab_guide"):
            with gr.Tabs():
                with gr.Tab("🇩🇪 Deutsch (Handbuch)"):
                    gr.Markdown(
                        """
                        ## 🎨 FIBO Studio – Benutzerhandbuch & Workflow-Guide

                        Willkommen bei **FIBO Studio**, dem nativen Kreativ-Studio für **Apple Silicon Macs** (M1/M2/M3/M4) auf Basis von **Apple MLX** und **Metal Performance Shaders (MPS)**.

                        ---

                        ### 1. ⚡ Architektur & Performance auf Apple Silicon
                        * **Zero-Copy Unified Memory:** FIBO Studio nutzt den gemeinsamen Speicher deines Mac (bis zu 800 GB/s auf Max-Chips). Es gibt keine VRAM-Limits durch PCIe-Kopieren wie bei PCs.
                        * **Nativer MLX-Backend:** Diffusionsmodelle laufen über Apples eigene Metal-Shader mit 8-Bit und 4-Bit Quantisierung (`QuantizedLinear`).
                        * **Externe SSD-Auslagerung:** Das System mountet große HuggingFace-Snapshots (`~40 GB`) automatisch auf deine externe SSD (`/Volumes/.../Hugginface`), sodass deine interne Macintosh HD frei bleibt.

                        ---

                        ### 2. ✏️ Image Edit (FIBO 2 Turbo – 4 Steps)
                        Der Edit-Modus ist darauf spezialisiert, bestehende Bilder extrem schnell und gezielt zu transformieren:
                        * **Referenz-Slots 1 bis 5:** Lade dein Hauptbild in Slot 1. In Slot 2–5 kannst du zusätzliche Referenzbilder ablegen (z.B. ein Kleidungsstück, einen Kunststil oder ein Gesicht).
                        * **Prompt-Syntax:** Beziehe dich im Prompt direkt auf Slots, z.B.:
                          * `Make the person in <image_1> wear the sunglasses from <image_2>.`
                          * `Change the car in <image_1> to a vintage red convertible.`
                        * **Inpainting-Maske (Optional):** Zeichne im Masken-Feld eine Maske, um ausschließlich bestimmte Regionen (z.B. nur die Augen oder den Hintergrund) zu bearbeiten.
                        * **Iterativer Workflow:** Klicke nach jedem Schritt auf **„🔄 Ergebnis als Referenzbild 1 übernehmen“**, um deine Bearbeitungen nahtlos schrittweise aufzubauen.
                        * **📦 Stapelverarbeitung (Bulk Mode):** Wähle einen Ordner oder lade mehrere Bilder per Drag & Drop hoch. Jedes Bild wird der Reihe nach als `<image_1>` verarbeitet – ideal für Filmsequenzen, Banderolen-/Watermark-Entfernung oder Serien-Restaurierung. *(Ungültige Dateien wie HTML oder TXT werden automatisch gefiltert).*
                        * **⚡ Shortcut-Toolbar (Toggles):** Nutze die Icon-Toolbar über dem Prompt für 1-Klick-Modifikatoren (Hollywood-Color-Grading, Video-Player-UI/Instrumente entfernen, Schärfen, Outpainting, Entrauschen).

                        ---

                        ### 3. 🎨 Text to Image (Turbo vs. Base/Dev)
                        * **FIBO 2 Turbo (4 Steps):** Extrem schneller SFT-Generator (ideal für schnelle Ideen, Prototypen und Drafts).
                        * **FIBO 2 Base/Dev (30 Steps):** Volles Diffusionsmodell mit 30 Schritten und CFG 3.5 für maximale Detailauflösung und Bildkohärenz.
                        * **VLM Prompt-Erweiterung:** Bei Aktivierung generiert ein vorgeschaltetes Vision-Language-Modell auf Basis deines kurzen Prompts eine reichhaltige Bildbeschreibung, bevor die Diffusion startet.
                        * **Seitenverhältnisse:** Schnellauswahl für 1:1, 16:9, 9:16, 21:9 oder freie Eingabe (wird automatisch auf 16er-Blöcke gerundet).

                        ---

                        ### 4. ✨ Post-Processing & Upscaling
                        Direkt unter jedem generierten Bild findest du das Post-Processing-Akkordion:
                        * **Lanczos Resizer (Instant 0,05s):** Skaliert das Bild auf 1.5x, 2x oder 4x auf der CPU mit einstellbarer Kantenschärfung (*Unsharp Mask*). Verbraucht 0 MB VRAM!
                        * **SeedVR2 (Natives MLX Diffusion):** 1-Step Diffusions-Upscaler speziell für Apple Silicon für fotorealistische 4K-Super-Resolution.
                        * **Real-ESRGAN:** Ergänzt fotorealistische Texturen, Gewebe- und Haardetails.
                        * **👤 Face Refiner:** Rekonstruiert unscharfe Gesichter, Augen, Zähne und Hautstrukturen.

                        ---

                        ### 5. 👁️ Vision AI & WD-Tagger (Reiter 3)
                        * **Qwen2-VL 4-Bit (MLX):** Beschreibt den Bildinhalt in natürlicher Sprache bei rasenden **~110 Tokens/Sekunde**.
                        * **WD-Tagger (SwinV2 auf Apple MPS):** Extrahiert detaillierte Danbooru- & Stil-Tags in unter **0,4 Sekunden**.
                        * **1-Klick-Übernahme:** Übertrage erkannte Prompts oder Tags direkt in den Edit- oder Text-to-Image-Tab.

                        ---

                        ### 6. 🔍 Metadaten & History-Fallback
                        * **Multi-Standard Embedding:** Jedes Bild wird beim Speichern mit Metadaten nach A1111 (`parameters`), ComfyUI (`prompt`), EXIF (`UserComment`), XMP und IPTC getaggt.
                        * **Clipboard-Hash-Wiederherstellung:** Wenn du ein Bild aus der Zwischenablage (`Cmd+V`) einfügst (wo Browser Header löschen), erkennt FIBO Studio das Bild anhand eines visuellen Hash-Abgleichs in deiner lokalen Historie wieder und stellt den Prompt automatisch wieder her!
                        """
                    )

                with gr.Tab("🇬🇧 English (User Guide)"):
                    gr.Markdown(
                        """
                        ## 🎨 FIBO Studio – User Guide & Workflow Reference

                        Welcome to **FIBO Studio**, the premier creative workstation built natively for **Apple Silicon Macs** (M1/M2/M3/M4) powered by **Apple MLX** and **Metal Performance Shaders (MPS)**.

                        ---

                        ### 1. ⚡ Architecture & Performance on Apple Silicon
                        * **Zero-Copy Unified Memory:** FIBO Studio allocates directly from unified system RAM at up to 800 GB/s bandwidth. No PCIe transfer or VRAM bottlenecking.
                        * **Native MLX Engine:** Diffusion transformers execute via Apple's custom Metal kernels with 8-bit and 4-bit weight quantization (`QuantizedLinear`).
                        * **External SSD Hub:** Automatically caches multi-gigabyte HuggingFace models on your high-speed external drive (`/Volumes/...`), keeping your internal Macintosh HD uncluttered.

                        ---

                        ### 2. ✏️ Image Edit (FIBO 2 Turbo – 4 Steps)
                        Engineered for high-precision, surgical modifications:
                        * **Reference Slots 1 to 5:** Place your base canvas in Slot 1. Add supporting references in Slots 2–5 (e.g. clothing, color palettes, face styles).
                        * **Prompt Syntax:** Reference specific slots directly:
                          * `Make the person in <image_1> wear the jacket from <image_2>.`
                          * `Replace the background with a cinematic cyberpunk skyline.`
                        * **Inpainting Mask:** Draw or upload a binary mask to constrain changes to exact regions.
                        * **Iterative Loop:** Click **"🔄 Ergebnis als Referenzbild 1 übernehmen"** to turn your output into the next iteration's base canvas.
                        * **📦 Bulk Batch Processing:** Specify a directory or drag & drop multiple images. Each file is sequentially assigned to `<image_1>` with matching prompts and references — ideal for film sequences, watermark/UI removal, or series upscaling. *(Non-image files like HTML or TXT are safely ignored).*
                        * **⚡ Additive Modifier Toolbar:** Toggle 1-click prompt modifiers directly above the prompt box (Hollywood cinematic color grading, video player UI / instrument cleaning, sharpening, outpainting, facial enhancement).

                        ---

                        ### 3. 🎨 Text to Image (Turbo vs. Base/Dev)
                        * **FIBO 2 Turbo (4 Steps):** Rapid SFT diffusion generator for lightning-fast prototyping.
                        * **FIBO 2 Base/Dev (30 Steps):** Full 30-step diffusion with CFG 3.5 for maximum structural fidelity and composition.
                        * **VLM Prompt Expansion:** Uses an on-device Vision Language Model to automatically enrich short prompts with photorealistic descriptors.
                        * **Aspect Ratio Engine:** One-click presets (1:1, 16:9, 9:16, 21:9) with automatic 16-pixel alignment.

                        ---

                        ### 4. ✨ Post-Processing & Upscaling
                        Conveniently located beneath the result display in both generation tabs:
                        * **Lanczos Resizer (Instant 0.05s):** Fast CPU scaling (1.5x, 2x, 4x) with adaptive Unsharp Mask detail sharpening. Zero VRAM footprint.
                        * **SeedVR2 (Native MLX Diffusion):** Dedicated 1-step diffusion super-resolution for high-fidelity 4K output.
                        * **Real-ESRGAN:** AI texture restoration for intricate patterns and fabric textures.
                        * **👤 Face Refiner:** Reconstructs facial features, eyes, skin texture, and mouth contours.

                        ---

                        ### 5. 👁️ Vision AI & WD-Tagger (Tab 3)
                        * **Qwen2-VL 4-Bit (MLX):** Deep visual understanding generating descriptive prompts at **~110 tokens/second**.
                        * **WD-Tagger (SwinV2 on Apple MPS):** High-speed tag classification running in **<0.4 seconds**.
                        * **One-Click Injection:** Send analyzed captions or tags straight into Edit or Text-to-Image with a single click.

                        ---

                        ### 6. 🔍 Metadata & Smart History Recovery
                        * **Multi-Standard Embedding:** Generated PNGs carry standard A1111 `parameters`, ComfyUI `prompt`, EXIF `UserComment`, XMP, and IPTC chunks.
                        * **Clipboard History Fallback:** If you paste an image via clipboard (`Cmd+V`) where browsers strip EXIF headers, FIBO Studio matches the structural thumbnail hash against local outputs to restore the exact prompt and generation parameters.
                        """
                    )

    with gr.Accordion("📁 Automatische Bildspeicherung (Outputs)", open=False):
        out_root_display = str(get_output_dir("edit").parent)
        gr.Markdown(
            f"💡 **Speicherort:** Alle generierten Bilder werden mit vollständigen eingebetteten Metadaten dauerhaft gespeichert in:  \n"
            f"`{out_root_display}`"
        )
        with gr.Row():
            btn_open_outputs = gr.Button("📂 Ausgabeordner im Mac-Finder öffnen", variant="secondary", scale=4)
            with gr.Column(scale=6):
                outputs_status_txt = gr.Markdown(value="*Bereit.*")
        btn_open_outputs.click(
            fn=open_outputs_in_finder,
            inputs=[],
            outputs=[outputs_status_txt],
        )


if __name__ == "__main__":
    cleanup_mac_files(os.path.dirname(__file__))

    allowed_roots = [
        str(Path.home()),
        "/Users",
        "/Volumes",
        "/home",
        os.path.dirname(__file__),
    ]
    allowed_roots = [d for d in allowed_roots if os.path.exists(d)]

    is_mac = platform.system().lower() == "darwin"
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1" if is_mac else "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7865"))

    demo.queue().launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        inbrowser=is_mac,
        allowed_paths=allowed_roots,
    )
