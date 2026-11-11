"""Internationalization (i18n) module for FIBO Studio.
Supports English (default) and German, with seamless dictionary lookups.
"""

from typing import Dict, Any

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        # App Header
        "app_title": "FIBO Studio",
        "app_subtitle": "High-Performance Cross-Platform Interface for BRIA FIBO 2.0",
        "app_tagline": "Auto-Hardware Acceleration • SageAttention 2 / MLX • WanGP Workflows",
        
        # Tabs
        "tab_edit": "✏️ Image Edit (FIBO 2 Turbo)",
        "tab_t2i": "🎨 Text to Image (FIBO 2 Base/Dev)",
        "tab_vision": "👁️ Vision & Tagging (WD14 / VLM)",
        "tab_png_info": "🔍 PNG Info & Metadata",
        "tab_help": "ℹ️ Help & Documentation",
        
        # Accordions
        "acc_model_manager": "📦 Model Status & Download Manager",
        "acc_hf_auth": "🔑 Hugging Face Authentication",
        "acc_ref_images": "Additional Reference Images (Slots 2 – 5)",
        "acc_inpaint_mask": "Inpainting Mask (Optional)",
        "acc_extract_prompt": "📥 Extract Prompt from any Image",
        "acc_advanced_settings": "Advanced Generation Settings",
        "acc_lora": "LoRA Model (Optional)",
        "acc_bulk_gallery": "📦 Batch Processing Gallery (Bulk Results)",
        "acc_postprocessing": "✨ Post-Processing & Upscaling (Lanczos, SeedVR2, Real-ESRGAN, Face Refiner)",
        
        # Form Controls
        "label_ref1": "Main Reference Image <image_1>",
        "label_ref2": "Reference Image <image_2>",
        "label_ref3": "Reference Image <image_3>",
        "label_ref4": "Reference Image <image_4>",
        "label_ref5": "Reference Image <image_5>",
        "label_mask": "Mask Image (White = Edit, Black = Keep)",
        "label_edit_prompt": "Edit Prompt",
        "placeholder_edit_prompt": "e.g. Make him wear sunglasses and a leather jacket in the style of <image_2>...",
        "label_t2i_prompt": "Prompt",
        "placeholder_t2i_prompt": "e.g. A cinematic photograph of an astronaut in a lush alien rainforest...",
        "label_t2i_negative": "Negative Prompt",
        "label_aspect_ratio": "Aspect Ratio & Resolution",
        "label_quantization": "Quantization / Precision",
        "label_steps": "Inference Steps",
        "label_guidance": "Guidance Scale (CFG)",
        "label_seed": "Seed",
        "label_random_seed": "Randomize Seed",
        "label_lora_path": "LoRA File Path (.safetensors)",
        "label_lora_scale": "LoRA Scale / Weight",
        "label_output_image": "Generated Output",
        "label_custom_modifiers": "Saved Modifiers (Custom Presets)",
        "label_new_mod_name": "New Modifier Name",
        "placeholder_new_mod_name": "e.g. 1980s Retro Synthwave",
        "label_new_mod_text": "Modifier Prompt Text",
        "placeholder_new_mod_text": "Text to be appended to the prompt...",
        
        # Bulk Processing
        "label_bulk_folder": "Local Folder Path with Images",
        "label_bulk_recursive": "Scan Subdirectories",
        "label_bulk_files": "Or Drag & Drop Images (Multi-selection)",
        "btn_bulk_scan": "🔍 Scan & Verify Images",
        "btn_bulk_clear": "Clear Selection",
        "btn_bulk_start": "🚀 Start Batch Processing (Run all as <image_1>)",
        "btn_bulk_open_finder": "📂 Open Output Folder in File Explorer",
        
        # Action Buttons
        "btn_run_edit": "✨ Run Edit (4 Steps Turbo)",
        "btn_run_t2i": "🎨 Generate Image",
        "btn_apply_mod": "📋 Apply",
        "btn_new_mod": "➕ New",
        "btn_del_mod": "🗑️ Delete",
        "btn_save_mod": "💾 Save",
        "btn_cancel": "Cancel",
        "btn_copy_cur_prompt": "⬇️ Copy Current Edit Prompt",
        "btn_send_to_ref1": "🔄 Use Output as Reference Image 1",
        "btn_download_model": "⬇️ Download Model Now",
        "btn_refresh_status": "🔄 Refresh Status",
        "btn_save_token": "💾 Save Token & Login",
        "btn_run_upscale": "✨ Apply Upscale / Refine",
        
        # Hardware Status
        "hw_detecting": "Detecting hardware acceleration...",
        "hw_cuda_ready": "🚀 NVIDIA CUDA Active",
        "hw_mlx_ready": "🍏 Apple Silicon MLX Active",
        "hw_cpu_ready": "⚙️ CPU Fallback Active",
    },
    
    "de": {
        # App Header
        "app_title": "FIBO Studio",
        "app_subtitle": "High-Performance Cross-Platform Interface für BRIA FIBO 2.0",
        "app_tagline": "Auto-Hardware-Beschleunigung • SageAttention 2 / MLX • WanGP Workflows",
        
        # Tabs
        "tab_edit": "✏️ Image Edit (FIBO 2 Turbo)",
        "tab_t2i": "🎨 Text to Image (FIBO 2 Base/Dev)",
        "tab_vision": "👁️ Bildanalyse & Tagging (WD14 / VLM)",
        "tab_png_info": "🔍 PNG Info & Metadaten",
        "tab_help": "ℹ️ Hilfe & Dokumentation",
        
        # Accordions
        "acc_model_manager": "📦 Modell-Status & Download-Manager",
        "acc_hf_auth": "🔑 Hugging Face Authentifizierung",
        "acc_ref_images": "Zusätzliche Referenzbilder (Slots 2 – 5)",
        "acc_inpaint_mask": "Inpainting-Maske (Optional)",
        "acc_extract_prompt": "📥 Prompt aus beliebigem Bild auslesen & einfügen",
        "acc_advanced_settings": "Erweiterte Einstellungen",
        "acc_lora": "LoRA Modell (Optional)",
        "acc_bulk_gallery": "📦 Galerie der Stapelverarbeitung (Bulk Results)",
        "acc_postprocessing": "✨ Post-Processing & Upscaling (Lanczos, SeedVR2, Real-ESRGAN, Face Refiner)",
        
        # Form Controls
        "label_ref1": "Haupt-Referenzbild <image_1>",
        "label_ref2": "Referenzbild <image_2>",
        "label_ref3": "Referenzbild <image_3>",
        "label_ref4": "Referenzbild <image_4>",
        "label_ref5": "Referenzbild <image_5>",
        "label_mask": "Mask Image (Weiß = Bearbeiten, Schwarz = Beibehalten)",
        "label_edit_prompt": "Edit Prompt",
        "placeholder_edit_prompt": "z.B. Make him wear sunglasses and a leather jacket in the style of <image_2>...",
        "label_t2i_prompt": "Prompt",
        "placeholder_t2i_prompt": "z.B. A cinematic photograph of an astronaut in a lush alien rainforest...",
        "label_t2i_negative": "Negativer Prompt",
        "label_aspect_ratio": "Seitenverhältnis & Auflösung",
        "label_quantization": "Quantisierung (Präzision)",
        "label_steps": "Inferenz-Steps",
        "label_guidance": "Guidance (CFG)",
        "label_seed": "Seed",
        "label_random_seed": "Zufälliger Seed",
        "label_lora_path": "LoRA Dateipfad (.safetensors)",
        "label_lora_scale": "LoRA Skalierung / Gewicht",
        "label_output_image": "Ergebnis",
        "label_custom_modifiers": "Gespeicherte Modifier (Custom Presets)",
        "label_new_mod_name": "Name des neuen Modifiers",
        "placeholder_new_mod_name": "z.B. 1980s Retro Synthwave",
        "label_new_mod_text": "Prompt / Text des Modifiers",
        "placeholder_new_mod_text": "Text, der an den Prompt angehängt werden soll...",
        
        # Bulk Processing
        "label_bulk_folder": "Lokaler Ordnerpfad mit Bildern",
        "label_bulk_recursive": "Unterordner ebenfalls durchsuchen",
        "label_bulk_files": "Oder Dateien per Drag & Drop hochladen (Mehrfachauswahl)",
        "btn_bulk_scan": "🔍 Bilder scannen & prüfen",
        "btn_bulk_clear": "Auswahl leeren",
        "btn_bulk_start": "🚀 Stapelverarbeitung jetzt starten (Alle Bilder als <image_1> abarbeiten)",
        "btn_bulk_open_finder": "📂 Ausgabeordner im Finder / Explorer öffnen",
        
        # Action Buttons
        "btn_run_edit": "✨ Run Edit (4 Steps Turbo)",
        "btn_run_t2i": "🎨 Bild jetzt generieren",
        "btn_apply_mod": "📋 Anwenden",
        "btn_new_mod": "➕ Neu",
        "btn_del_mod": "🗑️ Löschen",
        "btn_save_mod": "💾 Speichern",
        "btn_cancel": "Abbrechen",
        "btn_copy_cur_prompt": "⬇️ Aus aktuellem Edit Prompt",
        "btn_send_to_ref1": "🔄 Ergebnis als Referenzbild 1 übernehmen",
        "btn_download_model": "⬇️ Modell jetzt herunterladen",
        "btn_refresh_status": "🔄 Status aktualisieren",
        "btn_save_token": "💾 Token speichern & Login",
        "btn_run_upscale": "✨ Hochskalierung / Refinement anwenden",
        
        # Hardware Status
        "hw_detecting": "Hardware-Beschleunigung wird erkannt...",
        "hw_cuda_ready": "🚀 NVIDIA CUDA Aktiv",
        "hw_mlx_ready": "🍏 Apple Silicon MLX Aktiv",
        "hw_cpu_ready": "⚙️ CPU Fallback Aktiv",
    }
}


def t(key: str, lang: str = "en") -> str:
    """Translates a key to the requested language, falling back to English."""
    selected = TRANSLATIONS.get(lang.lower(), TRANSLATIONS["en"])
    return selected.get(key, TRANSLATIONS["en"].get(key, key))

