"""Utility functions for FIBO Studio."""

import os
from pathlib import Path


def cleanup_mac_files(directory: str | Path) -> int:
    """
    Recursively deletes hidden AppleDouble (._*) metadata files created by macOS.
    Returns the count of removed files.
    """
    deleted = 0
    target_dir = Path(directory)
    if not target_dir.exists():
        return 0

    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.startswith("._"):
                try:
                    os.remove(os.path.join(root, f))
                    deleted += 1
                except OSError:
                    pass
    return deleted

