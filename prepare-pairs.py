#!/usr/bin/env python3
"""

Copies the first N (=300 by default) matching JPGs from:
  - excluded/datasets/cloth  -> dataset/still-life-<ID>.jpg
  - excluded/datasets/image  -> dataset/on-model-<ID>.jpg

Where <ID> is the zero-padded number before the first underscore in the filename,
e.g., 00010_00.jpg -> 00010

Adds pre-flight checks:
- Verify cloth & image folders exist AND contain JPG/JPEG files.
- If not, log an error suggesting to download/unzip VITON-HD (zalando-hd-resized.zip).

Logging is verbose and emoji-friendly. Type hints included.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

# ----------------------------
# Logging setup
# ----------------------------
LOGGER_NAME = "pair-prep"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_handler)


# ----------------------------
# Helpers
# ----------------------------
def list_sorted_jpgs(folder: Path) -> List[Path]:
    """Return lexicographically sorted list of .jpg/.jpeg files in a folder. Raises if folder invalid."""
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder}")
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}])
    return files


def extract_id_from_filename(filename: str) -> str:
    """
    Extract the zero-padded numeric id before the first underscore.
    Example: '00010_00.jpg' -> '00010'
    Falls back to stem if underscore not present.
    """
    stem = Path(filename).stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def ensure_dir(folder: Path) -> None:
    """Create folder if it doesn't exist."""
    folder.mkdir(parents=True, exist_ok=True)


def copy_with_new_name(src: Path, dst_folder: Path, new_name: str) -> None:
    """Copy file to destination with a new name."""
    dst = dst_folder / new_name
    shutil.copy2(src, dst)


def validate_pair_files(cloth: Path, image: Path) -> None:
    """Basic validation: exist + non-zero size."""
    if not cloth.exists() or cloth.stat().st_size == 0:
        raise FileNotFoundError(f"Cloth file missing or empty: {cloth}")
    if not image.exists() or image.stat().st_size == 0:
        raise FileNotFoundError(f"Image file missing or empty: {image}")


def verify_source_folders(cloth_dir: Path, image_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Ensure source folders exist and contain JPG/JPEG files.
    If not, log a friendly error suggesting to download/unzip the dataset zip.
    Returns sorted file lists for both folders.
    """
    logger.info("🧪 Verifying source folders...")

    try:
        cloth_files = list_sorted_jpgs(cloth_dir)
    except Exception as e:
        logger.error(
            "❌ Cloth folder issue: %s\n"
            "🪄 Fix: Please download and unzip the VITON-HD preprocessed dataset "
            "(e.g., 'zalando-hd-resized.zip') so that it contains 'cloth' images here:\n"
            f"   → {cloth_dir}\n", exc_info=False
        )
        raise

    try:
        image_files = list_sorted_jpgs(image_dir)
    except Exception as e:
        logger.error(
            "❌ Image folder issue: %s\n"
            "🪄 Fix: Please download and unzip the VITON-HD preprocessed dataset "
            "(e.g., 'zalando-hd-resized.zip') so that it contains 'image' photos here:\n"
            f"   → {image_dir}\n", exc_info=False
        )
        raise

    if not cloth_files:
        msg = (
            f"❌ No JPG/JPEG files found in cloth folder: {cloth_dir}\n"
            "🪄 Fix: Download 'zalando-hd-resized.zip' (VITON-HD), unzip, and ensure "
            "the 'cloth' directory contains .jpg files."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    if not image_files:
        msg = (
            f"❌ No JPG/JPEG files found in image folder: {image_dir}\n"
            "🪄 Fix: Download 'zalando-hd-resized.zip' (VITON-HD), unzip, and ensure "
            "the 'image' directory contains .jpg files."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(f"📦 Found {len(cloth_files)} cloth files in {cloth_dir}")
    logger.info(f"📦 Found {len(image_files)} on-model files in {image_dir}")

    return cloth_files, image_files


# ----------------------------
# Core routine
# ----------------------------
def prepare_pairs(
    cloth_dir: Path,
    image_dir: Path,
    dest_dir: Path,
    limit: int = 300,
) -> Tuple[int, int, int]:
    """
    Process up to `limit` pairs, copying to dest_dir with required names.
    Returns (processed, skipped, errors).
    """
    # Pre-flight checks
    cloth_files, image_files = verify_source_folders(cloth_dir, image_dir)

    logger.info("🗂️  Scanning source files...")
    ensure_dir(dest_dir)

    processed = 0
    skipped = 0
    errors = 0

    # Build map for quick lookup on-model counterpart by exact filename
    image_map: Dict[str, Path] = {p.name: p for p in image_files}

    # Iterate the first `limit` cloth files in lexicographic order
    for idx, cloth_path in enumerate(cloth_files[:limit], start=1):
        fname = cloth_path.name
        on_model_path = image_map.get(fname)

        file_id = extract_id_from_filename(fname)
        still_life_name = f"still-life-{file_id}.jpg"
        on_model_name = f"on-model-{file_id}.jpg"

        try:
            if on_model_path is None:
                logger.warning(f"⚠️  Missing on-model counterpart for '{fname}'. Skipping.")
                skipped += 1
                continue

            validate_pair_files(cloth_path, on_model_path)

            # Copy with new names
            copy_with_new_name(cloth_path, dest_dir, still_life_name)
            logger.info(f"✅ [{idx}] Copied cloth  -> {dest_dir / still_life_name} 👕")

            copy_with_new_name(on_model_path, dest_dir, on_model_name)
            logger.info(f"✅ [{idx}] Copied model -> {dest_dir / on_model_name} 🧍")

            processed += 1

        except Exception as e:
            logger.error(f"❌ Error processing '{fname}': {e}")
            errors += 1

    logger.info("🧾 Summary:")
    logger.info(f"   • Processed pairs: {processed} ✅")
    logger.info(f"   • Skipped (missing counterpart): {skipped} ⚠️")
    logger.info(f"   • Errors: {errors} ❌")
    logger.info(f"📁 Output folder: {dest_dir.resolve()}")

    return processed, skipped, errors


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy first N paired images into a single dataset folder.")
    parser.add_argument(
        "--cloth-dir",
        type=Path,
        default=Path("excluded/datasets/cloth"),
        help="Path to the 'cloth' images directory.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("excluded/datasets/image"),
        help="Path to the 'image' (on-model) images directory.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("dataset"),
        help="Destination directory for the renamed copies.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of pairs to copy (default: 300).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("🚀 Starting pair preparation with settings:")
    logger.info(f"   • cloth_dir = {args.cloth_dir}")
    logger.info(f"   • image_dir = {args.image_dir}")
    logger.info(f"   • dest_dir  = {args.dest_dir}")
    logger.info(f"   • limit     = {args.limit}")

    try:
        prepare_pairs(args.cloth_dir, args.image_dir, args.dest_dir, args.limit)
    except Exception as e:
        logger.error(
            "💥 Fatal error: %s\n"
            "🪄 If you haven't yet, please download the VITON-HD preprocessed dataset "
            "(e.g., 'zalando-hd-resized.zip'), unzip it, and make sure you have:\n"
            "   • excluded/datasets/cloth/*.jpg\n"
            "   • excluded/datasets/image/*.jpg\n", exc_info=False
        )
        raise SystemExit(1)
    else:
        logger.info("🎉 Done.")


if __name__ == "__main__":
    main()
