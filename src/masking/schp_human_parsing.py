# src/masking/schp_human_parsing.py
from __future__ import annotations
import sys, subprocess, tempfile
from pathlib import Path
from typing import Iterable, List, Dict
import numpy as np
import cv2
from PIL import Image

from ..utils.logging_utils import get_logger

# LIP label ids (from SCHP README)
LIP_NAME_TO_ID: Dict[str, int] = {
    "background": 0, "hat": 1, "hair": 2, "glove": 3, "sunglasses": 4,
    "upper-clothes": 5, "dress": 6, "coat": 7, "socks": 8, "pants": 9,
    "jumpsuits": 10, "scarf": 11, "skirt": 12, "face": 13, "left-arm": 14,
    "right-arm": 15, "left-leg": 16, "right-leg": 17, "left-shoe": 18, "right-shoe": 19
}

def _largest_component(binm: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if num <= 1:
        return binm
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)

class SCHPParser:
    """
    Wraps GoGoDuck912/SCHP 'simple_extractor.py' via subprocess.
    Produces a binary garment mask (uint8 {0,255}) for selected garment labels.
    """

    def __init__(
        self,
        repo_dir: str = "third_party/schp",
        weights_path: str = "weights/exp-schp-201908261155-lip.pth",
        include_labels: Iterable[str] = ("upper-clothes", "coat", "dress"),
        python: str | None = None,
    ) -> None:
        self.log = get_logger("SCHP")
        self.repo = Path(repo_dir).resolve()              # absolute
        self.simple_name = "simple_extractor.py"          # run by name with cwd=self.repo
        self.weights = Path(weights_path).resolve()       # absolute
        self.python = python or sys.executable
        self.include_ids: List[int] = [LIP_NAME_TO_ID[n] for n in include_labels if n in LIP_NAME_TO_ID]

        # sanity checks
        simple_abs = self.repo / self.simple_name
        self._ready = self.repo.exists() and simple_abs.exists() and self.weights.exists()
        if not self.repo.exists():
            self.log.warning(f"⚠️ SCHP repo not found at {self.repo}")
        if not simple_abs.exists():
            self.log.warning(f"⚠️ simple_extractor.py not found at {simple_abs}")
        if not self.weights.exists():
            self.log.warning(f"⚠️ SCHP weights not found at {self.weights}")

        if self._ready:
            self.log.info("🧵 SCHP (CLI) ready.")
        else:
            self.log.warning("⚠️ SCHP (CLI) not ready; will be skipped if called.")

    def is_ready(self) -> bool:
        return self._ready

    def get_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self._ready:
            raise RuntimeError("SCHPParser not ready (missing repo or weights).")

        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            inp = (tdir / "in").resolve()
            out = (tdir / "out").resolve()
            inp.mkdir(); out.mkdir()

            # Save input
            src = inp / "img.png"
            cv2.imwrite(str(src), image_bgr)

            # Run SCHP CLI (CPU if no CUDA; works on macOS)
            cmd = [
                self.python, self.simple_name,
                "--dataset", "lip",
                "--model-restore", str(self.weights),  # absolute
                "--input-dir", str(inp),               # absolute
                "--output-dir", str(out),              # absolute
            ]
            run = subprocess.run(
                cmd, cwd=str(self.repo),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if run.returncode != 0:
                err = (run.stderr or run.stdout).decode(errors="ignore")
                raise RuntimeError(f"SCHP CLI failed ({run.returncode}): {err}")

            # Read predicted label PNG (palette image). PIL returns indices directly.
            pred_path = out / "img.png"
            if not pred_path.exists():
                raise RuntimeError("SCHP produced no output PNG.")
            lab = np.array(Image.open(pred_path))  # 'P' mode → integer label map

            # Keep only garment ids we care about (upper/dress/coat)
            mask = np.isin(lab, self.include_ids).astype(np.uint8) * 255

            # Post-process a bit
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
            mask = _largest_component(mask)
            return mask
