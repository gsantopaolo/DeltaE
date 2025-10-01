from __future__ import annotations
import numpy as np
import cv2
from rembg import remove
from .base import Masker

class StillLifeRembgMasker(Masker):
    def get_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        # rembg expects RGB
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgba = remove(rgb)  # returns RGBA
        if rgba.shape[-1] == 4:
            alpha = rgba[..., 3]
        else:
            # fallback: make a rough mask if alpha missing
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        # binarize + largest component
        _, binm = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
        if num <= 1:
            return binm
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return np.where(labels == largest, 255, 0).astype(np.uint8)
