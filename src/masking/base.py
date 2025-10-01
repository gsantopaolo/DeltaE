from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class Masker(ABC):
    @abstractmethod
    def get_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return binary mask (uint8 0/255), same HxW as image."""
        raise NotImplementedError

    @staticmethod
    def erode(mask: np.ndarray, px: int) -> np.ndarray:
        import cv2
        if px <= 0:
            return mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px*2+1, px*2+1))
        return cv2.erode(mask, k)

    @staticmethod
    def feather(mask: np.ndarray, px: int) -> np.ndarray:
        import cv2
        if px <= 0:
            return mask
        return cv2.GaussianBlur(mask, (0, 0), px)
