from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class ColorCorrector(ABC):
    @abstractmethod
    def correct(self,
                on_model_bgr: np.ndarray,
                on_model_mask: np.ndarray,
                ref_bgr: np.ndarray,
                ref_mask: np.ndarray) -> np.ndarray:
        """Return corrected BGR image same shape as on_model_bgr."""
        raise NotImplementedError
