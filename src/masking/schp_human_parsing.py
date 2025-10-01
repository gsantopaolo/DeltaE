from __future__ import annotations
import numpy as np, cv2, os
from typing import Dict, List, Optional
from ..utils.logging_utils import get_logger

# SCHP label names (LIP-like / CIHP-style may differ; adjust to your weights)
DEFAULT_LABELS = [
    "background","hat","hair","glove","sunglasses","upper-clothes","dress","coat","socks","pants",
    "jumpsuit","scarf","skirt","face","left-arm","right-arm","left-leg","right-leg","left-shoe","right-shoe",
    "t-shirt","jacket","vest","hoodie","cardigan","blouse","sweater"
]

class SCHPParser:
    def __init__(self, weights_path: str, device: str = "mps", include_labels: Optional[List[str]] = None):
        self.ok = False
        self.labels = DEFAULT_LABELS
        self.include_labels = set(include_labels or ["upper-clothes","coat","dress","jacket","t-shirt","vest","hoodie","cardigan","blouse","sweater"])
        self.device = device
        self.weights_path = weights_path
        self.logger = get_logger("SCHP")
        try:
            import torch
            from torchvision import transforms
            self.torch = torch
            self.transforms = transforms
            if not os.path.isfile(weights_path):
                self.logger.warning(f"⚠️ SCHP weights not found at {weights_path}. Will skip SCHP.")
                return
            # ---- Minimal demonstration model loader (replace with your arch) ----
            # We assume weights store a scripted model or state dict with an entry "model".
            model = torch.jit.load(weights_path, map_location=device) if weights_path.endswith(".pt") else None
            if model is None:
                # fallback: pretend loading a traced scripted model; your real project should load the actual arch
                self.logger.warning("⚠️ Could not load scripted SCHP; expecting a torchscript .pt. Skipping SCHP.")
                return
            model.eval()
            self.model = model
            self.ok = True
            self.logger.info("✅ SCHP loaded.")
        except Exception as e:
            self.logger.warning(f"⚠️ SCHP unavailable: {e}. Will use fallbacks.")
            self.ok = False

    def is_ready(self) -> bool:
        return self.ok

    def get_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self.ok:
            raise RuntimeError("SCHP not initialized")
        # Preprocess (dummy; replace with your model's expected preprocessing)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = self.transforms.ToTensor()(rgb).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            out = self.model(x)              # NxCxHxW logits
        logits = out[0].detach().cpu().numpy()
        labelmap = np.argmax(logits, axis=0).astype(np.int32)

        # Build garment mask by label names
        mask = np.zeros(labelmap.shape, np.uint8)
        for idx, name in enumerate(self.labels):
            if name in self.include_labels:
                mask[labelmap == idx] = 255
        # Cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=2)
        return mask
