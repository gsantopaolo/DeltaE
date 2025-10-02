from __future__ import annotations
import numpy as np, cv2, torch
from segment_anything import sam_model_registry, SamPredictor
from ..utils.logging_utils import get_logger

class SAMV1Refiner:
    def __init__(self, checkpoint_path: str, model_type: str = "vit_h"):
        self.logger = get_logger("SAMv1")
        self.ok = False
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device)
            self.pred = SamPredictor(sam)
            self.device = device
            self.ok = True
            self.logger.info(f"✅ SAM v1 loaded ({model_type}) on {device}.")
        except Exception as e:
            self.logger.error(f"❌ SAM v1 init failed: {e}")

    def is_ready(self) -> bool:
        return self.ok

    def refine(self, image_bgr: np.ndarray, init_mask: np.ndarray, expand_box_px: int = 16) -> np.ndarray:
        if not self.ok:
            return init_mask
        ys, xs = np.where(init_mask > 0)
        if xs.size == 0:
            return init_mask

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        H, W = image_bgr.shape[:2]
        x0 = max(0, x0 - expand_box_px); y0 = max(0, y0 - expand_box_px)
        x1 = min(W-1, x1 + expand_box_px); y1 = min(H-1, y1 + expand_box_px)
        box = np.array([x0, y0, x1, y1], dtype=np.float32)

        step = max(1, xs.size // 5)
        pts = np.stack([xs[::step][:5], ys[::step][:5]], axis=1).astype(np.float32)
        lbl = np.ones((pts.shape[0],), dtype=np.int32)

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.pred.set_image(img_rgb)
        masks, _, _ = self.pred.predict(
            point_coords=pts, point_labels=lbl, box=box[None], multimask_output=False
        )
        return (masks[0].astype(np.uint8) * 255)
