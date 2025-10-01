from __future__ import annotations
from dataclasses import dataclass

@dataclass
class CorrectionQC:
    dE_med: float
    dE_p95: float
    ssim_L: float
    spill_dE: float
    passed: bool

def evaluate(dE_med: float, dE_p95: float, ssim_L_val: float, spill_dE: float,
             max_dE_med: float, max_dE_p95: float, min_ssim_L: float, max_spill_dE: float) -> CorrectionQC:
    passed = (
        (dE_med <= max_dE_med) and
        (dE_p95 <= max_dE_p95) and
        (ssim_L_val >= min_ssim_L) and
        (spill_dE <= max_spill_dE)
    )
    return CorrectionQC(dE_med, dE_p95, ssim_L_val, spill_dE, passed)
