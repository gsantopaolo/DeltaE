from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
from colorlog import ColoredFormatter

def get_logger(name: str,
               log_dir: Optional[Path] = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        # Console handler (colored)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = ColoredFormatter(
            "%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler (plain)
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            logger.addHandler(fh)

    return logger
