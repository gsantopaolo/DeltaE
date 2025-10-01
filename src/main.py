from __future__ import annotations
import argparse, yaml
from pathlib import Path
from src.schemas.config import AppConfig
from src.utils.logging_utils import get_logger
from src.pipeline.io import discover_pairs
from src.pipeline.orchestrator import run_pairs

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Color Corrector Pipeline")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--input-dir", type=Path)
    ap.add_argument("--limit", type=int)
    return ap.parse_args()

def load_config(path: Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.input_dir:
        cfg.paths.input_dir = str(args.input_dir)
    if args.limit:
        cfg.run.limit = args.limit

    log = get_logger("main", Path(cfg.paths.logs_dir))
    log.info("🚀 Starting color-correction run")
    log.info(f"📁 input_dir={cfg.paths.input_dir} | output_dir={cfg.paths.output_dir} | masks_dir={cfg.paths.masks_dir} | limit={cfg.run.limit}")

    input_dir = Path(cfg.paths.input_dir)
    pairs = discover_pairs(input_dir, cfg.run.limit)
    if not pairs:
        log.error("❌ No pairs found. Ensure files like still-life-00010.jpg and on-model-00010.jpg exist.")
        raise SystemExit(1)

    run_pairs(
        pairs=pairs,
        masks_dir=Path(cfg.paths.masks_dir),
        output_dir=Path(cfg.paths.output_dir),
        logs_dir=Path(cfg.paths.logs_dir),
        cfg=cfg,
    )
    log.info("🎉 Done.")

if __name__ == "__main__":
    main()
