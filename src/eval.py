"""Evaluation entrypoint (Ultralytics YOLOv8)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    data_path = Path(cfg["data"])
    if not data_path.is_absolute():
        # Try resolving relative to config dir, then cwd, then repo root (parent of configs/)
        candidates = [
            (cfg_path.parent / data_path),
            (Path.cwd() / data_path),
            (cfg_path.parent.parent / data_path),
        ]
        for cand in candidates:
            if cand.exists():
                data_path = cand.resolve()
                break
        else:
            data_path = (cfg_path.parent / data_path).resolve()
    model = YOLO(args.weights)
    model.val(
        data=str(data_path),
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        device=args.device if args.device is not None else cfg.get("device", 0),
        conf=args.conf if args.conf is not None else cfg.get("conf", None),
        split="test",
    )
