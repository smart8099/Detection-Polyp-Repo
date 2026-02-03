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
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    model = YOLO(args.weights)
    model.val(
        data=cfg["data"],
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        device=cfg.get("device", 0),
        split="test",
    )
