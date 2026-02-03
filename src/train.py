"""Training entrypoint (Ultralytics YOLOv8)."""

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
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    model = YOLO(cfg.get("model", "yolov8n.pt"))
    model.train(
        data=cfg["data"],
        epochs=cfg.get("epochs", 100),
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        device=cfg.get("device", 0),
        workers=cfg.get("workers", 4),
        project=cfg.get("project", "runs"),
        name=cfg.get("name", "polyp"),
    )
