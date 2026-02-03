"""Export entrypoint (Ultralytics YOLOv8)."""

from __future__ import annotations

import argparse

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="onnx")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.export(format=args.format)
