# Detection (YOLOv8)

Standalone repo to train a polyp detector using masks-to-bboxes conversion.

## Setup
```
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Prepare data (YOLO format)
Put the zip files in `data/` and run:
```
python scripts/prepare_yolo_from_masks.py --data-dir data --out-root data/yolo_polyp
```

This will:
- unzip each dataset
- find `images/` + `masks/`
- convert masks to YOLO boxes
- write `data/yolo_polyp/data.yaml`

## Train
```
python src/train.py --config configs/train.yaml
```

## Eval
```
python src/eval.py --config configs/train.yaml --weights runs/polyp/weights/best.pt
```

## Export
```
python src/export.py --weights runs/polyp/weights/best.pt --format onnx
```
