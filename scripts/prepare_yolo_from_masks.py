from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


IMG_DIR_NAMES = {"images", "image", "imgs", "img", "original"}
MASK_DIR_NAMES = {"masks", "mask", "groundtruth", "ground truth", "gt"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_dir(root: Path, names: set[str]) -> Path | None:
    candidates = []
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in names:
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: ("png" not in str(p).lower(), len(p.parts)))
    return candidates[0]


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def prepare_dataset(zip_path: Path, out_root: Path, name: str) -> Path:
    out_dir = out_root / name
    if not out_dir.exists():
        extract_zip(zip_path, out_dir)

    images_dir = find_dir(out_dir, IMG_DIR_NAMES)
    masks_dir = find_dir(out_dir, MASK_DIR_NAMES)
    if images_dir is None or masks_dir is None:
        raise FileNotFoundError(f"Could not find images/masks in {out_dir}")

    # Canonical links
    canonical_images = out_dir / "images"
    canonical_masks = out_dir / "masks"
    if not canonical_images.exists():
        os.symlink(images_dir, canonical_images, target_is_directory=True)
    if not canonical_masks.exists():
        os.symlink(masks_dir, canonical_masks, target_is_directory=True)
    return out_dir


def image_pairs(root: Path) -> List[Tuple[Path, Path]]:
    images_dir = root / "images"
    masks_dir = root / "masks"
    mask_map = {p.stem: p for p in masks_dir.rglob("*") if p.is_file()}
    pairs: List[Tuple[Path, Path]] = []
    for img in images_dir.rglob("*"):
        if img.is_file():
            mask = mask_map.get(img.stem)
            if mask:
                pairs.append((img, mask))
    return pairs


def mask_to_bboxes(mask_path: Path, min_area: int) -> List[Tuple[int, int, int, int]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            bboxes.append((x, y, w, h))
    if not bboxes and np.count_nonzero(binary) > 0:
        ys, xs = np.where(binary > 0)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        bboxes.append((int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)))
    return bboxes


def write_yolo_label(label_path: Path, bboxes: List[Tuple[int, int, int, int]], w: int, h: int) -> None:
    ensure_dir(label_path.parent)
    with open(label_path, "w") as f:
        for x, y, bw, bh in bboxes:
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            wn = bw / w
            hn = bh / h
            f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")


def split_by_source(
    items: List[Tuple[Path, Path, str]], seed: int, split: Dict[str, float]
) -> Dict[str, List[Tuple[Path, Path]]]:
    rng = random.Random(seed)
    by_source: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
    for img, mask, source in items:
        by_source[source].append((img, mask))

    merged = {"train": [], "val": [], "test": []}
    for source, pairs in by_source.items():
        rng.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * split["train"])
        n_val = int(n * split["val"])
        merged["train"].extend(pairs[:n_train])
        merged["val"].extend(pairs[n_train : n_train + n_val])
        merged["test"].extend(pairs[n_train + n_val :])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-root", default="data/yolo_polyp")
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="0.8,0.1,0.1")
    parser.add_argument("--clean", action="store_true", help="Remove temporary _work directory after conversion")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    work_dir = out_root / "_work"
    ensure_dir(work_dir)

    split_vals = [float(v.strip()) for v in args.split.split(",")]
    split_cfg = {"train": split_vals[0], "val": split_vals[1], "test": split_vals[2]}

    items: List[Tuple[Path, Path, str]] = []
    for zip_path in sorted(data_dir.glob("*.zip")):
        name = zip_path.stem
        ds_root = prepare_dataset(zip_path, work_dir, name)
        for img, mask in image_pairs(ds_root):
            items.append((img, mask, name))

    splits = split_by_source(items, seed=args.seed, split=split_cfg)

    for split_name, pairs in splits.items():
        img_dir = out_root / "images" / split_name
        label_dir = out_root / "labels" / split_name
        ensure_dir(img_dir)
        ensure_dir(label_dir)
        for img, mask in pairs:
            dst_img = img_dir / img.name
            if not dst_img.exists():
                shutil.copy2(img, dst_img)
            image = cv2.imread(str(img))
            if image is None:
                continue
            h, w = image.shape[:2]
            bboxes = mask_to_bboxes(mask, min_area=args.min_area)
            label_path = label_dir / f"{img.stem}.txt"
            write_yolo_label(label_path, bboxes, w, h)

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "path: " + str(out_root) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: polyp\n"
    )

    stats = {k: len(v) for k, v in splits.items()}
    (out_root / "split_stats.json").write_text(json.dumps(stats, indent=2))

    if args.clean and work_dir.exists():
        shutil.rmtree(work_dir)

    print(f"done: {stats} out={out_root}")


if __name__ == "__main__":
    main()
