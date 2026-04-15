#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def _gray(rgb):
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _grad_mag(gray):
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def _pick_cell_points(gray, grad):
    h, w = gray.shape
    area = h * w
    target = int(np.clip(area // (160 * 160), 32, 96))
    stride = max(4, min(h, w) // 96)

    ys = np.arange(stride // 2, h, stride)
    xs = np.arange(stride // 2, w, stride)
    cand = []
    for y in ys:
        for x in xs:
            y0, y1 = max(0, y - 2), min(h, y + 3)
            x0, x1 = max(0, x - 2), min(w, x + 3)
            loc_g = gray[y0:y1, x0:x1].mean()
            loc_t = grad[y0:y1, x0:x1].mean()
            score = (255.0 - loc_g) + 0.35 * loc_t
            cand.append((score, int(x), int(y)))
    cand.sort(key=lambda t: t[0], reverse=True)
    return [(x, y) for _, x, y in cand[:target]]


def _tissue_grid(h, w):
    area = h * w
    target = int(np.clip(area // (256 * 256), 8, 36))
    aspect = w / max(h, 1)
    rows = max(1, int(round(math.sqrt(target / max(aspect, 1e-6)))))
    cols = max(1, int(math.ceil(target / rows)))
    y_edges = np.linspace(0, h, rows + 1, dtype=int)
    x_edges = np.linspace(0, w, cols + 1, dtype=int)

    regs = []
    rid = 1
    for ri in range(rows):
        for ci in range(cols):
            y0, y1 = int(y_edges[ri]), int(y_edges[ri + 1])
            x0, x1 = int(x_edges[ci]), int(x_edges[ci + 1])
            if y1 > y0 and x1 > x0:
                regs.append({
                    "id": rid,
                    "centroid": [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)],
                    "bbox": [x0, y0, x1, y1],
                    "region_type": 0,
                })
                rid += 1
    return regs


def _iter_images(bracs_root: Path, split: str, max_per_class: int):
    splits = ["train", "test"] if split == "all" else [split]
    rng = np.random.default_rng(42)
    for sp in splits:
        sp_dir = bracs_root / sp
        if not sp_dir.exists():
            continue
        for class_dir in sorted([d for d in sp_dir.iterdir() if d.is_dir()]):
            files = sorted(class_dir.glob("*.png"))
            files = files.copy()
            rng.shuffle(files)
            for p in files[:max_per_class]:
                yield sp, p


def main():
    parser = argparse.ArgumentParser(description="Generate heuristic entity annotations for BRACS")
    parser.add_argument("--bracs_root", type=Path, required=True)
    parser.add_argument("--annotations_root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--max_per_class", type=int, default=3)
    parser.add_argument("--cell_box_size", type=int, default=10)
    args = parser.parse_args()

    total = 0
    for sp, image_path in _iter_images(args.bracs_root, args.split, args.max_per_class):
        with Image.open(image_path) as img:
            rgb = np.asarray(img.convert("RGB"))
        h, w = rgb.shape[:2]
        gray = _gray(rgb)
        grad = _grad_mag(gray)

        half = max(2, args.cell_box_size // 2)
        cells = []
        for i, (x, y) in enumerate(_pick_cell_points(gray, grad), start=1):
            x0 = max(0, x - half)
            y0 = max(0, y - half)
            x1 = min(w - 1, x + half)
            y1 = min(h - 1, y + half)
            cells.append({
                "id": i,
                "centroid": [float(x), float(y)],
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "type": 0,
            })

        tissues = _tissue_grid(h, w)

        cell_out = args.annotations_root / "cells" / sp / f"{image_path.stem}.cells.json"
        tissue_out = args.annotations_root / "tissues" / sp / f"{image_path.stem}.tissues.json"
        cell_out.parent.mkdir(parents=True, exist_ok=True)
        tissue_out.parent.mkdir(parents=True, exist_ok=True)

        with open(cell_out, "w", encoding="utf-8") as f:
            json.dump({"cells": cells}, f, ensure_ascii=False, indent=2)
        with open(tissue_out, "w", encoding="utf-8") as f:
            json.dump({"tissues": tissues}, f, ensure_ascii=False, indent=2)
        total += 1

    print(f"[OK] generated entity annotations for {total} images -> {args.annotations_root}")


if __name__ == "__main__":
    main()
