#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


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


def _detect_cells(image_bgr: np.ndarray):
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

    min_area = max(16, (h * w) // 200000)
    max_area = max(400, (h * w) // 300)

    cells = []
    cid = 1
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if area < min_area or area > max_area:
            continue
        cx, cy = centroids[i]
        cells.append(
            {
                "id": cid,
                "centroid": [float(cx), float(cy)],
                "bbox": [int(x), int(y), int(x + ww - 1), int(y + hh - 1)],
                "type": 0,
            }
        )
        cid += 1

    # fallback if segmentation too sparse
    if len(cells) < 24:
        step = max(8, min(h, w) // 80)
        ys = np.arange(step // 2, h, step)
        xs = np.arange(step // 2, w, step)
        samples = []
        for yy in ys:
            for xx in xs:
                v = int(gray[yy, xx])
                samples.append((v, int(xx), int(yy)))
        samples.sort(key=lambda t: t[0])
        need = max(24, min(96, (h * w) // (180 * 180)))
        half = max(2, min(h, w) // 140)
        cells = []
        for idx, (_, xx, yy) in enumerate(samples[:need], start=1):
            x0, y0 = max(0, xx - half), max(0, yy - half)
            x1, y1 = min(w - 1, xx + half), min(h - 1, yy + half)
            cells.append(
                {
                    "id": idx,
                    "centroid": [float(xx), float(yy)],
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "type": 0,
                }
            )

    return cells


def _detect_tissues(image_bgr: np.ndarray):
    h, w = image_bgr.shape[:2]
    down = max(2, min(6, int(round(math.sqrt((h * w) / (1400 * 1400)))) + 2))
    small = cv2.resize(image_bgr, (w // down, h // down), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    feat = lab.reshape(-1, 3).astype(np.float32)

    k = int(np.clip((h * w) // (600 * 600), 6, 16))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    _, labels, _ = cv2.kmeans(feat, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    lbl_small = labels.reshape(lab.shape[:2]).astype(np.uint8)
    lbl = cv2.resize(lbl_small, (w, h), interpolation=cv2.INTER_NEAREST)

    min_area = max(200, (h * w) // 5000)
    tissues = []
    tid = 1
    for c in range(k):
        mask = (lbl == c).astype(np.uint8)
        n, cc, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            if area < min_area:
                continue
            cx, cy = cents[i]
            tissues.append(
                {
                    "id": tid,
                    "centroid": [float(cx), float(cy)],
                    "bbox": [int(x), int(y), int(x + ww - 1), int(y + hh - 1)],
                    "region_type": int(c),
                }
            )
            tid += 1

    if len(tissues) < 4:
        # fallback to coarse non-uniform blocks
        rows, cols = 2, 3
        y_edges = np.linspace(0, h, rows + 1, dtype=int)
        x_edges = np.linspace(0, w, cols + 1, dtype=int)
        tissues = []
        tid = 1
        for ri in range(rows):
            for ci in range(cols):
                y0, y1 = int(y_edges[ri]), int(y_edges[ri + 1])
                x0, x1 = int(x_edges[ci]), int(x_edges[ci + 1])
                tissues.append(
                    {
                        "id": tid,
                        "centroid": [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)],
                        "bbox": [x0, y0, x1 - 1, y1 - 1],
                        "region_type": 0,
                    }
                )
                tid += 1

    return tissues


def main():
    parser = argparse.ArgumentParser(description="Generate entity annotations for BRACS images")
    parser.add_argument("--bracs_root", type=Path, required=True)
    parser.add_argument("--annotations_root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--max_per_class", type=int, default=3)
    args = parser.parse_args()

    total = 0
    for sp, image_path in _iter_images(args.bracs_root, args.split, args.max_per_class):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        cells = _detect_cells(image_bgr)
        tissues = _detect_tissues(image_bgr)

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
