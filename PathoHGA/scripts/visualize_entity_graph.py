#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def draw_edges(draw, pos, edge_index, color, width=1, max_edges=300):
    if edge_index is None or edge_index.numel() == 0:
        return
    edge_count = edge_index.shape[1]
    step = max(1, edge_count // max_edges)
    for i in range(0, edge_count, step):
        s = int(edge_index[0, i].item())
        t = int(edge_index[1, i].item())
        x1, y1 = float(pos[s, 0]), float(pos[s, 1])
        x2, y2 = float(pos[t, 0]), float(pos[t, 1])
        draw.line((x1, y1, x2, y2), fill=color, width=width)


def draw_nodes(draw, pos, color, r):
    for i in range(pos.shape[0]):
        x, y = float(pos[i, 0]), float(pos[i, 1])
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, fill=color)


def main():
    parser = argparse.ArgumentParser(description="Visualize entity graph on image")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = torch.load(args.graph, map_location="cpu", weights_only=False)
    img = Image.open(args.image).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    cell_pos = data["cell"].pos
    tissue_pos = data["tissue"].pos

    # edges first
    draw_edges(draw, cell_pos, data["cell", "knn", "cell"].edge_index, (80, 180, 255, 90), width=1)
    draw_edges(draw, tissue_pos, data["tissue", "rag", "tissue"].edge_index, (255, 170, 60, 120), width=2)

    # belongs edges (sampled heavily)
    belongs = data["cell", "belongs", "tissue"].edge_index
    if belongs.numel() > 0:
        step = max(1, belongs.shape[1] // 200)
        for i in range(0, belongs.shape[1], step):
            c = int(belongs[0, i].item())
            t = int(belongs[1, i].item())
            x1, y1 = float(cell_pos[c, 0]), float(cell_pos[c, 1])
            x2, y2 = float(tissue_pos[t, 0]), float(tissue_pos[t, 1])
            draw.line((x1, y1, x2, y2), fill=(120, 255, 120, 80), width=1)

    # nodes
    draw_nodes(draw, tissue_pos, (255, 80, 40, 220), r=4)
    draw_nodes(draw, cell_pos, (40, 220, 255, 180), r=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"[OK] saved {args.out}")


if __name__ == "__main__":
    main()
