import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch_geometric.data import HeteroData


def _rng_from_path(path: Path) -> np.random.Generator:
    return np.random.default_rng(abs(hash(str(path))) % (2**32))


def _label_from_parent(path: Path) -> int:
    name = path.parent.name
    return int(name.split("_")[0]) if "_" in name else int(name)


def _knn_edge_index(coords: torch.Tensor, k: int) -> torch.Tensor:
    n = coords.size(0)
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    k = min(k, max(1, n - 1))
    dist = torch.cdist(coords, coords)
    nn = dist.topk(k=k + 1, largest=False).indices[:, 1:]
    src = torch.arange(n).unsqueeze(1).repeat(1, k).reshape(-1)
    dst = nn.reshape(-1)
    return torch.stack([src, dst], dim=0)


def build_graph_from_image(image_path: Path, feature_dim: int = 64, k_cell: int = 5, k_tissue: int = 3) -> HeteroData:
    rng = _rng_from_path(image_path)
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB"))

    height, width = rgb.shape[:2]
    n_cell = int(rng.integers(32, 96))
    n_tissue = int(rng.integers(8, 24))

    cell_xy = torch.tensor(
        np.stack([rng.uniform(0, width, size=n_cell), rng.uniform(0, height, size=n_cell)], axis=1),
        dtype=torch.float32,
    )
    tissue_xy = torch.tensor(
        np.stack([rng.uniform(0, width, size=n_tissue), rng.uniform(0, height, size=n_tissue)], axis=1),
        dtype=torch.float32,
    )

    cell_x = torch.tensor(rng.normal(size=(n_cell, feature_dim)), dtype=torch.float32)
    tissue_x = torch.tensor(rng.normal(size=(n_tissue, feature_dim)), dtype=torch.float32)

    cell_edges = _knn_edge_index(cell_xy, k_cell)
    tissue_edges = _knn_edge_index(tissue_xy, k_tissue)

    assign_dist = torch.cdist(cell_xy, tissue_xy)
    nearest_tissue = assign_dist.argmin(dim=1)
    belongs_edges = torch.stack([torch.arange(n_cell), nearest_tissue], dim=0)

    data = HeteroData()
    data["cell"].x = cell_x
    data["cell"].pos = cell_xy
    data["tissue"].x = tissue_x
    data["tissue"].pos = tissue_xy
    data["cell", "knn", "cell"].edge_index = cell_edges
    data["tissue", "rag", "tissue"].edge_index = tissue_edges
    data["cell", "belongs", "tissue"].edge_index = belongs_edges
    data.y = torch.tensor([_label_from_parent(image_path)], dtype=torch.long)
    data.sample_id = image_path.stem
    data.image_path = str(image_path)
    return data


def collect_images(bracs_root: Path, split: str):
    splits = ["train", "test"] if split == "all" else [split]
    out = []
    for sp in splits:
        split_dir = bracs_root / sp
        if not split_dir.exists():
            continue
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        for class_dir in class_dirs:
            for p in sorted(class_dir.glob("*.png")):
                out.append((sp, class_dir.name, p))
    return out


def main():
    parser = argparse.ArgumentParser(description="Build smoke HeteroData graphs from BRACS images")
    parser.add_argument("--bracs_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--max_per_class", type=int, default=3)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    images = collect_images(args.bracs_root, args.split)

    grouped = {}
    for sp, cls, p in images:
        grouped.setdefault((sp, cls), []).append(p)

    selected = []
    for key, files in sorted(grouped.items()):
        files = files.copy()
        rng.shuffle(files)
        selected.extend((key[0], key[1], f) for f in files[: args.max_per_class])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"total_graphs": 0, "by_split": {}, "items": []}

    for sp, cls, image_path in selected:
        graph = build_graph_from_image(image_path, feature_dim=args.feature_dim)
        out_path = args.out_dir / sp / cls / f"{image_path.stem}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, out_path)

        manifest["total_graphs"] += 1
        manifest["by_split"][sp] = manifest["by_split"].get(sp, 0) + 1
        manifest["items"].append({"split": sp, "class": cls, "image": str(image_path), "graph": str(out_path)})

    with open(args.out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] graphs: {manifest['total_graphs']} -> {args.out_dir}")


if __name__ == "__main__":
    main()
