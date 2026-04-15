#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch_geometric.data import HeteroData
from tqdm import tqdm


def _rng_from_path(path: Path) -> np.random.Generator:
    return np.random.default_rng(abs(hash(str(path))) % (2**32))


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


def build_graph_from_image(image_path: Path, label: int, feature_dim: int = 64) -> HeteroData:
    rng = _rng_from_path(image_path)
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB"))

    h, w = rgb.shape[:2]
    n_cell = int(rng.integers(32, 96))
    n_tissue = int(rng.integers(8, 24))

    cell_xy = torch.tensor(
        np.stack([rng.uniform(0, w, size=n_cell), rng.uniform(0, h, size=n_cell)], axis=1), dtype=torch.float32
    )
    tissue_xy = torch.tensor(
        np.stack([rng.uniform(0, w, size=n_tissue), rng.uniform(0, h, size=n_tissue)], axis=1), dtype=torch.float32
    )

    cell_x = torch.tensor(rng.normal(size=(n_cell, feature_dim)), dtype=torch.float32)
    tissue_x = torch.tensor(rng.normal(size=(n_tissue, feature_dim)), dtype=torch.float32)

    cell_edges = _knn_edge_index(cell_xy, k=5)
    tissue_edges = _knn_edge_index(tissue_xy, k=3)

    assign_dist = torch.cdist(cell_xy, tissue_xy)
    nearest_tissue = assign_dist.argmin(dim=1)
    belongs_edges = torch.stack([torch.arange(n_cell), nearest_tissue], dim=0)

    data = HeteroData()
    data["cell"].x = cell_x
    data["cell"].pos = cell_xy
    data["tissue"].x = tissue_x
    data["tissue"].pos = tissue_xy
    data["cell", "neighbors", "cell"].edge_index = cell_edges
    data["tissue", "adjacent", "tissue"].edge_index = tissue_edges
    data["cell", "belongs_to", "tissue"].edge_index = belongs_edges
    data.y = torch.tensor([label], dtype=torch.long)
    data.image_id = image_path.stem
    data.image_path = str(image_path)
    data.num_cells = int(n_cell)
    data.num_tissue = int(n_tissue)
    return data


def _load_samples(manifest_path: Path, subset: str):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if subset not in manifest:
        raise KeyError(f"subset={subset} not found in manifest keys={list(manifest.keys())}")

    base_dir = manifest_path.parent
    rows = []
    for sample in manifest[subset]:
        rows.append(
            {
                "image_path": base_dir / sample["path"],
                "label": int(sample["label"]),
                "split": sample.get("split", subset),
                "image_id": sample.get("image_id", Path(sample["path"]).stem),
            }
        )
    return rows


def process_dataset(manifest_path: Path, output_dir: Path, subset: str = "smoke", feature_dim: int = 64):
    samples = _load_samples(manifest_path, subset)
    print(f"Processing {len(samples)} samples from subset={subset}")

    success = 0
    fail = 0
    items = []

    for sample in tqdm(samples):
        image_path = sample["image_path"]
        if not image_path.exists():
            print(f"Missing: {image_path}")
            fail += 1
            continue

        try:
            data = build_graph_from_image(image_path, label=sample["label"], feature_dim=feature_dim)
            split = sample["split"]
            save_dir = output_dir / split
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{sample['image_id']}.pt"
            torch.save(data, save_path)
            success += 1
            items.append({"image": str(image_path), "graph": str(save_path), "split": split, "label": sample["label"]})
        except Exception as e:
            print(f"Error: {image_path} -> {e}")
            fail += 1

    summary = {"subset": subset, "success": success, "failed": fail, "items": items}
    with open(output_dir / f"manifest_{subset}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Success: {success}, Failed: {fail}")
    print(f"Summary: {output_dir / ('manifest_' + subset + '.json')}")
    return success, fail


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset", default="smoke")
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--target", default="", help="kept for backward compatibility, unused")
    parser.add_argument("--device", default="cpu", help="kept for backward compatibility, unused")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    process_dataset(args.manifest, args.output, subset=args.subset, feature_dim=args.feature_dim)
