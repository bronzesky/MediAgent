import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch_geometric.data import HeteroData


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


def _gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def _patch_features(rgb: np.ndarray, gray: np.ndarray, grad: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    h, w = gray.shape
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    patch_rgb = rgb[y0:y1, x0:x1].astype(np.float32)
    patch_gray = gray[y0:y1, x0:x1]
    patch_grad = grad[y0:y1, x0:x1]

    if patch_rgb.size == 0:
        return np.zeros(16, dtype=np.float32)

    rgb_mean = patch_rgb.reshape(-1, 3).mean(axis=0) / 255.0
    rgb_std = patch_rgb.reshape(-1, 3).std(axis=0) / 255.0
    gray_mean = np.array([patch_gray.mean() / 255.0], dtype=np.float32)
    gray_std = np.array([patch_gray.std() / 255.0], dtype=np.float32)
    gray_q = np.percentile(patch_gray, [10, 50, 90]).astype(np.float32) / 255.0
    grad_mean = np.array([patch_grad.mean() / 255.0], dtype=np.float32)
    grad_std = np.array([patch_grad.std() / 255.0], dtype=np.float32)
    grad_q = np.percentile(patch_grad, [50, 90]).astype(np.float32) / 255.0

    feats = np.concatenate([rgb_mean, rgb_std, gray_mean, gray_std, gray_q, grad_mean, grad_std, grad_q], axis=0)
    return feats.astype(np.float32)


def _to_dim(x: np.ndarray, out_dim: int) -> np.ndarray:
    in_dim = x.shape[1]
    if in_dim == out_dim:
        return x
    if in_dim > out_dim:
        return x[:, :out_dim]
    rep = math.ceil(out_dim / in_dim)
    tiled = np.tile(x, (1, rep))
    return tiled[:, :out_dim]


def _pick_cell_points(gray: np.ndarray, grad: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    area = h * w
    target = int(np.clip(area // (160 * 160), 32, 96))

    stride = max(4, min(h, w) // 96)
    ys = np.arange(stride // 2, h, stride)
    xs = np.arange(stride // 2, w, stride)

    candidates = []
    for y in ys:
        for x in xs:
            y0, y1 = max(0, y - 2), min(h, y + 3)
            x0, x1 = max(0, x - 2), min(w, x + 3)
            local_gray = gray[y0:y1, x0:x1].mean()
            local_grad = grad[y0:y1, x0:x1].mean()
            darkness = 255.0 - local_gray
            score = darkness + 0.35 * local_grad
            candidates.append((score, x, y))

    if not candidates:
        return np.array([[w // 2, h // 2]], dtype=np.int32)

    candidates.sort(key=lambda t: t[0], reverse=True)
    pts = np.array([[x, y] for _, x, y in candidates[:target]], dtype=np.int32)
    return pts


def _tissue_regions(h: int, w: int) -> list[tuple[int, int, int, int]]:
    area = h * w
    target = int(np.clip(area // (256 * 256), 8, 36))
    aspect = w / max(h, 1)

    rows = max(1, int(round(math.sqrt(target / max(aspect, 1e-6)))))
    cols = max(1, int(math.ceil(target / rows)))

    y_edges = np.linspace(0, h, rows + 1, dtype=int)
    x_edges = np.linspace(0, w, cols + 1, dtype=int)

    regions = []
    for ri in range(rows):
        for ci in range(cols):
            y0, y1 = y_edges[ri], y_edges[ri + 1]
            x0, x1 = x_edges[ci], x_edges[ci + 1]
            if y1 > y0 and x1 > x0:
                regions.append((x0, y0, x1, y1))
    return regions


def build_graph_from_image(image_path: Path, feature_dim: int = 64, k_cell: int = 5, k_tissue: int = 3) -> HeteroData:
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB"))

    h, w = rgb.shape[:2]
    gray = _gray(rgb)
    grad = _grad_mag(gray)

    cell_pts = _pick_cell_points(gray, grad)
    cell_radius = int(np.clip(min(h, w) // 64, 4, 16))

    cell_feats = np.stack([_patch_features(rgb, gray, grad, int(x), int(y), cell_radius) for x, y in cell_pts], axis=0)
    cell_feats = _to_dim(cell_feats, feature_dim)
    cell_xy = cell_pts.astype(np.float32)

    tissue_regions = _tissue_regions(h, w)
    tissue_xy = []
    tissue_feats = []
    for (x0, y0, x1, y1) in tissue_regions:
        reg_rgb = rgb[y0:y1, x0:x1].astype(np.float32)
        reg_gray = gray[y0:y1, x0:x1]
        reg_grad = grad[y0:y1, x0:x1]

        rgb_mean = reg_rgb.reshape(-1, 3).mean(axis=0) / 255.0
        rgb_std = reg_rgb.reshape(-1, 3).std(axis=0) / 255.0
        gray_stats = np.array([reg_gray.mean(), reg_gray.std()], dtype=np.float32) / 255.0
        grad_stats = np.array([reg_grad.mean(), reg_grad.std()], dtype=np.float32) / 255.0
        gray_q = np.percentile(reg_gray, [10, 50, 90]).astype(np.float32) / 255.0
        grad_q = np.percentile(reg_grad, [50, 90]).astype(np.float32) / 255.0

        feat = np.concatenate([rgb_mean, rgb_std, gray_stats, grad_stats, gray_q, grad_q], axis=0).astype(np.float32)
        tissue_feats.append(feat)
        tissue_xy.append([(x0 + x1) * 0.5, (y0 + y1) * 0.5])

    tissue_feats = _to_dim(np.stack(tissue_feats, axis=0), feature_dim)
    tissue_xy = np.asarray(tissue_xy, dtype=np.float32)

    cell_xy_t = torch.tensor(cell_xy, dtype=torch.float32)
    tissue_xy_t = torch.tensor(tissue_xy, dtype=torch.float32)

    cell_edges = _knn_edge_index(cell_xy_t, k_cell)
    tissue_edges = _knn_edge_index(tissue_xy_t, k_tissue)

    assign_dist = torch.cdist(cell_xy_t, tissue_xy_t)
    nearest_tissue = assign_dist.argmin(dim=1)
    belongs_edges = torch.stack([torch.arange(cell_xy_t.size(0)), nearest_tissue], dim=0)

    data = HeteroData()
    data["cell"].x = torch.tensor(cell_feats, dtype=torch.float32)
    data["cell"].pos = cell_xy_t
    data["tissue"].x = torch.tensor(tissue_feats, dtype=torch.float32)
    data["tissue"].pos = tissue_xy_t
    data["cell", "knn", "cell"].edge_index = cell_edges
    data["tissue", "rag", "tissue"].edge_index = tissue_edges
    data["cell", "belongs", "tissue"].edge_index = belongs_edges
    data.y = torch.tensor([_label_from_parent(image_path)], dtype=torch.long)
    data.sample_id = image_path.stem
    data.image_path = str(image_path)
    data.feature_source = "handcrafted-real-from-image"
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
    parser = argparse.ArgumentParser(description="Build HeteroData graphs from BRACS images using real handcrafted features")
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
    manifest = {"total_graphs": 0, "by_split": {}, "items": [], "feature_source": "handcrafted-real-from-image"}

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

    print(f"[OK] real-feature graphs: {manifest['total_graphs']} -> {args.out_dir}")


if __name__ == "__main__":
    main()
