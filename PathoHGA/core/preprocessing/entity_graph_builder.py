import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
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


def _gray(rgb):
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _grad_mag(gray):
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def _to_dim(x: np.ndarray, out_dim: int) -> np.ndarray:
    in_dim = x.shape[1]
    if in_dim == out_dim:
        return x
    if in_dim > out_dim:
        return x[:, :out_dim]
    rep = math.ceil(out_dim / in_dim)
    tiled = np.tile(x, (1, rep))
    return tiled[:, :out_dim]


def _pick_cell_points(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    grad = _grad_mag(gray)
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
            local_g = gray[y0:y1, x0:x1].mean()
            local_t = grad[y0:y1, x0:x1].mean()
            score = (255.0 - local_g) + 0.35 * local_t
            cand.append((score, int(x), int(y)))
    cand.sort(key=lambda t: t[0], reverse=True)
    pts = np.array([[x, y] for _, x, y in cand[:target]], dtype=np.int32)
    if pts.shape[0] == 0:
        pts = np.array([[w // 2, h // 2]], dtype=np.int32)
    return pts


def _tissue_grid(h: int, w: int):
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
                regs.append({"id": rid, "bbox": [x0, y0, x1, y1], "centroid": [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)], "region_type": 0})
                rid += 1
    return regs


def _heuristic_entities(rgb: np.ndarray):
    h, w = rgb.shape[:2]
    points = _pick_cell_points(_gray(rgb))
    half = max(2, min(h, w) // 128)
    cells = []
    for i, (x, y) in enumerate(points.tolist(), start=1):
        x0, y0 = max(0, x - half), max(0, y - half)
        x1, y1 = min(w - 1, x + half), min(h - 1, y + half)
        cells.append({"id": i, "bbox": [int(x0), int(y0), int(x1), int(y1)], "centroid": [float(x), float(y)], "type": 0})
    tissues = _tissue_grid(h, w)
    return cells, tissues


def _load_json_entities(path: Path, key: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    arr = data.get(key)
    if not isinstance(arr, list):
        raise ValueError(f"missing list key={key} in {path}")
    return arr


def _entity_mask(ent, h, w):
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    poly = ent.get("polygon")
    if poly and len(poly) >= 3:
        d.polygon([(float(p[0]), float(p[1])) for p in poly], fill=1)
    else:
        b = ent.get("bbox")
        if not b or len(b) != 4:
            raise ValueError("entity needs polygon or bbox")
        d.rectangle((float(b[0]), float(b[1]), float(b[2]), float(b[3])), fill=1)
    return np.asarray(m, dtype=bool)


def _entity_centroid(ent, mask):
    c = ent.get("centroid")
    if c and len(c) == 2:
        return float(c[0]), float(c[1])
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def _entity_feat(rgb, gray, grad, mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros(20, dtype=np.float32)
    p_rgb = rgb[ys, xs].astype(np.float32)
    p_gray = gray[ys, xs].astype(np.float32)
    p_grad = grad[ys, xs].astype(np.float32)
    rgb_mean = p_rgb.mean(axis=0) / 255.0
    rgb_std = p_rgb.std(axis=0) / 255.0
    gray_stats = np.array([p_gray.mean(), p_gray.std()], dtype=np.float32) / 255.0
    gray_q = np.percentile(p_gray, [10, 50, 90]).astype(np.float32) / 255.0
    grad_stats = np.array([p_grad.mean(), p_grad.std()], dtype=np.float32) / 255.0
    grad_q = np.percentile(p_grad, [50, 90]).astype(np.float32) / 255.0
    area = float(mask.sum())
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    hb = float(max(1, y1 - y0 + 1))
    wb = float(max(1, x1 - x0 + 1))
    shape = np.array([area / (mask.shape[0] * mask.shape[1]), wb / hb, area / (hb * wb)], dtype=np.float32)
    return np.concatenate([rgb_mean, rgb_std, gray_stats, gray_q, grad_stats, grad_q, shape], axis=0).astype(np.float32)


def _build_one(image_path: Path, annotations_root: Path, feature_dim: int):
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    gray = _gray(rgb)
    grad = _grad_mag(gray)

    rel = image_path.relative_to(image_path.parents[2])
    split = rel.parts[0]
    stem = image_path.stem

    cjson = annotations_root / "cells" / split / f"{stem}.cells.json"
    tjson = annotations_root / "tissues" / split / f"{stem}.tissues.json"

    if cjson.exists() and tjson.exists():
        cells = _load_json_entities(cjson, "cells")
        tissues = _load_json_entities(tjson, "tissues")
        source = "entity-from-annotation"
    else:
        cells, tissues = _heuristic_entities(rgb)
        source = "entity-heuristic-generated"

    if len(cells) == 0 or len(tissues) == 0:
        raise ValueError(f"empty entities for {image_path.name}")

    c_feat, c_xy = [], []
    t_feat, t_xy, t_masks = [], [], []

    for ent in cells:
        m = _entity_mask(ent, h, w)
        c_feat.append(_entity_feat(rgb, gray, grad, m))
        cx, cy = _entity_centroid(ent, m)
        c_xy.append([cx, cy])

    for ent in tissues:
        m = _entity_mask(ent, h, w)
        t_feat.append(_entity_feat(rgb, gray, grad, m))
        cx, cy = _entity_centroid(ent, m)
        t_xy.append([cx, cy])
        t_masks.append(m)

    c_x = _to_dim(np.stack(c_feat, axis=0), feature_dim)
    t_x = _to_dim(np.stack(t_feat, axis=0), feature_dim)
    c_xy_t = torch.tensor(np.asarray(c_xy, dtype=np.float32), dtype=torch.float32)
    t_xy_t = torch.tensor(np.asarray(t_xy, dtype=np.float32), dtype=torch.float32)

    c_edges = _knn_edge_index(c_xy_t, 5)
    t_edges = _knn_edge_index(t_xy_t, 3)

    src, dst = [], []
    for i, (cx, cy) in enumerate(c_xy):
        x = min(max(0, int(round(cx))), w - 1)
        y = min(max(0, int(round(cy))), h - 1)
        match = -1
        for j, tm in enumerate(t_masks):
            if tm[y, x]:
                match = j
                break
        if match < 0:
            d = torch.cdist(c_xy_t[i:i+1], t_xy_t).squeeze(0)
            match = int(torch.argmin(d).item())
        src.append(i)
        dst.append(match)

    belongs = torch.tensor([src, dst], dtype=torch.long)

    data = HeteroData()
    data["cell"].x = torch.tensor(c_x, dtype=torch.float32)
    data["cell"].pos = c_xy_t
    data["tissue"].x = torch.tensor(t_x, dtype=torch.float32)
    data["tissue"].pos = t_xy_t
    data["cell", "knn", "cell"].edge_index = c_edges
    data["tissue", "rag", "tissue"].edge_index = t_edges
    data["cell", "belongs", "tissue"].edge_index = belongs
    data["cell", "neighbors", "cell"].edge_index = c_edges
    data["tissue", "adjacent", "tissue"].edge_index = t_edges
    data["cell", "belongs_to", "tissue"].edge_index = belongs
    data.y = torch.tensor([_label_from_parent(image_path)], dtype=torch.long)
    data.sample_id = image_path.stem
    data.image_path = str(image_path)
    data.feature_source = source
    return data


def _collect_images(bracs_root: Path, split: str):
    splits = ["train", "test"] if split == "all" else [split]
    out = []
    for sp in splits:
        sp_dir = bracs_root / sp
        if not sp_dir.exists():
            continue
        for class_dir in sorted([d for d in sp_dir.iterdir() if d.is_dir()]):
            for p in sorted(class_dir.glob("*.png")):
                out.append((sp, class_dir.name, p))
    return out


def main():
    parser = argparse.ArgumentParser(description="Build entity graphs from BRACS")
    parser.add_argument("--bracs_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--annotations_root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--max_per_class", type=int, default=3)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    all_images = _collect_images(args.bracs_root, args.split)
    grouped = {}
    for sp, cls, p in all_images:
        grouped.setdefault((sp, cls), []).append(p)

    selected = []
    for key, files in sorted(grouped.items()):
        files = files.copy()
        rng.shuffle(files)
        selected.extend((key[0], key[1], f) for f in files[: args.max_per_class])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"total_graphs": 0, "failed_graphs": 0, "by_split": {}, "items": [], "graph_mode": "entity"}

    for sp, cls, image_path in selected:
        try:
            g = _build_one(image_path, args.annotations_root, args.feature_dim)
            out_path = args.out_dir / sp / cls / f"{image_path.stem}.pt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(g, out_path)
            manifest["total_graphs"] += 1
            manifest["by_split"][sp] = manifest["by_split"].get(sp, 0) + 1
            manifest["items"].append({"split": sp, "class": cls, "image": str(image_path), "graph": str(out_path), "feature_source": g.feature_source})
        except Exception as ex:
            manifest["failed_graphs"] += 1
            manifest["items"].append({"split": sp, "class": cls, "image": str(image_path), "error": str(ex)})

    with open(args.out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[OK] entity graphs={} failed={} -> {}".format(manifest.get("total_graphs"), manifest.get("failed_graphs"), args.out_dir))


if __name__ == "__main__":
    main()
