import json
from pathlib import Path

import torch


@torch.no_grad()
def build_index(model, dataset, device, out_path: Path):
    model.eval()
    rows = []
    for data in dataset:
        data = data.to(device)
        out = model(data)
        rows.append(
            {
                "sample_id": getattr(data, "sample_id", "unknown"),
                "label": int(data.y.item()),
                "graph_path": getattr(data, "graph_path", ""),
                "emb": out["graph_emb"].squeeze(0).cpu().tolist(),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return rows


def retrieve_topk(query_emb: torch.Tensor, index_rows, k=3):
    q = query_emb / query_emb.norm(p=2).clamp_min(1e-8)
    scored = []
    for row in index_rows:
        emb = torch.tensor(row["emb"], dtype=torch.float32)
        emb = emb / emb.norm(p=2).clamp_min(1e-8)
        score = torch.dot(q, emb).item()
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, **r} for s, r in scored[:k]]
