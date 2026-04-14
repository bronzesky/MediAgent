import argparse
import json
from pathlib import Path

import torch

from core.dataloader import GraphDataset
from core.models.pathoHGA import PathoHGAModel
from core.agent.graph_rag import retrieve_topk


def majority_label(rows):
    votes = {}
    for r in rows:
        votes[r["label"]] = votes.get(r["label"], 0) + 1
    return max(votes.items(), key=lambda x: x[1])[0] if votes else -1


def main():
    parser = argparse.ArgumentParser(description="Minimal graph-RAG reasoning smoke")
    parser.add_argument("--graph_root", type=Path, required=True)
    parser.add_argument("--index_json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--case_idx", type=int, default=0)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--use_c1", action="store_true")
    parser.add_argument("--use_c2", action="store_true")
    parser.add_argument("--out_json", type=Path, default=Path("results/smoke/reasoning_report.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PathoHGAModel(use_c1=args.use_c1, use_c2=args.use_c2)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    ds = GraphDataset(str(args.graph_root), args.split)
    data = ds[args.case_idx].to(device)
    with torch.no_grad():
        out = model(data)
        logits = out["logits"]
        pred = int(logits.argmax(dim=1).item())
        query_emb = out["graph_emb"].squeeze(0).cpu()

    with open(args.index_json, "r", encoding="utf-8") as f:
        index_rows = json.load(f)

    topk = retrieve_topk(query_emb, index_rows, k=args.topk)
    majority = majority_label(topk)
    constraint_passed = pred == majority
    final_pred = pred if constraint_passed else majority

    report = {
        "sample_id": getattr(data, "sample_id", "unknown"),
        "raw_pred": pred,
        "majority_retrieved_label": majority,
        "constraint_passed": constraint_passed,
        "final_pred": final_pred,
        "evidence": topk,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[OK] reasoning report:", args.out_json)


if __name__ == "__main__":
    main()
