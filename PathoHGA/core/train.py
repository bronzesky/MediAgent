import argparse
import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from core.dataloader import GraphDataset
from core.models.pathoHGA import PathoHGAModel
from core.agent.graph_rag import build_index


def safe_commit_hash(cwd: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(cwd), text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def compute_weighted_f1(preds, labels, num_classes=7):
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for p, y in zip(preds, labels):
        conf[y, p] += 1
    support = conf.sum(dim=1).float()
    tp = conf.diag().float()
    fp = conf.sum(dim=0).float() - tp
    fn = conf.sum(dim=1).float() - tp
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    weighted = (f1 * support).sum() / support.sum().clamp_min(1.0)
    acc = tp.sum() / support.sum().clamp_min(1.0)
    return float(acc), float(weighted), conf


def run_epoch(model, dataset, optimizer, device, lambda_align=0.1, train=True, max_steps=0):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    preds, labels = [], []

    for i, data in enumerate(dataset):
        if max_steps > 0 and i >= max_steps:
            break
        data = data.to(device)
        out = model(data)
        ce = F.cross_entropy(out["logits"], data.y.view(-1))
        loss = ce + lambda_align * out["loss_align"]

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        preds.append(int(out["logits"].argmax(dim=1).item()))
        labels.append(int(data.y.item()))

    acc, wf1, conf = compute_weighted_f1(preds, labels)
    denom = max(1, len(preds))
    return {"loss": total_loss / denom, "acc": acc, "wf1": wf1, "conf": conf}


def main():
    parser = argparse.ArgumentParser(description="Smoke trainer for PathoHGA fullstack")
    parser.add_argument("--graph_root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_c1", action="store_true")
    parser.add_argument("--use_c2", action="store_true")
    parser.add_argument("--lambda_align", type=float, default=0.1)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_test_steps", type=int, default=0)
    parser.add_argument("--out_dir", type=Path, default=Path("results/smoke"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = GraphDataset(str(args.graph_root), "train")
    test_ds = GraphDataset(str(args.graph_root), "test")
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Empty train/test graph split. Please run graph_builder first.")

    model = PathoHGAModel(use_c1=args.use_c1, use_c2=args.use_c2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for ep in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_ds, optimizer, device, lambda_align=args.lambda_align, train=True, max_steps=args.max_train_steps)
        test_m = run_epoch(model, test_ds, optimizer, device, lambda_align=args.lambda_align, train=False, max_steps=args.max_test_steps)
        row = {"epoch": ep, "train_loss": train_m["loss"], "train_acc": train_m["acc"], "train_wf1": train_m["wf1"], "test_loss": test_m["loss"], "test_acc": test_m["acc"], "test_wf1": test_m["wf1"]}
        history.append(row)
        print(f"[Epoch {ep}] train_wf1={train_m['wf1']:.4f} test_wf1={test_m['wf1']:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "model.pt"
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)

    with open(args.out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    conf_path = args.out_dir / "confusion_matrix.pt"
    torch.save(history[-1], args.out_dir / "last_epoch.pt")

    index_path = args.out_dir / "graph_index.json"
    build_index(model, train_ds, device, index_path)

    repo_root = Path(__file__).resolve().parents[2]
    registry = repo_root / "results" / "registry.csv"
    registry.parent.mkdir(parents=True, exist_ok=True)
    write_header = not registry.exists()
    with open(registry, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "mode", "use_c1", "use_c2", "epochs", "train_n", "test_n", "test_acc", "test_wf1", "commit"])
        w.writerow([
            dt.datetime.now().isoformat(timespec="seconds"),
            "smoke",
            int(args.use_c1),
            int(args.use_c2),
            args.epochs,
            len(train_ds),
            len(test_ds),
            history[-1]["test_acc"],
            history[-1]["test_wf1"],
            safe_commit_hash(repo_root),
        ])

    print(f"[OK] checkpoint: {ckpt}")
    print(f"[OK] index: {index_path}")
    print(f"[OK] registry: {registry}")


if __name__ == "__main__":
    main()
