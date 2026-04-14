# Tonight Plan: Runnable Fullstack Only (No Full Experiments)

> Date: 2026-04-15
> Scope reset: build a **runnable complete pipeline** only.
> We do **not** target ablation, significance tests, or publication-grade metrics tonight.

## 1. Definition of Done (Tonight)

A run is considered successful if all items below pass on a tiny BRACS subset:

1. Input BRACS images (few samples) can be discovered and loaded.
2. Preprocess pipeline produces valid graph `.pt` files.
3. Training entry launches and completes at least 1 epoch without crash.
4. C1 and C2 modules are connected in forward pass (can be toggled on/off).
5. C3 reasoning path runs once with graph retrieval + constraint check.
6. End-to-end command exits code 0 and writes logs/results artifacts.

## 2. Minimal Data Scope

Use only a tiny subset for smoke validation:
- BRACS: 2–5 images per class (or total 20–40 images if class split is hard)
- Goal: verify pipeline stability, not model quality.

Output should be written to isolated smoke directories:
- `data/smoke_bracs/`
- `logs/smoke/`
- `results/smoke/`

## 3. Execution Order (Hard Sequence)

1. Freeze I/O schema (`HeteroData` fields + label keys).
2. Run graph builder on tiny subset and validate file readability.
3. Run baseline train 1 epoch (smoke).
4. Enable C1, rerun 1 epoch smoke.
5. Enable C1+C2, rerun 1 epoch smoke.
6. Run C3 minimal inference for 1–3 cases.

If any stage fails, fix only blocker-level issues and continue.
No performance tuning loops tonight.

## 4. Multi-Agent Ownership

- Claude (`agent-graph`): preprocessing + graph builder + subset indexing.
- Codex (`agent-train`): dataloader/train path + C1/C2 switches + smoke logging.
- Codex (`agent-main`): merge integration + C3 minimal run orchestration.

## 5. Mandatory Artifacts

- `runbook_smoke_fullstack.md`
- `results/registry.csv` (smoke rows with commit/config)
- smoke log files under `logs/smoke/`
- one final checklist with PASS/FAIL per stage

## 6. Non-Goals (Explicit)

- No 3-seed evaluation
- No full BRACS training
- No statistical significance
- No paper table completion tonight

