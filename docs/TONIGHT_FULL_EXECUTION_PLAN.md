# Tonight Full Execution Plan (Phase 1 Full Push)

> Date: 2026-04-15 (Asia/Shanghai)
> Goal: If execution efficiency is high, finish **all Phase 1** tonight.
> Collaboration: Claude + Codex parallel development via git worktree.

## 1) Tonight Success Criteria (Definition of Done)

We consider "Phase 1 done" only if all items below are complete:

1. BRACS manifest finalized (count, class distribution, corrupted-file check).
2. Phase1 I/O schema frozen and documented (HeteroData fields + label/split keys).
3. `graph_builder.py` minimum runnable pipeline complete.
4. Preprocess script runs end-to-end on BRACS subset and full split.
5. Baseline training smoke test (1 epoch) successful.
6. Baseline 3-seed short run completed (or at least 2 seed if time-limited, with explicit note).
7. Outputs archived:
   - `results/registry.csv`
   - `runbook_phase1.md`
   - baseline confusion matrix + metric table.

---

## 2) Worktree + Branch Assignment (Hard Ownership)

- `agent-graph` (Claude)
  - Owns: `PathoHGA/core/preprocessing/*`, `PathoHGA/scripts/1_*`
  - Deliverables: graph builder + preprocess CLI + schema-compliant `.pt` output.

- `agent-train` (Codex)
  - Owns: `PathoHGA/core/dataloader.py`, `PathoHGA/core/train.py`, `PathoHGA/scripts/2_*`
  - Deliverables: smoke train + short baseline runs + metric logging.

- `agent-main` (Codex integration)
  - Owns: merge integration, conflict resolution, final nightly checkpoint.

No cross-ownership edits without explicit note in commit message.

---

## 3) Timeline (Targeting Full Completion Tonight)

### Block A (now → +60 min)
- Freeze schema and manifest.
- Confirm BRACS download status and missing files.
- Produce a stable sample list for preprocess/training.

### Block B (+60 min → +180 min)
- Claude: finish `graph_builder.py` minimum runnable path.
- Codex: finish dataloader/training compatibility and run on small subset.

### Block C (+180 min → +300 min)
- Integrate to `agent-main`.
- Run 1 epoch smoke train on full pipeline.
- Fix only blocking errors (no optimization detours).

### Block D (+300 min → end)
- Run baseline short multi-seed.
- Save metrics/plots/logs.
- Finalize Phase1 runbook and update execution board.

---

## 4) 03:45 Usage-Refresh Strategy

If model quota refresh is needed around ~03:45:

1. Before refresh, write a **handoff block** in `runbook_phase1.md`:
   - done / in-progress / blocked / next exact command.
2. Keep all work committed to branch (`agent-graph` or `agent-train`).
3. After restart, resume from handoff block only (no rework).

Handoff template:

```text
[HANDOFF @ timestamp]
Done:
In Progress:
Blocked:
Next Command:
Owner:
```

---

## 5) Command Checklist (Operational)

### For Claude (`agent-graph`)
```bash
cd /media/share/HDD_16T_1/AIFFPE/MediAgent/worktrees/agent-graph
git pull --rebase
# implement graph_builder + preprocess script
git add -A && git commit -m "feat(graph): phase1 graph builder minimum runnable"
git push
```

### For Codex (`agent-train`)
```bash
cd /media/share/HDD_16T_1/AIFFPE/MediAgent/worktrees/agent-train
git pull --rebase
# implement dataloader/train smoke path + registry logging
git add -A && git commit -m "feat(train): phase1 smoke train and registry"
git push
```

### Integration (`agent-main`)
```bash
cd /media/share/HDD_16T_1/AIFFPE/MediAgent/worktrees/agent-main
git pull --rebase
git merge --no-ff origin/agent-graph
git merge --no-ff origin/agent-train
# run end-to-end validation
git push
```

### Publish to `main`
```bash
cd /media/share/HDD_16T_1/AIFFPE/MediAgent/repos/mediagent-final-github
git checkout main
git pull --rebase origin main
git merge --no-ff agent-main
git push origin main
```

---

## 6) Merge/Quality Gates (Mandatory)

Before merge to `main`, all checks must pass:

1. `.pt` graph files load successfully via dataloader.
2. 1 epoch smoke train exits with code 0.
3. Metrics/log output file exists and is parseable.
4. `results/registry.csv` has new row(s) for tonight runs.
5. No edits to data/raw/model-weight directories in git diff.

---

## 7) Fast-Fail Policy (Avoid Wasting Night Window)

If a blocker exceeds 30 min:
1. Downgrade to minimal fallback (smaller subset / simpler feature path).
2. Keep interface stable; defer optimization to tomorrow.
3. Log blocker and workaround in runbook immediately.

---

## 8) Expected Nightly Outputs

By end of night, expected artifacts:

- Code:
  - `PathoHGA/core/preprocessing/graph_builder.py`
  - `PathoHGA/core/dataloader.py`
  - `PathoHGA/core/train.py` (smoke-path validated)
  - `PathoHGA/scripts/1_preprocess_bracs.sh`
  - `PathoHGA/scripts/2_train_baseline.sh`

- Docs:
  - `runbook_phase1.md`
  - updated `Execution_Plan_v2.md` progress board
  - updated `PathoHGA/IMPLEMENTATION_PLAN.md` progress board

- Results:
  - `results/registry.csv`
  - baseline metric table + confusion matrix

If all achieved, we mark **Phase 1 complete** and start C1 on next cycle.
