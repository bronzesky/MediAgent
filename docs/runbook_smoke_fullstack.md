# Runbook: Smoke Fullstack (BRACS tiny subset)

## Objective
Build a runnable complete pipeline only:
preprocess -> baseline train -> C1+C2 train -> C3 reasoning.

## Commands
From worktree root `/media/share/HDD_16T_1/AIFFPE/MediAgent/worktrees/agent-main`:

1. Preprocess tiny subset
```bash
./PathoHGA/scripts/1_preprocess_bracs.sh
```

2. Baseline smoke train
```bash
./PathoHGA/scripts/2_train_baseline.sh
```

3. PathoHGA (C1+C2) smoke train
```bash
./PathoHGA/scripts/3_train_pathoHGA.sh
```

4. Agent reasoning smoke
```bash
./PathoHGA/scripts/4_eval_agent.sh
```

## Current Status (2026-04-15)
- PASS: Graph build (`33` graphs)
- PASS: Baseline 1 epoch smoke
- PASS: C1+C2 1 epoch smoke
- PASS: C3 minimal reasoning output

## Artifacts
- `PathoHGA/data/smoke_bracs/manifest.json`
- `PathoHGA/results/smoke/baseline/*`
- `PathoHGA/results/smoke/pathohga/*`
- `PathoHGA/results/smoke/reasoning_report.json`
- `results/registry.csv`
