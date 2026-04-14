#!/usr/bin/env bash
set -euo pipefail
BASE=/media/share/HDD_16T_1/AIFFPE/MediAgent
REPO=$BASE/repos/mediagent-final-github
cd "$REPO"

echo "[worktrees]"
git worktree list

echo
echo "[branches ahead/behind]"
git fetch origin --prune >/dev/null 2>&1 || true
for b in main agent-main agent-graph agent-train agent-c1 agent-c2 agent-c3; do
  if git show-ref --verify --quiet "refs/heads/$b"; then
    up="origin/$b"
    if git show-ref --verify --quiet "refs/remotes/$up"; then
      read -r behind ahead < <(git rev-list --left-right --count "$b...$up")
      echo "$b  ahead:$ahead  behind:$behind"
    else
      echo "$b  (no remote branch yet)"
    fi
  fi
done
