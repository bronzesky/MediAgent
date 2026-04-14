# Worktree 协同规范（MediAgent）

## 目录约定
- 主仓库：`/media/share/HDD_16T_1/AIFFPE/MediAgent/repos/mediagent-final-github`
- 多工作树：`/media/share/HDD_16T_1/AIFFPE/MediAgent/worktrees`

已创建：
- `agent-main`（总控/集成）
- `agent-graph`（Phase1 构图）
- `agent-train`（Phase1 训练）
- `agent-c1`（超图模块）
- `agent-c2`（对齐模块）
- `agent-c3`（Agent 推理）

## 基本规则
1. 每个 agent 只在自己的 worktree 开发。
2. 每个 agent 只改自己负责的文件域，避免交叉改动。
3. 合入顺序：`agent-main` 先 pull，再逐分支 cherry-pick 或 merge。
4. 每次提交前必须记录实验条目到 `results/registry.csv`（config/seed/metrics/commit）。
5. 禁止在未同步 `main` 的情况下长时间漂移开发。

## 日常命令
在任一 worktree 中：
```bash
git status
git add -A
git commit -m "feat: ..."
git push -u origin <branch>
```

在总控 worktree（`agent-main`）集成：
```bash
git checkout agent-main
git pull --rebase origin main
# 方式1：merge
# git merge --no-ff origin/agent-graph
# 方式2：cherry-pick
# git cherry-pick <commit_sha>

git push origin agent-main
```

发布到主分支：
```bash
git checkout main
git pull --rebase origin main
git merge --no-ff agent-main
git push origin main
```

## 文件责任建议
- `agent-graph`：`PathoHGA/core/preprocessing/*`, `PathoHGA/scripts/1_*`
- `agent-train`：`PathoHGA/core/train.py`, `PathoHGA/core/dataloader.py`, `PathoHGA/scripts/2_*`
- `agent-c1`：`PathoHGA/core/models/hypergraph.py`
- `agent-c2`：`PathoHGA/core/models/alignment.py`
- `agent-c3`：`PathoHGA/core/agent/*`, `PathoHGA/scripts/4_*`
- `agent-main`：配置整合、冲突解决、release notes

## 冲突处理
1. 冲突优先由 `agent-main` 处理。
2. 若数据结构冲突（字段名/shape），以已冻结 I/O schema 为准。
3. 任何 breaking change 先在 `Execution_Plan_v2.md` 记录，再修改代码。
