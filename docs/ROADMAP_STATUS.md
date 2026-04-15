# MediAgent 路线图状态（更新于 2026-04-15）

## 一、已完成（✅）

### A. 仓库与协作基建
- [x] 建立主仓库与多 worktree 协作结构（`main`/`agent-*`）。
- [x] 建立协作规范文档（`WORKTREE_COLLAB_GUIDE.md`）。
- [x] 建立今晚执行计划与 smoke 目标文档。

### B. 可跑完全体（Smoke 闭环）
- [x] BRACS 小样本图构建脚本可运行（`1_preprocess_bracs.sh`）。
- [x] Baseline 训练链路可运行（`2_train_baseline.sh`）。
- [x] C1+C2 训练链路可运行（`3_train_pathoHGA.sh`）。
- [x] C3 最小推理链路可运行（`4_eval_agent.sh`）。
- [x] 产出 smoke 结果、checkpoint、index、reasoning report、registry。

### C. 文档与接口
- [x] 冻结 Phase1 I/O schema（`PHASE1_IO_SCHEMA.md`）。
- [x] 生成并维护 smoke runbook（`runbook_smoke_fullstack.md`）。

---

## 二、当前状态结论
- 当前阶段：**已完成“工程可跑原型”**。
- 当前能力：可在 BRACS 小样本上端到端跑通全链路。
- 当前限制：尚未进入“真实特征 + 正式实验 + 投稿证据”阶段。

---

## 三、下一阶段任务（按优先级）

### P0（必须先做）
- [x] 将 `graph_builder` 从 smoke 特征升级到真实病理特征链路（细胞/组织真实特征）。
- [ ] 增加统一配置系统（`configs/bracs_baseline.yml`, `configs/bracs_pathohga.yml`）。
- [ ] 统一 train/eval 参数入口，支持 seed、batch、设备、输出目录。
- [ ] 固化数据与运行 manifest（数据版本、代码版本、配置版本三元绑定）。

### P1（核心研究模块）
- [ ] C1 超图语义模块升级为论文版（可学习超边+稳定训练约束）。
- [ ] C2 对齐模块升级为论文版（PathoGraph 锚点对齐 + loss warmup）。
- [ ] C3 推理模块升级为可验证版本（硬约束验证器+回退机制+证据追溯）。

### P2（实验与证据）
- [ ] BRACS 正式训练（非 smoke），输出主指标与混淆矩阵。
- [ ] 消融矩阵（Baseline/C1/C1+C2/C1+C2+C3）。
- [ ] 多 seed 重复与方差报告。
- [ ] 外部验证集（如 TCGA-BRCA）接入与评估。

### P3（投稿准备）
- [ ] 主表主图定稿（性能、消融、可解释、案例分析）。
- [ ] 误差分析与失败案例归纳。
- [ ] 方法、实验、附录可复现材料整理。

---

## 四、本周可执行里程碑（建议）
- [x] 里程碑 M1：真实 `graph_builder` 可跑并替换 smoke 版本。
- [ ] 里程碑 M2：Baseline 正式训练跑通并有稳定日志。
- [ ] 里程碑 M3：C1+C2 论文版最小可训闭环完成。
- [ ] 里程碑 M4：C3 可验证推理闭环完成并可输出证据链。

---

## 五、给 Claude / Codex 的分工建议（下一阶段）
- Claude（agent-graph）：真实构图链路、数据 manifest、预处理性能优化。
- Codex（agent-train/c1/c2/c3）：训练框架、C1/C2/C3 升级、端到端集成与验证。
- agent-main：冲突处理、合并门禁、发布前回归。
