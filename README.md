# MediAgent Final GitHub Repo (Staging)

这个目录用于整理最终上传 GitHub 的主仓库内容。

## 当前结构
- `PathoHGA/`：核心代码与训练/推理脚本（从主工作区同步）
- `docs/`：投稿与复现实验所需核心文档
- `configs/`：预留配置目录
- `scripts/`：预留发布脚本目录

## 使用方式
1. 在主工作区开发（`/media/share/HDD_16T_1/AIFFPE/MediAgent`）
2. 执行 `../scripts/sync_release_repo.sh` 同步到本目录
3. 在本目录完成发布前清理（去除私有数据、权重、临时日志）
4. 再推送到 GitHub
