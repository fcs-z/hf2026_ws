# HF2026 快速开始

本页仅列出赛题一和赛题二的安装、检查及评测命令。完整算法说明请阅读
[README.md](README.md)。

## 1 准备官方平台

如果已经有完整的 `hf2026-sim`，可直接进入下一步。新环境执行：

```bash
git clone https://osredm.com/hf2026/hf2026-sim.git
cd hf2026-sim
git lfs pull
```

## 2 离线安装并检查

```bash
cd /path/to/hf2026_ws

chmod +x install_offline.sh check_submission.sh run_task1.sh run_task2.sh
./install_offline.sh /path/to/hf2026-sim
./check_submission.sh /path/to/hf2026-sim
```

安装脚本会使用项目自带的 uv 和 wheelhouse 创建或更新 `hf2026-sim/.venv`，全程不访问
网络。同名算法文件会先备份到 `hf2026-sim/.hf2026_submission_backup/`。

## 3 正式 UE 评测

```bash
cd /path/to/hf2026_ws

# 赛题一，随机种子为 1
./run_task1.sh /path/to/hf2026-sim 1 eval

# 赛题二，随机种子为 1
./run_task2.sh /path/to/hf2026-sim 1 eval
```

正式 UE 评测固定运行 600 秒。评测结束后，终端会打印本次 `evaluation.json` 路径。

## 4 无 UE 快速回归

```bash
./run_task1.sh /path/to/hf2026-sim 1 train 600
./run_task2.sh /path/to/hf2026-sim 1 train 600
```

`train` 模式用于快速检查控制和协同逻辑，不代表正式视觉评测成绩。

## 5 可选重新训练

官方数据集应位于 `hf2026-sim/dataset/`：

```bash
cd /path/to/hf2026-sim
./scripts/train_ue_yolo.sh dataset 80
```

已有模型权重随提交包提供，正常复现评测不需要重新训练。

## 6 停止平台

评测脚本结束后平台仍会运行。不再使用时执行：

```bash
cd /path/to/hf2026-sim
./stop.sh
```
