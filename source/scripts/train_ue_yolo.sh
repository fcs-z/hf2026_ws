#!/usr/bin/env bash
# 将官方 PNG+JSON 数据转换后，微调目标车/诱饵车二分类 YOLO 检测器。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DATASET="${1:-$ROOT_DIR/dataset}"
EPOCHS="${2:-80}"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
PREPARED_DATASET="${HF2026_YOLO_PREPARED_DATASET:-$ROOT_DIR/.cache/hf2026/official-yolo-binary-seed2026}"

if [ -n "${HF2026_YOLO_BASE_MODEL:-}" ]; then
    BASE_MODEL="$HF2026_YOLO_BASE_MODEL"
elif [ -s "$ROOT_DIR/models/hf2026_target_vehicle_domain_v1.pt" ]; then
    BASE_MODEL="$ROOT_DIR/models/hf2026_target_vehicle_domain_v1.pt"
else
    BASE_MODEL="$ROOT_DIR/examples/yolotrack/target_vehicle_yolov8s.pt"
fi

OUTPUT_MODEL="${HF2026_YOLO_OUTPUT_MODEL:-$ROOT_DIR/models/hf2026_vehicle_binary_v2.pt}"
RUN_NAME="${HF2026_YOLO_RUN_NAME:-official-vehicle-binary-v2}"
IMGSZ="${HF2026_YOLO_IMGSZ:-1024}"
BATCH="${HF2026_YOLO_BATCH:-8}"
WORKERS="${HF2026_YOLO_WORKERS:-4}"
SEED="${HF2026_YOLO_SEED:-2026}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "错误: 视觉环境不存在；先执行 uv sync --extra vision --locked" >&2
    exit 2
fi
if [ ! -s "$BASE_MODEL" ]; then
    echo "错误: 基础权重不存在: $BASE_MODEL（仓库权重缺失时执行 git lfs pull）" >&2
    exit 2
fi

# 既支持已经是 YOLO 格式的目录，也支持官方的 target_*/decoy_* PNG+JSON 原始目录。
if [ -s "$INPUT_DATASET/dataset.yaml" ]; then
    DATASET="$INPUT_DATASET"
elif find "$INPUT_DATASET" -mindepth 2 -maxdepth 2 -name '*.json' -print -quit 2>/dev/null | grep -q .; then
    DATASET="$PREPARED_DATASET"
    echo "发现官方原始数据，先生成无视角泄漏的 YOLO train/val/test 划分。"
    "$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_official_yolo_dataset.py" \
        "$INPUT_DATASET" "$DATASET" --seed "$SEED"
else
    echo "错误: $INPUT_DATASET 既没有 dataset.yaml，也没有官方 PNG+JSON 标注" >&2
    exit 2
fi

DEVICE="${HF2026_YOLO_DEVICE:-$($PYTHON_BIN -c 'import torch; print(0 if torch.cuda.is_available() else "cpu")')}"
if [ "$DEVICE" = "cpu" ]; then
    echo "警告: CUDA 不可用，将用 CPU 训练；1024 分辨率训练会非常慢。" >&2
fi

mkdir -p "$ROOT_DIR/models" "$ROOT_DIR/.cache/hf2026/yolo-runs"
cd "$ROOT_DIR"
echo "训练数据: $DATASET/dataset.yaml"
echo "基础权重: $BASE_MODEL"
echo "参数: epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH workers=$WORKERS device=$DEVICE seed=$SEED"

"$PYTHON_BIN" "$ROOT_DIR/scripts/train_official_yolo.py" \
    --data "$DATASET/dataset.yaml" \
    --base-model "$BASE_MODEL" \
    --output "$OUTPUT_MODEL" \
    --project "$ROOT_DIR/.cache/hf2026/yolo-runs" \
    --run-name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --seed "$SEED"
