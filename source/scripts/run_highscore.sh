#!/usr/bin/env bash
# 一键运行赛题一或赛题二 Agent；不会改写官方 scenario.json。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:-}"
DURATION="${2:-600}"
SEED="${3:-1}"
MODE="${4:-train}"
OUTPUT_DIR="${5:-$ROOT_DIR/benchmark_results/${TASK}-seed${SEED}-$(date +%Y%m%d-%H%M%S)}"
REDIS_PORT="${OPENSIM_REDIS_PORT:-6379}"

# 官方捆绑 Python 足够运行 train 的程序化感知，但刻意不含 numpy/torch；
# eval 的 YOLO worker 必须运行在项目视觉环境中。显式环境变量始终优先。
if [ -n "${HF2026_PYTHON_BIN:-}" ]; then
    PYTHON_BIN="$HF2026_PYTHON_BIN"
elif [ "$MODE" = "eval" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
    PYTHON_BIN="$ROOT_DIR/python/bin/python3.12"
fi

usage() {
    echo "用法: $0 {task1|task2} [时长秒=600] [随机种子=1] [train|eval] [输出目录]"
}

case "$TASK" in
    task1|search_track)
        MODEL_TASK="task1"
        SCENARIO="search_track"
        AGENT="competition.user_algorithms.search_track.highscore_agent:HighScoreSearchTrackAgent"
        ;;
    task2|coop_decoy)
        MODEL_TASK="task2"
        SCENARIO="coop_decoy"
        AGENT="competition.user_algorithms.coop_decoy.highscore_agent:HighScoreCoopAgent"
        ;;
    *)
        usage
        exit 2
        ;;
esac

if [ ! -x "$PYTHON_BIN" ]; then
    echo "错误: 缺少运行环境 Python: $PYTHON_BIN" >&2
    if [ "$MODE" = "eval" ]; then
        echo "请先执行提交包的 install_offline.sh 创建 .venv 并安装 YOLO 依赖。" >&2
    fi
    exit 2
fi
if [ ! -x "$ROOT_DIR/opensim-sim" ] || [ ! -x "$ROOT_DIR/bin/redis-server" ]; then
    echo "错误: opensim-sim 或 Redis 二进制缺失，请重新执行 git lfs pull。" >&2
    exit 2
fi
if [ "$MODE" != "train" ] && [ "$MODE" != "eval" ]; then
    echo "错误: 模式只能是 train 或 eval。" >&2
    exit 2
fi
if [ "$MODE" = "eval" ] && ! "$PYTHON_BIN" -c "import numpy, torch, ultralytics" 2>/dev/null; then
    echo "错误: eval 环境缺少 numpy/torch/ultralytics: $PYTHON_BIN" >&2
    echo "请执行提交包的 install_offline.sh 补齐视觉依赖。" >&2
    exit 2
fi

mkdir -p "$ROOT_DIR/.cache/hf2026" "$OUTPUT_DIR"
SOURCE_JSON="$ROOT_DIR/competition/scenarios/$SCENARIO/scenario.json"
RUNTIME_JSON="$ROOT_DIR/.cache/hf2026/${SCENARIO}-port${REDIS_PORT}.json"
PREPARE_ARGS=("$SOURCE_JSON" "$RUNTIME_JSON" --redis-port "$REDIS_PORT")
# 仅用于无 UE 的本地快速回归；正式评测不要设置，保持官方 1 倍速。
if [ -n "${HF2026_TIME_SCALE:-}" ]; then
    PREPARE_ARGS+=(--time-scale "$HF2026_TIME_SCALE")
fi
"$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_scenario.py" "${PREPARE_ARGS[@]}"

OWN_REDIS=0
if ! "$ROOT_DIR/bin/redis-cli" -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG; then
    mkdir -p "$ROOT_DIR/.cache/hf2026/redis"
    "$ROOT_DIR/bin/redis-server" --port "$REDIS_PORT" --daemonize yes \
        --pidfile "$ROOT_DIR/.cache/hf2026/redis-${REDIS_PORT}.pid" \
        --logfile "$ROOT_DIR/.cache/hf2026/redis-${REDIS_PORT}.log" \
        --dir "$ROOT_DIR/.cache/hf2026/redis" --save "" --appendonly no
    OWN_REDIS=1
fi
cleanup() {
    if [ "$OWN_REDIS" -eq 1 ]; then
        "$ROOT_DIR/bin/redis-cli" -p "$REDIS_PORT" shutdown nosave >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

EXTRA_ARGS=()
if [ "$MODE" = "eval" ]; then
    BINARY_MODEL="$ROOT_DIR/models/hf2026_vehicle_binary_v2.pt"
    DOMAIN_MODEL="$ROOT_DIR/models/hf2026_target_vehicle_domain_v1.pt"
    BUNDLED_MODEL="$ROOT_DIR/examples/yolotrack/target_vehicle_yolov8s.pt"
    if [ -n "${HF2026_YOLO_MODEL:-}" ]; then
        YOLO_MODEL="$HF2026_YOLO_MODEL"
    elif [ "$MODEL_TASK" = "task1" ] && [ -s "$DOMAIN_MODEL" ]; then
        # 赛题一无诱饵，单类 UE 域模型召回率明显高于二分类模型。
        YOLO_MODEL="$DOMAIN_MODEL"
    elif [ "$MODEL_TASK" != "task1" ]; then
        # 赛题二/三的身份判定必须包含诱饵类别，不能静默回退到单类模型。
        YOLO_MODEL="$BINARY_MODEL"
    else
        YOLO_MODEL="$BUNDLED_MODEL"
    fi
    if [ ! -s "$YOLO_MODEL" ]; then
        echo "错误: YOLO 模型不存在: $YOLO_MODEL" >&2
        if [ "$MODEL_TASK" != "task1" ]; then
            echo "请先执行 ./scripts/train_ue_yolo.sh dataset 80 生成二分类权重。" >&2
        else
            echo "仓库基础权重缺失时请执行 git lfs pull。" >&2
        fi
        exit 2
    fi
    "$PYTHON_BIN" "$ROOT_DIR/scripts/validate_yolo_model.py" "$YOLO_MODEL" "$MODEL_TASK"
    echo "YOLO 权重: $YOLO_MODEL"
    export HF2026_SENSOR_MODEL="$YOLO_MODEL"
    if [ "$MODEL_TASK" = "task1" ]; then
        export HF2026_TASK1_ROUTE_VISION="${HF2026_TASK1_ROUTE_VISION:-1}"
        echo "赛题一视觉融合: HF2026_TASK1_ROUTE_VISION=$HF2026_TASK1_ROUTE_VISION"
    fi
    EXTRA_ARGS+=(--photo-mode on --yolo-model "$YOLO_MODEL")
else
    EXTRA_ARGS+=(--photo-mode off)
fi

cd "$ROOT_DIR"
echo "运行 $SCENARIO | seed=$SEED | duration=${DURATION}s | mode=$MODE"
echo "Agent: $AGENT"
echo "输出: $OUTPUT_DIR"
HF2026_ROUTE_SEED="$SEED" "$PYTHON_BIN" -m competition run \
    --scenario "$SCENARIO" \
    --scenario-json "$RUNTIME_JSON" \
    --agent "$AGENT" \
    --duration "$DURATION" \
    --seed "$SEED" \
    --mode "$MODE" \
    --redis-port "$REDIS_PORT" \
    --output "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
