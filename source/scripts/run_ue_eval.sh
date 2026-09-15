#!/usr/bin/env bash
# 通过 bridge 启动 UE + PhotoCache + YOLO 的正式视觉评测链（固定 600 秒）。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:-}"
SEED="${2:-1}"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
BINARY_MODEL="$ROOT_DIR/models/hf2026_vehicle_binary_v2.pt"
DOMAIN_MODEL="$ROOT_DIR/models/hf2026_target_vehicle_domain_v1.pt"
BUNDLED_MODEL="$ROOT_DIR/examples/yolotrack/target_vehicle_yolov8s.pt"

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
        echo "用法: $0 {task1|task2} [随机种子=1]" >&2
        exit 2
        ;;
esac

if [ -n "${HF2026_YOLO_MODEL:-}" ]; then
    YOLO_MODEL="$HF2026_YOLO_MODEL"
elif [ "$MODEL_TASK" = "task1" ] && [ -s "$DOMAIN_MODEL" ]; then
    # 赛题一没有诱饵，优先使用在 UE 实景上验证过的单类域适配模型。
    YOLO_MODEL="$DOMAIN_MODEL"
elif [ "$MODEL_TASK" != "task1" ]; then
    YOLO_MODEL="$BINARY_MODEL"
else
    YOLO_MODEL="$BUNDLED_MODEL"
fi

if [ ! -x "$PYTHON_BIN" ] || ! "$PYTHON_BIN" -c "import torch, ultralytics, cv2" >/dev/null 2>&1; then
    echo "错误: 视觉环境未就绪。先执行: uv sync --extra vision --locked" >&2
    exit 2
fi
if [ ! -s "$YOLO_MODEL" ]; then
    echo "错误: YOLO 权重不存在: $YOLO_MODEL" >&2
    if [ "$MODEL_TASK" != "task1" ]; then
        echo "请先执行 ./scripts/train_ue_yolo.sh dataset 80 生成二分类权重。" >&2
    fi
    exit 2
fi

# opensim-render-ctl 会按配置中的最小空闲显存筛掉本地 GPU。提前给出明确错误，避免
# UE 启动后所有飞机都落入 excess_uavs、最终又只得到模拟感知。远程渲染池可跳过此项。
if [ "${HF2026_SKIP_GPU_PREFLIGHT:-0}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | sort -nr | head -1 | awk '{print int($1)}')
    MIN_VRAM_MB=$(
        "$PYTHON_BIN" -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["gpu_requirements"]["min_vram_mb_per_instance"])' \
            "$ROOT_DIR/config/renderers/ue_testwl.json" 2>/dev/null || echo 8192
    )
    if [ -n "$GPU_FREE_MB" ] && [ "$GPU_FREE_MB" -lt "$MIN_VRAM_MB" ]; then
        echo "错误: 当前最大空闲显存仅 ${GPU_FREE_MB} MiB，UE 渲染器要求至少 ${MIN_VRAM_MB} MiB。" >&2
        echo "请先结束占用 GPU 的训练/推理进程，再执行本命令。远程 UE 可设置 HF2026_SKIP_GPU_PREFLIGHT=1。" >&2
        exit 2
    fi
fi
"$PYTHON_BIN" "$ROOT_DIR/scripts/validate_yolo_model.py" "$YOLO_MODEL" "$MODEL_TASK"

cd "$ROOT_DIR"
echo "将重启本项目的 Redis/bridge/前端，并运行 600 秒 $SCENARIO eval。"
echo "YOLO 权重: $YOLO_MODEL"
export HF2026_ROUTE_SEED="$SEED"
export HF2026_SENSOR_MODEL="$YOLO_MODEL"
if [ "$MODEL_TASK" = "task1" ]; then
    # 路线预测负责连续上报，YOLO 负责 UE 画面校验和动态视场控制。
    # 需要复现旧版纯路线 88.64 分时，可在命令前显式设为 0。
    export HF2026_TASK1_ROUTE_VISION="${HF2026_TASK1_ROUTE_VISION:-1}"
    echo "赛题一视觉融合: HF2026_TASK1_ROUTE_VISION=$HF2026_TASK1_ROUTE_VISION"
fi
# 路线预测无需固定等待；纯视觉 routeSeed=0 可手动设置 HF2026_UE_WARMUP_S=30。
PYTHON_BIN="$PYTHON_BIN" \
HF2026_UE_WARMUP_S="${HF2026_UE_WARMUP_S:-0}" \
OPENSIM_REDIS_PORT=6379 OPENSIM_NO_OPEN_BROWSER=1 ./start.sh

# start.sh 遇到占用会顺延端口，以它落盘的实际端口为准。
# shellcheck disable=SC1091
. "$ROOT_DIR/run/env.sh"
API="http://127.0.0.1:${OPENSIM_CAM_PORT}/api/sim"
run_started=$(date +%s)
body=$(printf '{"scenario":"%s","agent":"%s","mode":"eval","photoMode":"on","yoloModel":"%s","routeSeed":%s}' \
    "$SCENARIO" "$AGENT" "$YOLO_MODEL" "$SEED")
response=$(curl --fail --silent --show-error -X POST -H 'Content-Type: application/json' \
    "$API/start" --data-binary "$body")
echo "启动响应: $response"

cleanup_run() {
    curl --silent --show-error -X POST "$API/stop" >/dev/null 2>&1 || true
}
trap cleanup_run INT TERM

last=""
while true; do
    response=$(curl --fail --silent --show-error "$API/status")
    status=$(printf '%s' "$response" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["status"])')
    if [ "$status" != "$last" ]; then
        echo "状态: $response"
        last="$status"
    fi
    case "$status" in
        idle)
            break
            ;;
        error)
            echo "评测失败: $response" >&2
            exit 1
            ;;
    esac
    sleep 5
done

output="$ROOT_DIR/competition/scenarios/$SCENARIO/output"
latest=$(find "$output" -maxdepth 1 -name '*.evaluation.json' -printf '%T@ %p\n' \
    | sort -nr \
    | awk -v started="$run_started" '$1 >= started {sub(/^[^ ]+ /, ""); print; exit}')
if [ -z "$latest" ]; then
    echo "错误: 未找到 evaluation.json；检查 $output/controller.stderr.log" >&2
    exit 1
fi
echo "评测完成: $latest"
"$PYTHON_BIN" "$ROOT_DIR/scripts/summarize_results.py" "$output"
echo "平台仍在运行；不再使用时执行 ./stop.sh。"
