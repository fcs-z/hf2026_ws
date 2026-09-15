#!/usr/bin/env bash
set -euo pipefail
SIM_ROOT="${1:-}"
SEED="${2:-1}"
MODE="${3:-eval}"
DURATION="${4:-600}"
if [ -z "$SIM_ROOT" ]; then
    echo "用法: $0 /绝对路径/hf2026-sim [seed=1] [eval|train] [duration=600]" >&2
    exit 2
fi
cd "$SIM_ROOT"
case "$MODE" in
    eval)
        exec env HF2026_TASK1_ROUTE_VISION=1 ./scripts/run_ue_eval.sh task1 "$SEED"
        ;;
    train)
        exec ./scripts/run_highscore.sh task1 "$DURATION" "$SEED" train
        ;;
    *)
        echo "错误: 模式只能是 eval 或 train，当前值: $MODE" >&2
        exit 2
        ;;
esac
