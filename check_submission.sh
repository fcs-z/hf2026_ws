#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="${1:-}"
if [ -z "$SIM_ROOT" ]; then
    echo "用法: $0 /绝对路径/hf2026-sim" >&2
    exit 2
fi
SIM_ROOT="$(cd "$SIM_ROOT" && pwd)"
PYTHON_BIN="$SIM_ROOT/.venv/bin/python"

test -x "$SIM_ROOT/opensim-sim"
test -x "$SIM_ROOT/bin/redis-server"
test -x "$PYTHON_BIN"
test -s "$SIM_ROOT/models/hf2026_target_vehicle_domain_v1.pt"
test -s "$SIM_ROOT/models/hf2026_vehicle_binary_v2.pt"

expected_task1="5ffda110c99513ebb087924531b38c6063f46c806593c227ebb157118c1ad4e0"
expected_task2="abc028f08372efe7988a860c40417931d592d47b1cc510537191921fd99d9dec"
actual_task1=$(sha256sum "$SIM_ROOT/models/hf2026_target_vehicle_domain_v1.pt" | awk '{print $1}')
actual_task2=$(sha256sum "$SIM_ROOT/models/hf2026_vehicle_binary_v2.pt" | awk '{print $1}')
test "$actual_task1" = "$expected_task1"
test "$actual_task2" = "$expected_task2"

cd "$SIM_ROOT"
PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import cv2
import numpy
import torch
import ultralytics
from competition.user_algorithms.search_track.highscore_agent import HighScoreSearchTrackAgent
from competition.user_algorithms.coop_decoy.highscore_agent import HighScoreCoopAgent
from competition.user_algorithms.route_prior import load_public_routes

assert HighScoreSearchTrackAgent and HighScoreCoopAgent
assert len(load_public_routes(expanded=False)) == 26
assert len(load_public_routes(expanded=True)) == 26
print("Python 与算法导入正常")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
print("opencv", cv2.__version__, "numpy", numpy.__version__)
PY

echo "提交材料校验通过。"
echo "赛题一入口: competition.user_algorithms.search_track.highscore_agent:HighScoreSearchTrackAgent"
echo "赛题二入口: competition.user_algorithms.coop_decoy.highscore_agent:HighScoreCoopAgent"
