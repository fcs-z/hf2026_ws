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
test -s "$SIM_ROOT/config/public_astar_routes.json"
test -s "$SIM_ROOT/config/public_astar_routes_v203.json"
test -s "$SIM_ROOT/scripts/configure_ue_camera_capacity.py"

expected_task1="5ffda110c99513ebb087924531b38c6063f46c806593c227ebb157118c1ad4e0"
expected_task2="abc028f08372efe7988a860c40417931d592d47b1cc510537191921fd99d9dec"
actual_task1=$(sha256sum "$SIM_ROOT/models/hf2026_target_vehicle_domain_v1.pt" | awk '{print $1}')
actual_task2=$(sha256sum "$SIM_ROOT/models/hf2026_vehicle_binary_v2.pt" | awk '{print $1}')
test "$actual_task1" = "$expected_task1"
test "$actual_task2" = "$expected_task2"

cd "$SIM_ROOT"
PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import os
import re
from pathlib import Path

import cv2
import numpy
import torch
import ultralytics
from competition.user_algorithms.search_track.highscore_agent import HighScoreSearchTrackAgent
from competition.user_algorithms.coop_decoy.highscore_agent import HighScoreCoopAgent
from competition.user_algorithms.route_prior import _expanded_route_file, load_public_routes

assert HighScoreSearchTrackAgent and HighScoreCoopAgent
assert len(load_public_routes(expanded=False)) == 26
for cache in (
    "config/public_astar_routes.json",
    "config/public_astar_routes_v203.json",
):
    os.environ["HF2026_ROUTE_CACHE"] = cache
    routes = load_public_routes(expanded=True)
    assert len(routes) == 26
    assert all(route.length_m > 100.0 for route in routes)
os.environ.pop("HF2026_ROUTE_CACHE", None)
version_text = Path("VERSION").read_text(encoding="utf-8", errors="replace")
match = re.search(r"Simulation\s+(\d+)\.(\d+)\.(\d+)", version_text)
version = tuple(int(part) for part in match.groups()) if match else (0, 0, 0)
expected_cache = "public_astar_routes_v203.json" if version >= (2, 0, 3) else "public_astar_routes.json"
assert _expanded_route_file().name == expected_cache
print("路线缓存", _expanded_route_file().name)
print("Python 与算法导入正常")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
print("opencv", cv2.__version__, "numpy", numpy.__version__)
PY

"$PYTHON_BIN" "$SIM_ROOT/scripts/configure_ue_camera_capacity.py" "$SIM_ROOT" 3 --check

echo "提交材料校验通过。"
echo "赛题一入口: competition.user_algorithms.search_track.highscore_agent:HighScoreSearchTrackAgent"
echo "赛题二入口: competition.user_algorithms.coop_decoy.highscore_agent:HighScoreCoopAgent"
