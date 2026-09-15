#!/usr/bin/env bash
# 将赛题一、赛题二算法和离线 Python 环境安装到官方 Linux 仿真平台。
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="${1:-}"
UV_BIN="$PACKAGE_DIR/runtime/bin/uv"

if [ -z "$SIM_ROOT" ]; then
    echo "用法: $0 /绝对路径/hf2026-sim" >&2
    exit 2
fi
SIM_ROOT="$(cd "$SIM_ROOT" 2>/dev/null && pwd)" || {
    echo "错误: 仿真平台目录不存在: $SIM_ROOT" >&2
    exit 2
}
if [ ! -d "$SIM_ROOT/competition/sdk" ] || [ ! -f "$SIM_ROOT/competition/__main__.py" ]; then
    echo "错误: $SIM_ROOT 不是完整的 HF2026 仿真平台目录。" >&2
    exit 2
fi
if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "错误: 本离线依赖包仅适配 Linux x86_64。" >&2
    exit 2
fi
if [ ! -x "$UV_BIN" ]; then
    echo "错误: 缺少离线 uv 运行程序: $UV_BIN" >&2
    exit 2
fi
if ! command -v python3.12 >/dev/null 2>&1; then
    echo "错误: 未找到 Ubuntu 24.04 默认的 Python 3.12。" >&2
    exit 2
fi

backup_root="$SIM_ROOT/.hf2026_submission_backup/$(date +%Y%m%d-%H%M%S)"
backup_one() {
    local relative="$1"
    if [ -f "$SIM_ROOT/$relative" ]; then
        mkdir -p "$backup_root/$(dirname "$relative")"
        cp -p "$SIM_ROOT/$relative" "$backup_root/$relative"
    fi
}
install_one() {
    local relative="$1"
    local mode="${2:-0644}"
    backup_one "$relative"
    install -D -m "$mode" "$PACKAGE_DIR/source/$relative" "$SIM_ROOT/$relative"
}

install_one competition/user_algorithms/__init__.py
install_one competition/user_algorithms/perception_geometry.py
install_one competition/user_algorithms/route_prior.py
install_one competition/user_algorithms/vision_sensor.py
install_one competition/user_algorithms/search_track/__init__.py
install_one competition/user_algorithms/search_track/highscore_agent.py
install_one competition/user_algorithms/coop_decoy/__init__.py
install_one competition/user_algorithms/coop_decoy/highscore_agent.py
install_one competition/scenarios/search_track/config/algorithm.yaml
install_one competition/scenarios/coop_decoy/config/algorithm.yaml
install_one config/points.json
install_one config/public_astar_routes.json

for script in prepare_official_yolo_dataset.py prepare_scenario.py \
    summarize_results.py train_official_yolo.py validate_yolo_model.py; do
    install_one "scripts/$script"
done
for script in run_highscore.sh run_ue_eval.sh train_ue_yolo.sh; do
    install_one "scripts/$script" 0755
done

for model in hf2026_target_vehicle_domain_v1.pt hf2026_vehicle_binary_v2.pt \
    hf2026_vehicle_binary_v2.metrics.json; do
    relative="models/$model"
    backup_one "$relative"
    install -D -m 0644 "$PACKAGE_DIR/$relative" "$SIM_ROOT/$relative"
done

wheel_count=$(find "$PACKAGE_DIR/runtime/wheels" -maxdepth 1 -type f -name '*.whl' | wc -l)
if [ "$wheel_count" -eq 0 ]; then
    echo "错误: 离线 wheel 目录为空，无法创建视觉环境。" >&2
    exit 2
fi

if [ ! -x "$SIM_ROOT/.venv/bin/python" ]; then
    "$UV_BIN" venv --offline --no-python-downloads \
        --python "$(command -v python3.12)" "$SIM_ROOT/.venv"
fi
if [ "$("$SIM_ROOT/.venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]; then
    echo "错误: 已有 $SIM_ROOT/.venv 不是 Python 3.12，请改名后重新安装。" >&2
    exit 2
fi
UV_CACHE_DIR="$SIM_ROOT/.cache/uv-offline" "$UV_BIN" pip install \
    --offline --no-python-downloads --no-index \
    --find-links "$PACKAGE_DIR/runtime/wheels" \
    --python "$SIM_ROOT/.venv/bin/python" \
    --requirement "$PACKAGE_DIR/runtime/requirements.txt"

export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$SIM_ROOT"
"$SIM_ROOT/.venv/bin/python" - <<'PY'
from competition.user_algorithms.search_track.highscore_agent import HighScoreSearchTrackAgent
from competition.user_algorithms.coop_decoy.highscore_agent import HighScoreCoopAgent
from ultralytics import YOLO

assert HighScoreSearchTrackAgent and HighScoreCoopAgent
assert set(YOLO("models/hf2026_target_vehicle_domain_v1.pt").names.values()) == {"TargetVehicle"}
assert set(YOLO("models/hf2026_vehicle_binary_v2.pt").names.values()) == {"TargetVehicle", "DecoyVehicle"}
PY

echo "安装完成: $SIM_ROOT"
if [ -d "$backup_root" ]; then
    echo "被替换文件的备份: $backup_root"
fi
echo "下一步: $PACKAGE_DIR/check_submission.sh '$SIM_ROOT'"
