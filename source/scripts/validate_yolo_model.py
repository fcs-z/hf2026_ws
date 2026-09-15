#!/usr/bin/env python3
"""在启动昂贵的 UE 评测前，验证 YOLO 权重的 HF2026 类别表。"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("task", choices=("task1", "task2"))
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"YOLO 权重不存在: {model_path}")
    names = YOLO(str(model_path)).names
    values = set(names.values())
    required = {"TargetVehicle"}
    if args.task == "task2":
        required = {"TargetVehicle", "DecoyVehicle"}
        if values != required:
            raise SystemExit(
                f"{args.task} 权重必须恰好包含 {sorted(required)}；实际类别表: {names}。"
                "赛题二禁止回退到单类模型或使用未适配类别表。"
            )
    missing = required - values
    if missing:
        raise SystemExit(
            f"{args.task} 权重缺少类别 {sorted(missing)}；实际类别表: {names}。"
            "赛题二禁止回退到单类模型。"
        )
    print(f"YOLO 类别校验通过: {model_path} -> {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
