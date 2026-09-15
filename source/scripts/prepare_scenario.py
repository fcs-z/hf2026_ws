#!/usr/bin/env python3
"""为并行基准生成只改端口/时间倍率的场景副本，不改官方 scenario.json。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--redis-port", type=int, required=True)
    parser.add_argument("--time-scale", type=float, default=None)
    args = parser.parse_args()

    data = json.loads(args.source.read_text(encoding="utf-8-sig"))
    simulation = data.setdefault("simulation", {})
    simulation["redis_port"] = args.redis_port
    if args.time_scale is not None:
        simulation["time_scale"] = args.time_scale
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

