#!/usr/bin/env python3
"""汇总 competition 生成的 evaluation.json，不依赖第三方包。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "scenario",
    "total_score",
    "passed",
    "n_reports",
    "targeting_rmse_m",
    "n_destroyed",
    "alive_rate",
    "undestroyed_decoy_misid_s",
    "dimension_scores",
    "penalty_breakdown",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for path in sorted(args.root.rglob("*.evaluation.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {key: data.get(key) for key in FIELDS if key in data}
        row["evaluation_file"] = str(path.relative_to(args.root))
        rows.append(row)
    destination = args.root / "summary.json"
    destination.write_text(
        json.dumps({"runs": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"runs": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

