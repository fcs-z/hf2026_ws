#!/usr/bin/env python3
"""Keep the local UE capture capacity consistent with the active renderer."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
from pathlib import Path


def _positive_int(value: object, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _backup(path: Path, root: Path, backup_root: Path | None) -> None:
    if backup_root is None:
        return
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path("external_ue") / path.name
    destination = backup_root / relative
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def configure(root: Path, required: int, *, check: bool, backup_root: Path | None) -> int:
    renderer_dir = root / "config" / "renderers"
    captures_found = 0
    failures: list[str] = []

    for renderer_path in sorted(renderer_dir.glob("*.json")):
        if renderer_path.name.endswith(".template.json"):
            continue
        try:
            renderer = json.loads(renderer_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            failures.append(f"无法读取 {renderer_path}: {exc}")
            continue

        workdir_text = str(renderer.get("executable", {}).get("workdir", "")).strip()
        if not workdir_text or (workdir_text.startswith("<") and workdir_text.endswith(">")):
            continue
        renderer_capacity = _positive_int(
            renderer.get("capacity", {}).get("max_aircraft_per_instance")
        )
        if renderer_capacity and renderer_capacity < required:
            failures.append(
                f"{renderer_path.name} 仅声明 {renderer_capacity} 路，少于所需 {required} 路"
            )
            continue

        workdir = Path(workdir_text)
        if not workdir.is_absolute():
            workdir = root / workdir
        capture_path = workdir / "testwl" / "Content" / "Config" / "capture_config.json"
        if not capture_path.is_file():
            continue
        captures_found += 1

        try:
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            failures.append(f"无法读取 {capture_path}: {exc}")
            continue
        current = _positive_int(capture.get("render", {}).get("max_aircraft"))
        if current >= required:
            print(f"UE 相机容量正常: {current} 路 ({capture_path})")
            continue
        if check:
            failures.append(
                f"{capture_path} 的 max_aircraft={current}，赛题二至少需要 {required}"
            )
            continue

        _backup(capture_path, root, backup_root)
        capture.setdefault("render", {})["max_aircraft"] = required
        original_mode = stat.S_IMODE(capture_path.stat().st_mode)
        capture_path.write_text(
            json.dumps(capture, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.chmod(capture_path, original_mode)
        print(f"UE 相机容量已修正: {current} -> {required} 路 ({capture_path})")

    if captures_found == 0:
        print("未发现本地 UE capture_config.json；远程渲染池由平台侧配置容量。")
    if failures:
        for failure in failures:
            print(f"错误: {failure}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="HF2026 仿真平台根目录")
    parser.add_argument("required", type=int, nargs="?", default=3)
    parser.add_argument("--check", action="store_true", help="只检查，不修改")
    parser.add_argument("--backup-root", type=Path)
    args = parser.parse_args()
    if args.required < 1:
        parser.error("required 必须大于 0")
    return configure(
        args.root.resolve(),
        args.required,
        check=args.check,
        backup_root=args.backup_root.resolve() if args.backup_root else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
