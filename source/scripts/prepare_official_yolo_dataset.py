#!/usr/bin/env python3
"""将 HF2026 官方 PNG+JSON 数据转换为两类 YOLO 检测数据集。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CLASS_IDS = {"TargetVehicle": 0, "DecoyVehicle": 1}
FOLDER_RE = re.compile(r"^(target|decoy)_(.+)_fov(\d+)$")
VIEW_RE = re.compile(r"^(p\d+)_az(\d+)$")


@dataclass(frozen=True)
class Sample:
    image: Path
    annotation: Path
    folder: str
    view_group: str
    primary_kind: str
    width: int
    height: int
    detections: tuple[tuple[int, float, float, float, float], ...]


def _parse_sample(annotation: Path, source: Path) -> Sample:
    folder = annotation.parent.name
    folder_match = FOLDER_RE.fullmatch(folder)
    if folder_match is None:
        raise ValueError(f"无法解析目录名: {folder}")
    expected_kind = "TargetVehicle" if folder_match.group(1) == "target" else "DecoyVehicle"
    data = json.loads(annotation.read_text(encoding="utf-8"))
    image = annotation.with_suffix(".png")
    if not image.is_file():
        raise ValueError(f"缺少同名 PNG: {image}")
    if data.get("image") != image.name:
        raise ValueError(f"JSON image 字段不匹配: {annotation}")
    if data.get("point_kind") != expected_kind:
        raise ValueError(
            f"目录类别与 point_kind 不匹配: {annotation}: {data.get('point_kind')}"
        )
    width, height = int(data.get("width", 0)), int(data.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"图像尺寸无效: {annotation}")

    view_match = VIEW_RE.fullmatch(annotation.stem)
    if view_match is None:
        raise ValueError(f"无法解析拍摄位置/方位: {annotation.name}")
    # 同一个 pXX+azimuth 的所有天气、FOV、目标/诱饵必须留在同一 split，
    # 否则几乎相同的背景会让验证指标虚高。
    view_group = f"{view_match.group(1)}_az{view_match.group(2)}"

    rows: list[tuple[int, float, float, float, float]] = []
    for detection in data.get("detections", []):
        class_name = str(detection.get("class", ""))
        if class_name not in CLASS_IDS:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in detection["bbox"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"bbox 无效: {annotation}: {detection}") from exc
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox 面积无效: {annotation}: {detection['bbox']}")
        rows.append(
            (
                CLASS_IDS[class_name],
                (x1 + x2) / (2.0 * width),
                (y1 + y2) / (2.0 * height),
                (x2 - x1) / width,
                (y2 - y1) / height,
            )
        )
    if not rows:
        raise ValueError(f"没有可用车辆标注: {annotation}")
    return Sample(
        image=image,
        annotation=annotation,
        folder=folder,
        view_group=view_group,
        primary_kind=expected_kind,
        width=width,
        height=height,
        detections=tuple(rows),
    )


def _split_groups(
    samples: list[Sample], seed: int, val_ratio: float, test_ratio: float
) -> dict[str, str]:
    group_kinds: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        group_kinds[sample.view_group].add(sample.primary_kind)

    strata: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for group, kinds in group_kinds.items():
        strata[tuple(sorted(kinds))].append(group)

    assignments: dict[str, str] = {}
    for signature, groups in sorted(strata.items()):
        groups.sort()
        signature_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{','.join(signature)}".encode()).digest()[:8], "big"
        )
        random.Random(signature_seed).shuffle(groups)
        count = len(groups)
        n_test = max(1, round(count * test_ratio)) if count >= 3 and test_ratio > 0 else 0
        n_val = max(1, round(count * val_ratio)) if count >= 3 and val_ratio > 0 else 0
        if n_test + n_val >= count:
            n_test, n_val = 1, 1
        for index, group in enumerate(groups):
            assignments[group] = (
                "test" if index < n_test else "val" if index < n_test + n_val else "train"
            )
    return assignments


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, nargs="?", default=Path("dataset"))
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path(".cache/hf2026/official-yolo-binary-seed2026"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir():
        raise SystemExit(f"官方数据集目录不存在: {source}")
    if output == source or source in output.parents:
        raise SystemExit("输出目录不能等于官方数据集或位于其内部")
    if not (0.0 <= args.val_ratio < 0.5 and 0.0 <= args.test_ratio < 0.5):
        raise SystemExit("val/test 比例必须在 [0, 0.5) 内")
    if args.val_ratio + args.test_ratio >= 0.6:
        raise SystemExit("val_ratio + test_ratio 必须小于 0.6")
    if output.exists() and any(output.iterdir()):
        manifest = output / "manifest.json"
        if manifest.is_file() and (output / "dataset.yaml").is_file():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            requested = {
                "source": str(source),
                "seed": args.seed,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
            }
            # 兼容本脚本早期 manifest（当时固定 0.15/0.15，未落盘比例字段）。
            legacy_defaults = {"val_ratio": 0.15, "test_ratio": 0.15}
            mismatch = {}
            for key, value in requested.items():
                current = existing.get(key, legacy_defaults.get(key))
                if current != value:
                    mismatch[key] = (current, value)
            if not mismatch:
                print(f"已存在转换结果，直接复用: {output}")
                print(manifest.read_text(encoding="utf-8"))
                return 0
            raise SystemExit(
                f"现有转换结果参数不同: {mismatch}；请为新参数指定另一个输出目录"
            )
        raise SystemExit(f"输出目录非空且不是完整转换结果: {output}")

    annotations = sorted(source.glob("*/*.json"))
    if not annotations:
        raise SystemExit(f"未找到官方 JSON 标注: {source}")
    samples = [_parse_sample(path, source) for path in annotations]
    assignments = _split_groups(samples, args.seed, args.val_ratio, args.test_ratio)

    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    link_modes: Counter[str] = Counter()
    for sample in samples:
        split = assignments[sample.view_group]
        stem = f"{sample.folder}__{sample.annotation.stem}"
        image_destination = output / "images" / split / f"{stem}.png"
        label_destination = output / "labels" / split / f"{stem}.txt"
        link_modes[_link_or_copy(sample.image, image_destination)] += 1
        lines = []
        for class_id, xc, yc, width, height in sample.detections:
            lines.append(f"{class_id} {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}")
            class_name = "TargetVehicle" if class_id == 0 else "DecoyVehicle"
            class_counts[f"{split}:{class_name}"] += 1
        label_destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        image_counts[split] += 1
        box_counts[split] += len(lines)

    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(
        f"path: {output}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: TargetVehicle\n"
        "  1: DecoyVehicle\n",
        encoding="utf-8",
    )
    manifest_data = {
        "generator": "prepare_official_yolo_dataset.py",
        "source": str(source),
        "output": str(output),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "split_unit": "camera-position+azimuth",
        "images": dict(sorted(image_counts.items())),
        "boxes": dict(sorted(box_counts.items())),
        "class_boxes": dict(sorted(class_counts.items())),
        "view_groups": dict(Counter(assignments.values())),
        "storage": dict(link_modes),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest_data, ensure_ascii=False, indent=2))
    print(f"YOLO 配置: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
