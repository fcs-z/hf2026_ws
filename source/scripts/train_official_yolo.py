#!/usr/bin/env python3
"""训练 HF2026 TargetVehicle/DecoyVehicle 两类 YOLOv8s 检测模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from ultralytics import YOLO
import yaml


EXPECTED_NAMES = {"TargetVehicle", "DecoyVehicle"}


def _validate_dataset(data_path: Path) -> None:
    dataset = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = dataset.get("names", {})
    name_values = set(names.values() if isinstance(names, dict) else names)
    if name_values != EXPECTED_NAMES:
        raise SystemExit(
            f"dataset.yaml 必须恰好包含 {sorted(EXPECTED_NAMES)}，实际为 {names}"
        )

    label_root = data_path.parent / "labels"
    seen_ids: set[int] = set()
    label_count = 0
    for label in label_root.rglob("*.txt"):
        label_count += 1
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                class_id = int(line.split(maxsplit=1)[0])
            except ValueError as exc:
                raise SystemExit(f"标注类别不是整数: {label}: {line}") from exc
            if class_id not in (0, 1):
                raise SystemExit(f"标注出现未知类别 {class_id}: {label}")
            seen_ids.add(class_id)
    if label_count == 0 or not {0, 1}.issubset(seen_ids):
        raise SystemExit(
            f"标注必须同时包含类别 0 和 1，实际文件数={label_count}，"
            f"类别={sorted(seen_ids)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="dataset.yaml")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run-name", default="official-vehicle-binary-v2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    base_model = args.base_model.resolve()
    output_path = args.output.resolve()
    project = args.project.resolve()
    if not data_path.is_file():
        raise SystemExit(f"dataset.yaml 不存在: {data_path}")
    if not base_model.is_file():
        raise SystemExit(f"基础权重不存在: {base_model}")
    if min(args.epochs, args.imgsz, args.batch) <= 0 or args.workers < 0:
        raise SystemExit("epochs/imgsz/batch 必须为正数，workers 不能为负数")
    _validate_dataset(data_path)

    model = YOLO(str(base_model))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        patience=15,
        close_mosaic=10,
        mosaic=0.50,
        mixup=0.05,
        degrees=7.0,
        translate=0.08,
        scale=0.25,
        shear=1.0,
        perspective=0.0002,
        flipud=0.05,
        fliplr=0.50,
        hsv_h=0.012,
        hsv_s=0.45,
        hsv_v=0.30,
        amp=True,
        cache=False,
        plots=True,
        project=str(project),
        name=args.run_name,
        exist_ok=False,
    )
    trainer = model.trainer
    if trainer is None:
        raise SystemExit("Ultralytics 未返回 trainer，无法定位训练权重")
    best = Path(trainer.best).resolve()
    if not best.is_file():
        raise SystemExit(f"未生成 best.pt: {best}")

    trained_model = YOLO(str(best))
    trained_names = trained_model.names
    if set(trained_names.values()) != EXPECTED_NAMES:
        raise SystemExit(f"训练权重类别表错误: {trained_names}")

    # test split 从未参与 early stopping 或 best.pt 挑选。
    test_metrics = trained_model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        project=str(project),
        name=f"{args.run_name}-test",
        exist_ok=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    result_dict = getattr(test_metrics, "results_dict", {})
    metrics = {
        "model": str(output_path),
        "base_model": str(base_model),
        "dataset": str(data_path),
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "class_names": trained_names,
        "test": {str(key): float(value) for key, value in result_dict.items()},
        "train_run": str(Path(trainer.save_dir).resolve()),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"最佳权重已写入: {output_path}")
    print(f"独立测试指标已写入: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
