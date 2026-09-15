"""Low-latency shared YOLO sensor used by all three high-score agents.

The stock detector exposes only the single highest-confidence box.  That is
fragile in this map: a rock false positive, or a nearby decoy, can hide the
real target.  Player sensors are part of the public SDK contract, so we run
the same supplied/trained models here, keep every box, georeference it with
terrain-aware camera geometry, and let the agent select by its track prior.
"""
from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

from competition.sdk.core.observation import Detection

from competition.user_algorithms.perception_geometry import local_offset, offset_point


Point = Tuple[float, float]
_ROOT = Path(__file__).resolve().parents[2]
_MODELS: dict[str, object] = {}
_MODEL_LOCK = threading.Lock()


def _distance_m(a: Point, b: Point) -> float:
    north, east = local_offset(a, b)
    return math.hypot(north, east)


def _model(path: Path):
    key = str(path.resolve())
    with _MODEL_LOCK:
        if key not in _MODELS:
            from ultralytics import YOLO

            _MODELS[key] = YOLO(key)
        return _MODELS[key]


class SharedYoloSensor:
    """Per-agent throttling around a process-wide shared Ultralytics model."""

    def __init__(
        self,
        model_filename: str,
        confidence: float,
        interval_s: float,
        imgsz: int = 1024,
        terrain_scale: float = 0.62,
    ) -> None:
        override = os.environ.get("HF2026_SENSOR_MODEL", "").strip()
        self.model_path = Path(override) if override else _ROOT / "models" / model_filename
        self.confidence = float(confidence)
        self.interval_s = float(interval_s)
        self.imgsz = int(imgsz)
        self.terrain_scale = float(terrain_scale)
        self._last_wall = -1e9

    def detect(
        self,
        obs,
        *,
        target_hint: Optional[Point] = None,
        hint_gate_m: Optional[float] = None,
        target_only: bool = False,
    ) -> Optional[list[Detection]]:
        """Return ``None`` without a photo (retain train-mode simulator)."""
        if obs.self.photo is None:
            return None
        now = time.monotonic()
        if now - self._last_wall < self.interval_s:
            return []
        self._last_wall = now

        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(obs.self.photo, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return []
        model = _model(self.model_path)
        with _MODEL_LOCK:
            result = model(
                image,
                imgsz=self.imgsz,
                conf=self.confidence,
                verbose=False,
            )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        height, width = image.shape[:2]
        hfov = max(5.0, min(50.0, float(obs.self.gimbal_fov_deg)))
        vfov = math.degrees(
            2.0
            * math.atan(math.tan(math.radians(hfov) / 2.0) * height / width)
        )
        detections: list[tuple[Detection, float]] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = str(result.names.get(class_id, class_id))
            normalized = class_name.lower().replace("_", "")
            if target_only and normalized != "targetvehicle":
                continue
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
            centre_x, centre_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            pan_delta = (centre_x - width / 2.0) / width * hfov
            tilt_delta = (centre_y - height / 2.0) / height * vfov

            tilt = float(obs.self.gimbal_tilt) + tilt_delta
            if tilt >= -3.0:
                continue
            horizontal = (
                max(150.0, float(obs.self.alt) * self.terrain_scale)
                / math.tan(math.radians(-tilt))
            )
            if not math.isfinite(horizontal) or horizontal > 3_200.0:
                continue
            bearing = math.radians(
                float(obs.self.heading_deg)
                + float(obs.self.gimbal_pan)
                + pan_delta
            )
            point = offset_point(
                (obs.self.lat, obs.self.lon),
                horizontal * math.cos(bearing),
                horizontal * math.sin(bearing),
            )
            hint_distance = 0.0 if target_hint is None else _distance_m(target_hint, point)
            if hint_gate_m is not None and hint_distance > hint_gate_m:
                continue
            # The ``_geo`` suffix tells our post-filter not to apply the SDK
            # sea-level correction a second time; semantic substring matching
            # in tasks 2/3 still sees target/decoy normally.
            detections.append(
                (
                    Detection(
                        detected=True,
                        confidence=confidence,
                        target_lat=point[0],
                        target_lon=point[1],
                        target_type=f"{class_name}_geo",
                    ),
                    hint_distance,
                )
            )

        if target_hint is not None:
            detections.sort(key=lambda row: (row[1], -row[0].confidence))
        else:
            # Prefer target semantics so a high-confidence nearby decoy cannot
            # permanently hide a real target elsewhere in the same frame.
            detections.sort(
                key=lambda row: (
                    "target" not in row[0].target_type.lower(),
                    -row[0].confidence,
                )
            )
        return [row[0] for row in detections]
