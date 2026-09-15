"""Shared geometry helpers for the UE/YOLO competition agents.

The SDK's compiled :class:`YoloDetector` intersects the camera ray with
``altitude == 0``.  The HF2026 map is roughly 150--250 m above that datum,
while aircraft fly at 500 m AMSL.  Consequently an otherwise correct box is
projected too far away.  This module corrects only semantic YOLO detections;
the train-mode ``ground_vehicle`` observations are already georeferenced and
must not be scaled a second time.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple


Point = Tuple[float, float]


def local_offset(origin: Point, point: Point) -> Tuple[float, float]:
    """Return ``(north, east)`` metres from *origin* to *point*."""
    north = (point[0] - origin[0]) * 111_320.0
    east = (
        (point[1] - origin[1])
        * 111_320.0
        * math.cos(math.radians(origin[0]))
    )
    return north, east


def offset_point(origin: Point, north_m: float, east_m: float) -> Point:
    cosine = max(0.05, math.cos(math.radians(origin[0])))
    return (
        origin[0] + north_m / 111_320.0,
        origin[1] + east_m / (111_320.0 * cosine),
    )


def is_visual_detection(target_type: str) -> bool:
    """Whether a Detection came from our semantic UE YOLO model."""
    normalized = str(target_type or "").lower().replace("_", "")
    return normalized.startswith("targetvehicle") or normalized.startswith("decoyvehicle")


class TerrainRangeCorrector:
    """Compensate the SDK's sea-level ray/ground intersection.

    ``scale`` is ``(uav_altitude - terrain_altitude) / uav_altitude``.  A
    conservative 0.60 default matches the current map (500 m UAV, about
    200 m terrain).  When a target prediction is available, along-ray least
    squares refines it slowly while rejecting cross-track outliers.
    """

    def __init__(self, default_scale: float = 0.62) -> None:
        self.scale = min(0.90, max(0.35, float(default_scale)))
        self.samples = 0

    def correct(
        self,
        uav: Point,
        raw: Point,
        target_type: str,
        predicted: Optional[Point] = None,
    ) -> Point:
        # AccuracySimulator and custom sensors already return ground truth
        # coordinates (plus their configured noise).
        normalized = str(target_type or "").lower().replace("_", "")
        if not is_visual_detection(target_type) or normalized.endswith("geo"):
            return raw

        raw_n, raw_e = local_offset(uav, raw)
        # YoloDetector already adds aircraft heading before calling the
        # compiled projector.  Only the ray length needs terrain correction.
        # Do not fit scale to the track prediction here.  Doing so creates a
        # positive feedback loop: a background false positive on roughly the
        # same bearing can be pulled onto the prediction and then accepted by
        # the association gate.  The fixed map-wide value leaves only a few
        # tens of metres of range error, which the temporal filters absorb.

        return offset_point(uav, self.scale * raw_n, self.scale * raw_e)

    def vertical_separation(self, uav_alt_m: float) -> float:
        """Estimated UAV-to-terrain vertical separation in metres."""
        return min(430.0, max(160.0, float(uav_alt_m) * self.scale))
