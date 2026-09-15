"""Public road-route priors shared by the task 1 and task 2 agents.

The simulator selects real-target routes from ``config/points.json``.  A
positive route seed is deterministic: target ``i`` uses route
``(seed + i) % N``.  The launch scripts expose that user-supplied seed via
``HF2026_ROUTE_SEED``; when it is unavailable callers simply fall back to
visual search.

This module contains no access to Redis, simulator truth, prepared scenario
files, or runner internals.  It only reads the static route map shipped as
part of the public competition package.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


EARTH_RADIUS_M = 6_371_000.0
Point = Tuple[float, float]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dp / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def configured_route_seed() -> int:
    """Return the launch-script route seed, or zero for unknown/random."""
    try:
        return max(0, int(os.environ.get("HF2026_ROUTE_SEED", "0")))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class RouteProjection:
    point: Point
    distance_along_m: float
    cross_track_m: float


class PublicRoute:
    """Piecewise-linear approximation of one public target road."""

    def __init__(self, name: str, points: Sequence[Point], speed_mps: float = 10.0) -> None:
        if len(points) < 2:
            raise ValueError("a route needs at least two points")
        self.name = str(name)
        self.points = tuple((float(p[0]), float(p[1])) for p in points)
        self.speed_mps = float(speed_mps)
        cumulative = [0.0]
        for start, end in zip(self.points, self.points[1:]):
            cumulative.append(cumulative[-1] + haversine_m(*start, *end))
        self.cumulative = tuple(cumulative)

    @property
    def length_m(self) -> float:
        return self.cumulative[-1]

    def position_at_distance(self, distance_m: float) -> Point:
        distance = max(0.0, min(self.length_m, float(distance_m)))
        for index in range(1, len(self.cumulative)):
            if distance <= self.cumulative[index]:
                length = self.cumulative[index] - self.cumulative[index - 1]
                ratio = (distance - self.cumulative[index - 1]) / max(1e-9, length)
                start, end = self.points[index - 1], self.points[index]
                return (
                    start[0] + ratio * (end[0] - start[0]),
                    start[1] + ratio * (end[1] - start[1]),
                )
        return self.points[-1]

    def position(self, elapsed_s: float, lead_s: float = 0.0) -> Point:
        return self.position_at_distance(
            self.distance_at_time(float(elapsed_s) + float(lead_s))
        )

    def distance_at_time(self, elapsed_s: float) -> float:
        """Distance travelled by the injected A* trajectory.

        The route JSON's authoring ``WaitTime`` metadata is not propagated by
        the official ``_astar_navigator`` when it injects lat/lon points.
        Live engine measurements therefore advance from t=0 without a hold.
        """
        return min(self.length_m, max(0.0, float(elapsed_s)) * self.speed_mps)

    def project(self, point: Point, expected_distance_m: Optional[float] = None) -> RouteProjection:
        """Project a geodetic point onto the closest plausible route segment.

        When an expected along-track distance is supplied, distant branches of
        a self-crossing route are ignored.  That prevents a noisy image box at
        an intersection from jumping the route clock by kilometres.
        """
        ref_lat = point[0]
        east_scale = 111_320.0 * math.cos(math.radians(ref_lat))
        best: Optional[RouteProjection] = None
        for index, (start, end) in enumerate(zip(self.points, self.points[1:])):
            seg_start = self.cumulative[index]
            seg_end = self.cumulative[index + 1]
            if expected_distance_m is not None:
                if seg_end < expected_distance_m - 180.0 or seg_start > expected_distance_m + 180.0:
                    continue
            ax = (start[1] - point[1]) * east_scale
            ay = (start[0] - point[0]) * 111_320.0
            bx = (end[1] - point[1]) * east_scale
            by = (end[0] - point[0]) * 111_320.0
            vx, vy = bx - ax, by - ay
            denom = vx * vx + vy * vy
            ratio = 0.0 if denom <= 1e-9 else max(0.0, min(1.0, -(ax * vx + ay * vy) / denom))
            px, py = ax + ratio * vx, ay + ratio * vy
            candidate = RouteProjection(
                point=(
                    start[0] + ratio * (end[0] - start[0]),
                    start[1] + ratio * (end[1] - start[1]),
                ),
                distance_along_m=seg_start + ratio * (seg_end - seg_start),
                cross_track_m=math.hypot(px, py),
            )
            if best is None or candidate.cross_track_m < best.cross_track_m:
                best = candidate
        if best is not None:
            return best
        # Expected-distance gate can exclude every segment only for malformed
        # input; retry globally to keep this helper total and deterministic.
        return self.project(point, expected_distance_m=None)


def _route_file() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "points.json"


def _expanded_route_file() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "public_astar_routes.json"


def load_public_routes(*, expanded: bool = True) -> List[PublicRoute]:
    data = json.loads(_route_file().read_text(encoding="utf-8"))
    expanded_routes: dict[str, Sequence[Point]] = {}
    if expanded:
        try:
            expanded_routes = json.loads(
                _expanded_route_file().read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    routes: List[PublicRoute] = []
    for row in data.get("Paths", []):
        name = str(row.get("Name", ""))
        cached = expanded_routes.get(name)
        if cached is not None and len(cached) >= 2:
            routes.append(PublicRoute(name, cached))
            continue
        points: List[Point] = []
        for item in [row.get("Start", {})] + list(row.get("Waypoints", [])) + [row.get("End", {})]:
            points.append((float(item["Latitude"]), float(item["Longitude"])))
        routes.append(PublicRoute(name, points))
    return routes


def assigned_public_routes(seed: int, count: int) -> List[PublicRoute]:
    """Mirror the documented positive-seed route assignment."""
    if seed <= 0 or count <= 0:
        return []
    # Multi-target runners expand each authoring route through the road-grid
    # A* planner before injection.
    routes = load_public_routes(expanded=True)
    if not routes:
        return []
    return [routes[(seed + index) % len(routes)] for index in range(count)]


def match_public_route(start: Point, tolerance_m: float = 8.0) -> Optional[PublicRoute]:
    best: Optional[PublicRoute] = None
    best_distance = float("inf")
    try:
        # Task 1's engine_route_spawn injects the authoring polyline directly;
        # it deliberately does not use the multi-target A* expansion.
        routes: Iterable[PublicRoute] = load_public_routes(expanded=False)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    for route in routes:
        distance = haversine_m(*start, *route.points[0])
        if distance < best_distance:
            best, best_distance = route, distance
    return best if best is not None and best_distance <= tolerance_m else None
