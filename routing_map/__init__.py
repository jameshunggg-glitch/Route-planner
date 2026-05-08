"""routing_map: modular coastal/sea routing builder.

This package is a refactor template extracted from the notebook `route_planner_point_to_port.ipynb`.
Use `build_aoi.build_aoi()` as the main entrypoint for AOI runs.
"""

from .config import RoutingMapConfig, NgzRingBuildConfig
from .build_aoi import build_aoi
from .ngz import (
    NgzInput,
    NgzGroup,
    NgzRingResult,
    NgzOverlay,
    NgzInsideError,
    build_ngz_overlay,
    build_ngz_t_rings,
    normalize_ngz_inputs,
    compose_ngz_into_graph,
    build_ngz_collision_geom,
    detect_inside_ngz,
    apply_ngz_mode,
    split_polygon_at_antimeridian,
)

__all__ = [
    "RoutingMapConfig",
    "build_aoi",
    "NgzRingBuildConfig",
    "NgzInput",
    "NgzGroup",
    "NgzRingResult",
    "NgzOverlay",
    "NgzInsideError",
    "build_ngz_overlay",
    "build_ngz_t_rings",
    "normalize_ngz_inputs",
    "compose_ngz_into_graph",
    "build_ngz_collision_geom",
    "detect_inside_ngz",
    "apply_ngz_mode",
    "split_polygon_at_antimeridian",
]
