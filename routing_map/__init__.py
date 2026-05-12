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
    NgzPatchUnreachableError,
    BlockedSubpath,
    apply_ngz_mode,
    apply_patches_to_baseline,
    build_local_visibility_graph,
    build_ngz_collision_geom,
    build_ngz_overlay_lite,
    build_ngz_t_rings,
    detect_blocked_subpaths,
    detect_inside_ngz,
    normalize_ngz_inputs,
    solve_local_patch,
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
    "NgzPatchUnreachableError",
    "BlockedSubpath",
    "apply_ngz_mode",
    "apply_patches_to_baseline",
    "build_local_visibility_graph",
    "build_ngz_collision_geom",
    "build_ngz_overlay_lite",
    "build_ngz_t_rings",
    "detect_blocked_subpaths",
    "detect_inside_ngz",
    "normalize_ngz_inputs",
    "solve_local_patch",
    "split_polygon_at_antimeridian",
]
