"""routing_map: modular coastal/sea routing builder.

This package is a refactor template extracted from the notebook `route_planner_point_to_port.ipynb`.
Use `build_aoi.build_aoi()` as the main entrypoint for AOI runs.
"""

from .config import RoutingMapConfig
from .build_aoi import build_aoi

__all__ = ["RoutingMapConfig", "build_aoi"]
