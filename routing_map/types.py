from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Iterable, Any

LonLat = Tuple[float, float]          # (lon, lat)
BBoxLL = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)

@dataclass(frozen=True)
class XY:
    x_m: float
    y_m: float
