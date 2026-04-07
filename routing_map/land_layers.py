from __future__ import annotations
from typing import Dict, List

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

try:
    # shapely 2.x
    from shapely import union_all, set_precision
except Exception:  # pragma: no cover
    union_all = None
    set_precision = None

try:
    from shapely.validation import make_valid
except Exception:  # pragma: no cover
    make_valid = None

from .geom_utils import AOIProjector, geom_to_m

def build_land_layers(
    polys_ll: List,
    proj: AOIProjector,
    *,
    buffer_km: float,
    avoid_km: float,
    collision_safety_km: float,
    grid_size_m: float = 5.0,
) -> Dict[str, object]:
    """Build union/ring/collision layers in meters.

    Outputs:
      UNION_M: land union (meters)
      AVOID_M: land buffered by avoid_km
      COLLISION_M: land buffered by avoid_km+collision_safety_km
      RING_M: boundary ring (LineString/MultiLineString)
      TARGET_RING_M: boundary of AVOID_M
    """
    polys_m = []
    for g in polys_ll:
        if g is None or g.is_empty:
            continue
        gg = geom_to_m(g, proj)
        if make_valid is not None:
            gg = make_valid(gg)
        gg = gg.buffer(0)
        if set_precision is not None:
            gg = set_precision(gg, grid_size_m)
        if not gg.is_empty:
            polys_m.append(gg)

    if not polys_m:
        empty = Polygon()
        return {"UNION_M": empty, "AVOID_M": empty, "COLLISION_M": empty, "RING_M": empty.boundary, "TARGET_RING_M": empty.boundary}

    if union_all is not None:
        union_m = union_all(polys_m)
    else:
        union_m = unary_union(polys_m)

    # buffers in meters
    buffer_m = buffer_km * 1000.0
    avoid_m = avoid_km * 1000.0
    safety_m = collision_safety_km * 1000.0

    # UNION_M is raw union; AVOID_M used for ring extraction
    union_m = union_m.buffer(0)
    avoid_union_m = union_m.buffer(avoid_m).buffer(0)
    collision_m = union_m.buffer(avoid_m + safety_m).buffer(0)

    ring_m = union_m.boundary
    target_ring_m = avoid_union_m.boundary

    return {
        "UNION_M": union_m,
        "AVOID_M": avoid_union_m,
        "COLLISION_M": collision_m,
        "RING_M": ring_m,
        "TARGET_RING_M": target_ring_m,
    }
