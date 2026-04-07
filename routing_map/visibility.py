from __future__ import annotations
from shapely.geometry import LineString
from shapely.prepared import PreparedGeometry

def segment_clear(a_xy, b_xy, *, collision_prep: PreparedGeometry) -> bool:
    """Return True if segment AB does NOT intersect collision geometry."""
    seg = LineString([(float(a_xy[0]), float(a_xy[1])), (float(b_xy[0]), float(b_xy[1]))])
    return not collision_prep.intersects(seg)
