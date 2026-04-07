from __future__ import annotations
from shapely.ops import unary_union

def smooth_union_for_features_from_union(union_m, *, a2_smooth_km: float, a2_tol_km: float):
    """A2 smooth union used for feature extraction.

    Matches notebook behavior:
      - buffer(+a2_smooth) then buffer(-a2_smooth)
      - simplify(tol) and buffer(0)
    """
    s = a2_smooth_km * 1000.0
    tol = a2_tol_km * 1000.0
    sm = union_m.buffer(s).buffer(-s)
    sm = sm.simplify(tol, preserve_topology=True)
    sm = sm.buffer(0)
    return sm
