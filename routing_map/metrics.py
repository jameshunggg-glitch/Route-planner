# routing_map/metrics.py
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Optional

from routing_map.routing_graph import haversine_km

LonLat = Tuple[float, float]
NM_PER_KM = 1.0 / 1.852  # 1 nm = 1.852 km


def _unwrap_lon(lon: float, ref_lon: float) -> float:
    """Unwrap lon to be closest to ref_lon (handles dateline crossings)."""
    lon = float(lon)
    ref_lon = float(ref_lon)
    d = lon - ref_lon
    if d > 180.0:
        lon -= 360.0
    elif d < -180.0:
        lon += 360.0
    return lon


def _dateline_unwrap_path(path_ll: Sequence[LonLat]) -> List[LonLat]:
    if not path_ll:
        return []
    out: List[LonLat] = []
    ref = float(path_ll[0][0])
    for lon, lat in path_ll:
        lon_u = _unwrap_lon(float(lon), ref)
        out.append((lon_u, float(lat)))
        ref = lon_u
    return out


def path_length_km_nm(
    path_ll: Sequence[LonLat],
    *,
    dateline_unwrap: bool = True,
) -> Tuple[float, float]:
    """
    Compute total path length for a lon/lat polyline.

    Args:
      path_ll: [(lon, lat), ...]
      dateline_unwrap: if True, unwrap longitudes to avoid dateline artifacts.

    Returns:
      (total_km, total_nm)
    """
    if path_ll is None:
        return 0.0, 0.0

    pts = [(float(p[0]), float(p[1])) for p in path_ll if p is not None]
    if len(pts) < 2:
        return 0.0, 0.0

    if dateline_unwrap:
        pts = _dateline_unwrap_path(pts)

    total_km = 0.0
    for a, b in zip(pts, pts[1:]):
        # haversine_km expects (lon,lat)
        total_km += float(haversine_km(a, b))

    total_nm = total_km * NM_PER_KM
    return total_km, total_nm


def format_distance(km: float, nm: float, *, digits: int = 2) -> str:
    """Pretty string like: '1234.56 km | 666.67 nm'"""
    return f"{km:.{digits}f} km | {nm:.{digits}f} nm"
