"""
routing_map.path_simplifier

Greedy visibility-based path simplification that preserves collision-free constraint.

Designed for routing_map pipelines where you already have:
- a polyline path in lon/lat: [(lon, lat), ...]
- a land/avoid "collision" geometry in metric CRS (meters), ideally prepared for fast intersects
- a projection helper (lon/lat <-> meters)

Author: ChatGPT (generated)
"""
from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any

try:
    from shapely.geometry import LineString
    from shapely.prepared import prep as shapely_prep
except Exception as e:  # pragma: no cover
    LineString = None
    shapely_prep = None


LonLat = Tuple[float, float]
XY = Tuple[float, float]


@dataclass
class SimplifyStats:
    n_in: int
    n_out: int
    ratio: float
    n_checks: int
    n_blocked: int
    n_skipped: int
    strategy: str
    window_size: int
    max_tries: int


def _wrap_lon(lon: float) -> float:
    # normalize to (-180, 180]
    x = (lon + 180.0) % 360.0 - 180.0
    # keep +180 instead of -180 if you care; here we keep -180
    return x


def _unwrap_lon(lon: float, ref_lon: float) -> float:
    """Adjust lon to the equivalent value (±360) closest to ref_lon."""
    lon = float(lon); ref_lon = float(ref_lon)
    d = lon - ref_lon
    if d > 180.0:
        lon -= 360.0
    elif d < -180.0:
        lon += 360.0
    return lon


def _unwrap_path_ll(path_ll: Sequence[LonLat]) -> List[LonLat]:
    if not path_ll:
        return []
    out: List[LonLat] = []
    ref = float(path_ll[0][0])
    for lon, lat in path_ll:
        lon_u = _unwrap_lon(float(lon), ref)
        out.append((lon_u, float(lat)))
        ref = lon_u
    return out


# --- Great-circle (spherical) segment densification ---
# Used for collision checks so that "skip" decisions reflect great-circle routing, not planar straight lines.
_EARTH_R_KM = 6371.0088
_GC_STEP_KM_DEFAULT = 10.0  # fixed step to avoid changing public function signatures


def _gc_densify_ll(p0: LonLat, p1: LonLat, *, step_km: float = _GC_STEP_KM_DEFAULT) -> List[LonLat]:
    """Densify the great-circle path from p0 to p1 into a polyline in lon/lat.

    Notes
    -----
    - Assumes longitudes are already continuous if you enabled dateline_unwrap upstream.
    - Returns a list including both endpoints.
    """

    lon0, lat0 = float(p0[0]), float(p0[1])
    lon1, lat1 = float(p1[0]), float(p1[1])

    # Convert to unit vectors on the sphere
    def _to_vec(lon_deg: float, lat_deg: float) -> Tuple[float, float, float]:
        lon = math.radians(lon_deg)
        lat = math.radians(lat_deg)
        clat = math.cos(lat)
        return (clat * math.cos(lon), clat * math.sin(lon), math.sin(lat))

    def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _norm(a: Tuple[float, float, float]) -> float:
        return math.sqrt(_dot(a, a))

    def _scale(a: Tuple[float, float, float], s: float) -> Tuple[float, float, float]:
        return (a[0] * s, a[1] * s, a[2] * s)

    def _add(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def _unit(a: Tuple[float, float, float]) -> Tuple[float, float, float]:
        n = _norm(a)
        if n <= 0.0:
            return (0.0, 0.0, 0.0)
        return (a[0] / n, a[1] / n, a[2] / n)

    def _to_lonlat(v: Tuple[float, float, float]) -> LonLat:
        x, y, z = v
        lon = math.degrees(math.atan2(y, x))
        hyp = math.hypot(x, y)
        lat = math.degrees(math.atan2(z, hyp))
        return (lon, lat)

    u = _to_vec(lon0, lat0)
    v = _to_vec(lon1, lat1)

    # Central angle
    dot_uv = max(-1.0, min(1.0, _dot(u, v)))
    omega = math.acos(dot_uv)
    dist_km = _EARTH_R_KM * omega

    if step_km <= 0.0 or dist_km <= step_km or omega < 1e-12:
        return [(lon0, lat0), (lon1, lat1)]

    n_seg = max(1, int(math.ceil(dist_km / step_km)))
    so = math.sin(omega)

    out: List[LonLat] = []
    ref_lon = lon0
    for k in range(n_seg + 1):
        t = k / n_seg
        a = math.sin((1.0 - t) * omega) / so
        b = math.sin(t * omega) / so
        w = _add(_scale(u, a), _scale(v, b))
        w = _unit(w)
        lon, lat = _to_lonlat(w)
        # keep longitude continuous w.r.t the previous point
        lon = _unwrap_lon(lon, ref_lon)
        ref_lon = lon
        out.append((lon, lat))

    # Ensure exact endpoints (preserve original inputs)
    out[0] = (lon0, lat0)
    out[-1] = (lon1, lat1)
    return out



def _coalesce_consecutive_duplicates(path: Sequence[LonLat], *, eps: float = 0.0) -> List[LonLat]:
    """Drop consecutive duplicates (and near-duplicates if eps > 0)."""
    if not path:
        return []
    out = [tuple(map(float, path[0]))]  # type: ignore
    for lon, lat in path[1:]:
        lon = float(lon); lat = float(lat)
        plon, plat = out[-1]
        if eps <= 0.0:
            if lon == plon and lat == plat:
                continue
        else:
            if abs(lon - plon) <= eps and abs(lat - plat) <= eps:
                continue
        out.append((lon, lat))
    return out


def _make_projectors(
    proj: Any,
    ll_to_m: Optional[Callable[[LonLat], XY]] = None,
    m_to_ll: Optional[Callable[[XY], LonLat]] = None,
) -> Tuple[Callable[[LonLat], XY], Callable[[XY], LonLat]]:
    """
    Accepts either:
    - explicit callables ll_to_m / m_to_ll, OR
    - a proj object with methods like:
        * proj.ll_to_m(lon, lat) -> (x, y)  OR proj.fwd((lon,lat))
        * proj.m_to_ll(x, y) -> (lon, lat)  OR proj.inv((x,y))
        * proj.forward / proj.inverse
    """
    if ll_to_m and m_to_ll:
        return ll_to_m, m_to_ll

    # common method names in custom codebases
    candidates = [
        ("ll_to_m", "m_to_ll"),
        ("to_m", "to_ll"),
        ("fwd", "inv"),
        ("forward", "inverse"),
    ]

    def _apply_xy(fn: Any, x: float, y: float, xy_tuple: Tuple[float, float]):
        """Robustly apply a forward/inverse projector.

        Handles common patterns:
        - callable(lon, lat) -> (x, y)
        - callable((lon, lat)) -> (x, y)
        - pyproj.Transformer-like object with .transform(lon, lat)
        """
        # pyproj Transformer-like
        if hasattr(fn, "transform") and callable(getattr(fn, "transform")):
            return fn.transform(x, y)

        if callable(fn):
            try:
                return fn(x, y)
            except TypeError:
                return fn(xy_tuple)

        # if it's not callable but has a transform method, try that
        raise TypeError(f"Projector of type {type(fn)} is not callable and has no .transform")
    for a, b in candidates:
        if hasattr(proj, a) and hasattr(proj, b):
            f = getattr(proj, a)
            g = getattr(proj, b)

            def _ll2m(p: LonLat) -> XY:
                lon, lat = p
                r = _apply_xy(f, lon, lat, (lon, lat))
                return (float(r[0]), float(r[1]))

            def _m2ll(q: XY) -> LonLat:
                x, y = q
                r = _apply_xy(g, x, y, (x, y))
                return (float(r[0]), float(r[1]))

            return _ll2m, _m2ll

    # allow pyproj Transformer-like .transform
    if hasattr(proj, "transform"):
        tr = getattr(proj, "transform")

        def _ll2m(p: LonLat) -> XY:
            lon, lat = p
            x, y = tr(lon, lat)
            return (float(x), float(y))

        # Try to use pyproj's inverse direction if available
        def _m2ll(q: XY) -> LonLat:
            x, y = q
            try:
                # pyproj>=3 supports direction keyword
                from pyproj.enums import TransformDirection  # type: ignore

                lon, lat = tr(x, y, direction=TransformDirection.INVERSE)
                return (float(lon), float(lat))
            except Exception:
                raise ValueError(
                    "proj.transform found but inverse is unavailable; please pass m_to_ll explicitly."
                )

        return _ll2m, _m2ll

    raise ValueError("Cannot infer projection functions. Pass ll_to_m and m_to_ll, or provide a proj with ll_to_m/m_to_ll.")


def simplify_path_visibility(
    path_ll: Sequence[LonLat],
    *,
    collision_m: Any,
    proj: Any = None,
    ll_to_m: Optional[Callable[[LonLat], XY]] = None,
    m_to_ll: Optional[Callable[[XY], LonLat]] = None,
    window_size: int = 80,
    max_tries: int = 300,
    use_prepared_collision: bool = True,
    dateline_unwrap: bool = True,
    wrap_output_lon: bool = True,
    strategy: str = "linear_backscan",
) -> Tuple[List[LonLat], SimplifyStats]:
    """
    Greedy visibility simplification.

    Parameters
    ----------
    path_ll : [(lon,lat), ...]
        Original polyline in lon/lat.
    collision_m : shapely geometry (metric CRS, meters) OR prepared geometry
        Collision geometry (e.g., buffered land) in the SAME metric CRS as your projection.
        If not prepared, set use_prepared_collision=True to auto-prep.
    proj / ll_to_m / m_to_ll
        Projection helpers. Provide either proj (with methods) or explicit callables.
    window_size : int
        Max look-ahead candidates from each i.
    max_tries : int
        Max collision checks per i (cap complexity).
    use_prepared_collision : bool
        If True and collision_m is not prepared, prepare it.
    dateline_unwrap : bool
        If True, unwrap longitudes before projection to avoid 179 -> -179 huge jumps.
    wrap_output_lon : bool
        Normalize output longitudes back to (-180, 180] after inverse-projection.
    strategy : str
        Currently: "linear_backscan" (try farthest j, scan backward until feasible).

    Returns
    -------
    simplified_ll, stats
    """
    if LineString is None:
        raise ImportError("shapely is required for simplify_path_visibility")

    path0 = _coalesce_consecutive_duplicates(list(path_ll))
    if len(path0) <= 2:
        out = list(path0)
        if wrap_output_lon:
            out = [(_wrap_lon(lon), lat) for lon, lat in out]
        stats = SimplifyStats(n_in=len(path0), n_out=len(out), ratio=(len(out)/max(1,len(path0))),
                              n_checks=0, n_blocked=0, n_skipped=0,
                              strategy=strategy, window_size=int(window_size), max_tries=int(max_tries))
        return out, stats

    path_use = _unwrap_path_ll(path0) if dateline_unwrap else list(path0)

    ll2m, m2ll = _make_projectors(proj, ll_to_m=ll_to_m, m_to_ll=m_to_ll)

    # collision prepared
    col = collision_m
    if use_prepared_collision and shapely_prep is not None:
        # detect already prepared: prepared geometry has .context
        if hasattr(col, "context"):
            col_prep = col
        else:
            col_prep = shapely_prep(col)
    else:
        col_prep = col

    # project
    path_m: List[XY] = [ll2m(p) for p in path_use]
    n = len(path_m)

    keep_idx: List[int] = [0]
    i = 0

    n_checks = 0
    n_blocked = 0
    n_skipped = 0

    def seg_ok(i_: int, j_: int) -> bool:
        nonlocal n_checks, n_blocked
        n_checks += 1
        # Great-circle collision check: densify (lon/lat) along GC, then project and test.
        gc_ll = _gc_densify_ll(path_use[i_], path_use[j_])
        gc_m = [ll2m(p) for p in gc_ll]
        ls = LineString(gc_m)
        ok = not col_prep.intersects(ls)
        if not ok:
            n_blocked += 1
        return ok

    while i < n - 1:
        j_max = min(n - 1, i + max(1, int(window_size)))
        chosen = None

        if strategy != "linear_backscan":
            # fallback to linear_backscan
            strategy = "linear_backscan"

        # try farthest, then back-scan
        tries = 0
        j = j_max
        while j > i:
            if tries >= max_tries:
                break
            tries += 1
            if seg_ok(i, j):
                chosen = j
                break
            j -= 1

        if chosen is None:
            # worst case: cannot skip anything -> keep next point
            chosen = i + 1
            n_skipped += 1

        keep_idx.append(chosen)
        i = chosen

    # rebuild lonlat
    simplified_m = [path_m[k] for k in keep_idx]
    simplified_ll: List[LonLat] = [m2ll(xy) for xy in simplified_m]

    if wrap_output_lon:
        simplified_ll = [(_wrap_lon(lon), lat) for lon, lat in simplified_ll]

    stats = SimplifyStats(
        n_in=len(path0),
        n_out=len(simplified_ll),
        ratio=len(simplified_ll) / max(1, len(path0)),
        n_checks=n_checks,
        n_blocked=n_blocked,
        n_skipped=n_skipped,
        strategy=strategy,
        window_size=int(window_size),
        max_tries=int(max_tries),
    )
    return simplified_ll, stats
