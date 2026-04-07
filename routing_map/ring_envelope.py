from __future__ import annotations

from typing import Any, Dict, List, Tuple

import math

from shapely.geometry import LineString, Point, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, nearest_points

from .ring_types import RingBuildConfig, RingResult, XY


def _iter_polygons(geom) -> List[Polygon]:
    """Robustly iterate polygons from Polygon/MultiPolygon/GeometryCollection."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        polys: List[Polygon] = []
        for g in geom.geoms:
            polys.extend(_iter_polygons(g))
        return polys
    return []


def build_envelope_polys_m(land_union_m: Any, *, clearance_m: float) -> List[Polygon]:
    """
    Build envelope polygons by buffering land_union outward by clearance_m.
    land_union_m is expected in metric CRS.
    """
    if land_union_m is None or land_union_m.is_empty:
        return []
    env = land_union_m.buffer(float(clearance_m)).buffer(0)
    return _iter_polygons(env)


def extract_exterior_lines(polys_m: List[Polygon]) -> List[LineString]:
    """Return exterior rings as LineStrings (closed)."""
    lines: List[LineString] = []
    for p in polys_m:
        if p.is_empty:
            continue
        lines.append(LineString(p.exterior.coords))
    return lines


def _sample_line_by_step(line_m: LineString, step_m: float) -> List[XY]:
    """
    Sample a (closed) LineString approximately every step_m along its length.
    Ensures closure (first == last).
    """
    step_m = float(step_m)
    if step_m <= 0:
        raise ValueError("step_m must be > 0")

    L = float(line_m.length)
    if L <= 0:
        return []

    n = max(4, int(math.ceil(L / step_m)))
    pts: List[XY] = []
    for i in range(n):
        d = (i / n) * L
        p = line_m.interpolate(d)
        pts.append((float(p.x), float(p.y)))

    # close
    if pts and (pts[0] != pts[-1]):
        pts.append(pts[0])

    return pts


def sample_ring_lines_m(lines_m: List[LineString], *, step_m: float) -> List[List[XY]]:
    """Sample each exterior line into closed point lists."""
    out: List[List[XY]] = []
    for ln in lines_m:
        pts = _sample_line_by_step(ln, step_m=step_m)
        if len(pts) >= 4:
            out.append(pts)
    return out


def _is_point_inside_collision(pt: XY, collision_geom: Any) -> bool:
    if collision_geom is None or getattr(collision_geom, "is_empty", True):
        return False
    return bool(Point(pt).within(collision_geom))


def fix_ring_points_outside_collision(
    pts_m: List[XY],
    *,
    collision_geom: Any,
    cfg: RingBuildConfig,
) -> Tuple[List[XY], int]:
    """
    If a sampled ring point is inside collision_geom, push it outward.

    Strategy (simple & safe):
    - For a bad point p, find nearest point q on collision boundary.
    - Push along vector (p - q). If degenerate, push along a small fixed axis.
    - Iterate until outside or max iter reached.
    """
    if collision_geom is None or getattr(collision_geom, "is_empty", True):
        return pts_m, 0
    if not pts_m:
        return pts_m, 0

    fixed = []
    n_fixed = 0
    step = float(cfg.point_fix_step_m)
    max_iter = int(cfg.point_fix_max_iter)

    # Use boundary for stable nearest-point direction
    boundary = getattr(collision_geom, "boundary", collision_geom)

    for (x, y) in pts_m:
        p = (float(x), float(y))
        if not _is_point_inside_collision(p, collision_geom):
            fixed.append(p)
            continue

        n_fixed += 1
        cur = Point(p)

        for _ in range(max_iter):
            # nearest points between boundary and the point
            q_geom, p_geom = nearest_points(boundary, cur)
            qx, qy = float(q_geom.x), float(q_geom.y)
            px, py = float(cur.x), float(cur.y)
            vx, vy = px - qx, py - qy
            norm = math.hypot(vx, vy)
            if norm < 1e-9:
                # degenerate: pick a deterministic direction
                vx, vy = 1.0, 0.0
                norm = 1.0
            ux, uy = vx / norm, vy / norm
            cur = Point(px + ux * step, py + uy * step)
            if not cur.within(collision_geom):
                break

        fixed.append((float(cur.x), float(cur.y)))

    # re-close if needed
    if fixed and (fixed[0] != fixed[-1]):
        fixed[-1] = fixed[0]

    return fixed, n_fixed


def build_envelope_rings_m(
    land_union_m: Any,
    *,
    collision_hard_m: Any,
    cfg: RingBuildConfig,
) -> Tuple[Any, List[RingResult]]:
    """
    Build envelope rings (no taut yet).

    Returns:
      ring_base_m: union of envelope polygons (viz)
      rings: list[RingResult] with envelope_* filled
    """
    env_polys = build_envelope_polys_m(land_union_m, clearance_m=cfg.clearance_m)
    if not env_polys:
        return Polygon(), []

    # Stats & filtering
    rows: List[RingResult] = []
    ring_id = 0

    # Determine "mainland" per connected component by max area
    max_area = max(float(p.area) for p in env_polys) if env_polys else 0.0

    lines = extract_exterior_lines(env_polys)
    sampled = sample_ring_lines_m(lines, step_m=float(cfg.ring_sample_km) * 1000.0)

    for poly, pts in zip(env_polys, sampled):
        if poly.is_empty or len(pts) < 4:
            continue
        area_km2 = float(poly.area) / 1e6
        length_km = float(LineString(pts).length) / 1000.0

        if area_km2 < float(cfg.min_island_area_km2):
            continue
        if length_km < float(cfg.min_ring_length_km):
            continue

        pts_fixed, n_fixed = fix_ring_points_outside_collision(
            pts, collision_geom=collision_hard_m, cfg=cfg
        )

        res = RingResult(
            ring_id=ring_id,
            is_mainland=bool(float(poly.area) == max_area),
            envelope_poly_m=poly,
            envelope_pts_m=pts_fixed,
            taut_pts_m=[],
            stats={
                "area_km2": area_km2,
                "length_km_envelope": length_km,
                "n_pts_envelope": len(pts_fixed),
                "n_pts_fixed": int(n_fixed),
            },
        )
        rows.append(res)
        ring_id += 1

    ring_base_m = unary_union(env_polys).buffer(0) if env_polys else Polygon()
    return ring_base_m, rows
