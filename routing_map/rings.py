from __future__ import annotations

from typing import List, Tuple, Any, Optional
import pandas as pd

from shapely.geometry import Polygon, MultiPolygon, LineString, GeometryCollection
from shapely.ops import unary_union

# ---- existing helper ----
def _iter_polygons(geom) -> List[Polygon]:
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


# =============================================================================
# Legacy: envelope-only rings (kept as-is for backwards compatibility)
# =============================================================================
def build_coast_rings_smooth_v2(
    union_smooth_m,
    *,
    avoid_km: float,
    island_area_min_km2: float,
) -> tuple[object, List[LineString], pd.DataFrame]:
    """
    Build coast rings from A2 smooth union (meters).

    - Buffer outward by avoid_km to create offset ring polygons.
    - Keep exterior rings, drop small islands (< island_area_min_km2).
    Returns:
      ring_base_m: union of ring polygons (for viz)
      rings_m: list of LineString exteriors (meters)
      rings_df: per-ring stats
    """
    avoid_m = float(avoid_km) * 1000.0
    polys = _iter_polygons(union_smooth_m)

    if not polys:
        empty = Polygon()
        return empty, [], pd.DataFrame(
            columns=["ring_id", "area_km2", "length_km", "is_mainland", "original_area_km2"]
        )

    all_rings: List[LineString] = []
    rows = []
    ring_id = 0

    for p in polys:
        original_area_km2 = float(p.area) / 1e6

        # offset outwards then clean
        rp = p.buffer(avoid_m).buffer(0)
        if rp.is_empty:
            continue

        rp_polys = _iter_polygons(rp)
        if not rp_polys:
            continue

        max_area = max(float(pp.area) for pp in rp_polys)

        for pp in rp_polys:
            area_km2 = float(pp.area) / 1e6
            if area_km2 < float(island_area_min_km2):
                continue

            length_km = float(pp.exterior.length) / 1000.0
            is_mainland = (float(pp.area) == max_area)

            all_rings.append(pp.exterior)
            rows.append(
                {
                    "ring_id": ring_id,
                    "area_km2": area_km2,
                    "length_km": length_km,
                    "is_mainland": bool(is_mainland),
                    "original_area_km2": original_area_km2,
                }
            )
            ring_id += 1

    if all_rings:
        ring_base_m = unary_union([Polygon(r) for r in all_rings]).buffer(0)
    else:
        ring_base_m = Polygon()

    rings_df = pd.DataFrame(rows)
    return ring_base_m, all_rings, rings_df


# =============================================================================
# New: envelope + taut rings (v1)
# =============================================================================
from .ring_types import RingBuildConfig, RingResult
from .ring_envelope import build_envelope_rings_m
from .ring_taut import taut_simplify_closed_ring


def build_envelope_and_taut_rings_v1(
    land_union_m,
    *,
    collision_hard_m,
    cfg: Optional[RingBuildConfig] = None,
) -> tuple[Any, List[LineString], List[LineString], pd.DataFrame, List[RingResult]]:
    """
    Build rings in 2 phases:
      1) envelope ring: buffer( clearance_m ) and sample exteriors
      2) taut ring: visibility simplification on envelope points (cycle-aware)

    Returns:
      ring_base_m: union of envelope polygons (viz)
      envelope_lines_m: list of LineString (meters) from envelope points
      taut_lines_m: list of LineString (meters) from taut points
      rings_df: per-ring stats including taut outcome
      rings: list[RingResult] for integration steps
    """
    if cfg is None:
        cfg = RingBuildConfig()

        # collision used by taut checks (configurable)
    collision_taut_m = None
    if collision_hard_m is not None and (not getattr(collision_hard_m, "is_empty", True)):
        if getattr(cfg, "taut_use_clearance_buffer", True):
            buf_m = cfg.taut_collision_buffer_m
            print(buf_m)
            if buf_m is None:
                buf_m = cfg.clearance_m
                print(buf_m)
            collision_taut_m = collision_hard_m.buffer(float(buf_m)).buffer(0)
        else:
            # use original collision as-is
            collision_taut_m = collision_hard_m



    ring_base_m, rings = build_envelope_rings_m(
        land_union_m,
        collision_hard_m=collision_hard_m,
        cfg=cfg,
    )

    envelope_lines: List[LineString] = []
    taut_lines: List[LineString] = []
    rows = []

    for r in rings:
        env_pts = r.envelope_pts_m
        envelope_lines.append(LineString(env_pts))

        taut_pts, taut_stats = taut_simplify_closed_ring(
            env_pts, collision_taut_m=collision_taut_m, collision_hard_m=collision_hard_m,cfg=cfg
        )
        r.taut_pts_m = taut_pts
        r.stats.update(
            {
                "taut_ok": bool(taut_stats.get("ok", False)),
                "taut_reason": taut_stats.get("reason", ""),
                "n_pts_taut": int(len(taut_pts)) if taut_pts else 0,
            }
        )
        if taut_pts:
            r.stats["length_km_taut"] = float(LineString(taut_pts).length) / 1000.0
        else:
            r.stats["length_km_taut"] = 0.0

        taut_lines.append(LineString(taut_pts if taut_pts else env_pts))

        rows.append(
            {
                "ring_id": r.ring_id,
                "is_mainland": r.is_mainland,
                "area_km2": float(r.stats.get("area_km2", 0.0)),
                "length_km_envelope": float(r.stats.get("length_km_envelope", 0.0)),
                "length_km_taut": float(r.stats.get("length_km_taut", 0.0)),
                "n_pts_envelope": int(r.stats.get("n_pts_envelope", 0)),
                "n_pts_taut": int(r.stats.get("n_pts_taut", 0)),
                "n_pts_fixed": int(r.stats.get("n_pts_fixed", 0)),
                "taut_ok": bool(r.stats.get("taut_ok", False)),
                "taut_reason": str(r.stats.get("taut_reason", "")),
            }
        )

    rings_df = pd.DataFrame(rows)
    return ring_base_m, envelope_lines, taut_lines, rings_df, rings
