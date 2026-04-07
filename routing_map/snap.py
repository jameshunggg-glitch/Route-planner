# routing_map/snap.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.prepared import prep
from shapely.ops import nearest_points

from .routing_graph import haversine_km, L_INJECT, B_HIGH_LAT
from .geom_utils import wrap_lon, unwrap_lon as _unwrap_lon_ref, coord_id

LonLat = Tuple[float, float]


# -------------------------
# Small utilities
# -------------------------
def normalize_lonlat(p: LonLat) -> LonLat:
    """Normalize lon into [-180,180) range."""
    lon, lat = float(p[0]), float(p[1])
    return (wrap_lon(lon), float(lat))


def unwrap_lon(lon: float, ref_lon: float) -> float:
    return float(_unwrap_lon_ref(float(lon), float(ref_lon)))


def bearing_deg(a: LonLat, b: LonLat) -> float:
    """Initial bearing from a->b in degrees [0,360)."""
    lon1, lat1 = np.deg2rad(float(a[0])), np.deg2rad(float(a[1]))
    lon2, lat2 = np.deg2rad(float(b[0])), np.deg2rad(float(b[1]))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.rad2deg(np.arctan2(y, x))
    brng = (brng + 360.0) % 360.0
    return float(brng)


def ang_diff_deg(a_deg: float, b_deg: float) -> float:
    """Smallest absolute difference between two angles (deg) in [0,180]."""
    d = (float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0
    return float(abs(d))


def _get_proj(out: Dict[str, Any]):
    proj = out.get("proj", None)
    if proj is None:
        raise ValueError("out['proj'] is required for snap (AOIProjector).")
    return proj


def _get_collision_geom_m(out: Dict[str, Any]):
    layers = out.get("layers", None)
    if not isinstance(layers, dict) or "COLLISION_M" not in layers:
        return None
    return layers["COLLISION_M"]


def _point_ll_to_m(out: Dict[str, Any], p_ll: LonLat) -> Tuple[float, float]:
    proj = _get_proj(out)
    x_m, y_m = proj.to_m.transform(float(p_ll[0]), float(p_ll[1]))
    return float(x_m), float(y_m)


def _point_m_to_ll(out: Dict[str, Any], p_m: Tuple[float, float]) -> LonLat:
    proj = _get_proj(out)
    lon, lat = proj.to_ll.transform(float(p_m[0]), float(p_m[1]))
    return (float(lon), float(lat))


def _is_in_collision(out: Dict[str, Any], p_ll: LonLat, collision_prep=None) -> bool:
    """Check if lon/lat point falls in collision geometry (meters)."""
    geom_m = _get_collision_geom_m(out)
    if geom_m is None:
        return False
    if collision_prep is None:
        collision_prep = prep(geom_m)
    x, y = _point_ll_to_m(out, p_ll)
    pt = Point(x, y)
    return bool(collision_prep.contains(pt) or collision_prep.intersects(pt))


def _guess_nudge_buffer_m(out: Dict[str, Any], default_km: float = 0.5) -> float:
    """Use cfg.land.collision_safety_km if present, else default."""
    cfg = out.get("cfg", None)
    try:
        km = float(getattr(getattr(cfg, "land"), "collision_safety_km"))
        if km > 0:
            return km * 1000.0
    except Exception:
        pass
    return float(default_km) * 1000.0


def _kdt_query_indices(kdt, x: float, y: float, k: int):
    """
    Return list of indices for k nearest points.
    Compatible with sklearn.neighbors.KDTree and scipy.spatial.cKDTree.
    """
    k = max(1, int(k))
    try:
        # sklearn KDTree
        dists, idxs = kdt.query(np.array([[x, y]], dtype=float), k=k)
        return idxs[0].tolist()
    except Exception:
        # scipy cKDTree
        dists, idxs = kdt.query([x, y], k=k)
        idxs = np.atleast_1d(idxs)
        return [int(i) for i in idxs.tolist()]


# -------------------------
# Sea adjacency for local entrance augmentation
# -------------------------
def _get_or_build_sea_adjacency(out: Dict[str, Any]) -> Dict[int, List[int]]:
    """
    Build adjacency list for S_nodes indices using out["S_edges"].

    Supports S_edges endpoints as:
      - (i, j) integer indices
      - ((lon,lat),(lon,lat)) tuples
      - ("lon,lat","lon,lat") strings
    """
    if isinstance(out.get("sea_adj", None), dict):
        return out["sea_adj"]

    S_nodes = out.get("S_nodes", None)
    S_edges = out.get("S_edges", None)
    if not isinstance(S_nodes, pd.DataFrame) or S_edges is None:
        out["sea_adj"] = {}
        return out["sea_adj"]

    n = len(S_nodes)
    # map lon/lat -> idx (rounded to match edge precision)
    ll2idx = {}
    for i, r in S_nodes.reset_index(drop=True).iterrows():
        try:
            key = (round(float(r["lon"]), 6), round(float(r["lat"]), 6))
            ll2idx[key] = int(i)
        except Exception:
            continue

    def parse_ll(obj):
        # ((lon,lat),) or (lon,lat)
        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            try:
                return (float(obj[0]), float(obj[1]))
            except Exception:
                return None
        # "lon,lat"
        if isinstance(obj, str) and "," in obj:
            try:
                a, b = obj.split(",")
                return (float(a), float(b))
            except Exception:
                return None
        return None

    adj: Dict[int, List[int]] = {}

    def add(u: int, v: int):
        adj.setdefault(u, []).append(v)

    for e in S_edges:
        try:
            if not isinstance(e, (list, tuple)) or len(e) < 2:
                continue
            a, b = e[0], e[1]

            # case1: indices
            if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
                u, v = int(a), int(b)
            else:
                # case2: lonlat endpoints
                a_ll = parse_ll(a)
                b_ll = parse_ll(b)
                if a_ll is None or b_ll is None:
                    continue
                ua = ll2idx.get((round(a_ll[0], 6), round(a_ll[1], 6)))
                vb = ll2idx.get((round(b_ll[0], 6), round(b_ll[1], 6)))
                if ua is None or vb is None:
                    continue
                u, v = int(ua), int(vb)

            if u < 0 or v < 0 or u >= n or v >= n:
                continue
            add(u, v)
            add(v, u)
        except Exception:
            continue

    out["sea_adj"] = adj
    return adj


def _interp_lonlat(a: LonLat, b: LonLat, t: float) -> LonLat:
    """Linear interpolation in lon/lat with simple dateline-safe lon unwrap."""
    lon1, lat1 = float(a[0]), float(a[1])
    lon2, lat2 = float(b[0]), float(b[1])
    lon2u = unwrap_lon(lon2, lon1)
    lon = lon1 + float(t) * (lon2u - lon1)
    lat = lat1 + float(t) * (lat2 - lat1)
    return normalize_lonlat((lon, lat))


def _virtual_candidates_from_seed_node(
    out: Dict[str, Any],
    *,
    seed_idx: int,
    p_used_ll: LonLat,
    component: Optional[int],
    max_neighbors: int = 12,
    t_samples: Sequence[float] = (0.5, 1.0 / 3.0, 2.0 / 3.0),
    round_ndigits: int = 6,
) -> List["SnapCandidate"]:
    """
    Create virtual sea candidates by sampling points along sea edges adjacent to seed_idx.
    These points are ONLY used for injection; they do not exist in the base sea graph.
    """
    S_nodes: pd.DataFrame = out.get("S_nodes")
    adj = _get_or_build_sea_adjacency(out)
    if not isinstance(S_nodes, pd.DataFrame) or len(S_nodes) == 0:
        return []

    if seed_idx not in adj:
        return []

    try:
        a_ll = (float(S_nodes.iloc[int(seed_idx)]["lon"]), float(S_nodes.iloc[int(seed_idx)]["lat"]))
    except Exception:
        return []

    nbrs = adj.get(seed_idx, [])
    if not nbrs:
        return []

    # cap neighbors for compute
    nbrs = nbrs[: max(0, int(max_neighbors))]

    out_cands: List[SnapCandidate] = []
    seen = set()

    vid = -10_000_000  # negative ids for virtual
    for j in nbrs:
        try:
            b_ll = (float(S_nodes.iloc[int(j)]["lon"]), float(S_nodes.iloc[int(j)]["lat"]))
        except Exception:
            continue

        for t in t_samples:
            try:
                p_ll = _interp_lonlat(a_ll, b_ll, float(t))
                key = (round(p_ll[0], round_ndigits), round(p_ll[1], round_ndigits))
                if key in seen:
                    continue
                seen.add(key)

                d_km = float(haversine_km(p_used_ll, p_ll))
                cand = SnapCandidate(
                    node_id=coord_id(p_ll[0], p_ll[1], prefix="SV:"),
                    node_idx=None,
                    node_ll=(float(p_ll[0]), float(p_ll[1])),
                    dist_km=d_km,
                    component=(int(component) if component is not None else None),
                    ok=True,
                )
                # attach bridge endpoints so inject can connect it back to the sea graph
                cand._virtual_bridge = {"u_ll": a_ll, "v_ll": b_ll, "u_id": coord_id(a_ll[0], a_ll[1], prefix="S:"), "v_id": coord_id(b_ll[0], b_ll[1], prefix="S:")}
                out_cands.append(cand)
                vid -= 1
            except Exception:
                continue

    out_cands.sort(key=lambda c: c.dist_km)
    return out_cands


# -------------------------
# Coastal KDTree helpers
# -------------------------
def _get_coast_nodes_df(out):
    C_nodes = out.get("C_nodes", None)
    if C_nodes is None:
        return None
    if not hasattr(C_nodes, "columns"):
        return None
    need = {"lon", "lat"}
    if not need.issubset(set(C_nodes.columns)):
        return None
    return C_nodes


def _get_or_build_coast_kdt(out):
    """Build & cache KDTree for coastal nodes in metric xy."""
    if out.get("coast_kdt", None) is not None and out.get("coast_xy_m", None) is not None:
        return out["coast_kdt"], out["coast_xy_m"]

    C_nodes = _get_coast_nodes_df(out)
    if C_nodes is None or len(C_nodes) == 0:
        return None, None

    xs = C_nodes["lon"].astype(float).to_numpy()
    ys = C_nodes["lat"].astype(float).to_numpy()
    proj = _get_proj(out)
    xm, ym = proj.to_m.transform(xs, ys)
    xy = np.column_stack([np.asarray(xm, dtype=float), np.asarray(ym, dtype=float)])

    # KDTree (prefer sklearn, fallback scipy)
    try:
        from sklearn.neighbors import KDTree  # type: ignore

        kdt = KDTree(xy, leaf_size=40)
    except Exception:
        from scipy.spatial import cKDTree  # type: ignore

        kdt = cKDTree(xy)

    out["coast_xy_m"] = xy
    out["coast_kdt"] = kdt
    return kdt, xy


def nudge_to_nearest_coastal_node(
    out: Dict[str, Any],
    p_ll: LonLat,
    *,
    k_near: int = 80,
    r_max_km: float = 150.0,
    collision_prep=None,
) -> Tuple[LonLat, bool, Dict[str, Any]]:
    """
    If p_ll is in collision, move it to nearest coastal node (C_nodes) that is outside collision.
    """
    p_ll = normalize_lonlat(p_ll)

    geom_m = _get_collision_geom_m(out)
    if geom_m is not None and collision_prep is None:
        collision_prep = prep(geom_m)

    if geom_m is None or (not _is_in_collision(out, p_ll, collision_prep=collision_prep)):
        return p_ll, False, {"inside": False}

    kdt, _xy = _get_or_build_coast_kdt(out)
    C_nodes = _get_coast_nodes_df(out)
    if kdt is None or C_nodes is None or len(C_nodes) == 0:
        return p_ll, False, {"inside": True, "fail": "missing_C_nodes_or_coast_kdt"}

    x, y = _point_ll_to_m(out, p_ll)
    idxs = _kdt_query_indices(kdt, x, y, k=min(int(k_near), len(C_nodes)))

    best = None
    for i in idxs:
        row = C_nodes.iloc[int(i)]
        cand = (float(row["lon"]), float(row["lat"]))
        d_km = float(haversine_km(p_ll, cand))
        if d_km > float(r_max_km):
            continue
        if _is_in_collision(out, cand, collision_prep=collision_prep):
            continue
        best = (cand, d_km, int(i))
        break

    if best is None:
        return p_ll, False, {
            "inside": True,
            "fail": "no_valid_coastal_node_within_radius",
            "k": len(idxs),
            "r_max_km": float(r_max_km),
        }

    cand, d_km, idx = best
    return normalize_lonlat(cand), True, {"inside": True, "picked_idx": idx, "dist_km": d_km}


# -------------------------
# Dataclasses
# -------------------------
@dataclass
class SnapCandidate:
    node_id: str
    node_idx: Optional[int]
    node_ll: LonLat
    dist_km: float
    component: Optional[int]
    ok: bool


@dataclass
class SnapResult:
    p_input_ll: LonLat
    p_used_ll: LonLat
    was_nudged: bool
    in_collision_input: bool
    candidates: List[SnapCandidate]
    reason: str
    debug: Dict[str, Any]


@dataclass
class SnapPairResult:
    start: SnapResult
    end: SnapResult
    chosen_common_component: Optional[int]
    largest_component: Optional[int]
    start_pick: List[SnapCandidate]
    end_pick: List[SnapCandidate]
    reason: str
    debug: Dict[str, Any]


# -------------------------
# Nudge: push point out of collision (boundary-based)
# -------------------------
def nudge_out_of_collision(
    out: Dict[str, Any],
    p_ll: LonLat,
    *,
    buffer_m: Optional[float] = None,
    max_step_m: float = 50_000.0,
    collision_prep=None,
) -> Tuple[LonLat, bool, Dict[str, Any]]:
    """
    If p_ll is inside collision geom, push it to outside:
    - Find nearest point on collision boundary in meters.
    - Move from p towards boundary and go beyond by (distance_to_boundary + buffer).
    Returns (p_ll_new, moved?, debug)
    """
    p_ll = normalize_lonlat(p_ll)
    geom_m = _get_collision_geom_m(out)
    if geom_m is None:
        return p_ll, False, {"note": "no_collision_geom"}

    if collision_prep is None:
        collision_prep = prep(geom_m)

    x, y = _point_ll_to_m(out, p_ll)
    pt = Point(x, y)

    inside = bool(collision_prep.contains(pt) or collision_prep.intersects(pt))
    if not inside:
        return p_ll, False, {"inside": False}

    if buffer_m is None:
        buffer_m = _guess_nudge_buffer_m(out, default_km=0.5)

    # distance from point to boundary (meters)
    try:
        boundary = geom_m.boundary
        d_to_boundary = float(boundary.distance(pt))
    except Exception:
        boundary = geom_m
        d_to_boundary = float(boundary.distance(pt))

    # find nearest boundary point
    try:
        _, nb = nearest_points(pt, boundary)
        bx, by = float(nb.x), float(nb.y)
    except Exception:
        bx, by = x + 1.0, y  # fallback

    vx, vy = (bx - x), (by - y)
    norm = float(np.hypot(vx, vy))

    dbg = {
        "inside": True,
        "buffer_m": float(buffer_m),
        "d_to_boundary_m": float(d_to_boundary),
        "nearest_boundary_xy": (bx, by),
        "pt_xy": (x, y),
    }

    if norm < 1e-6:
        # Degenerate: just step east by buffer
        new_xy = (x + float(buffer_m), y)
        new_ll = _point_m_to_ll(out, new_xy)
        step = float(buffer_m)
        while step < float(max_step_m):
            if not _is_in_collision(out, new_ll, collision_prep=collision_prep):
                return normalize_lonlat(new_ll), True, {**dbg, "degenerate": True, "step_m": step}
            step *= 1.5
            new_ll = _point_m_to_ll(out, (x + step, y))
        return p_ll, False, {**dbg, "degenerate": True, "fail": "cannot_escape"}

    ux, uy = vx / norm, vy / norm
    step_m = min(d_to_boundary + float(buffer_m), float(max_step_m))

    new_xy = (x + ux * step_m, y + uy * step_m)
    new_ll = normalize_lonlat(_point_m_to_ll(out, new_xy))

    step = step_m
    while step < float(max_step_m) and _is_in_collision(out, new_ll, collision_prep=collision_prep):
        step *= 1.5
        new_xy = (x + ux * step, y + uy * step)
        new_ll = normalize_lonlat(_point_m_to_ll(out, new_xy))

    ok = not _is_in_collision(out, new_ll, collision_prep=collision_prep)
    dbg.update({"step_m": step, "escaped": ok})

    return (new_ll if ok else p_ll), bool(ok), dbg


# -------------------------
# Core: sea-first candidate selection (with local entrance augmentation)
# -------------------------
def _sea_first_candidates(
    out: Dict[str, Any],
    *,
    p_input_ll: LonLat,
    p_used_ll: LonLat,
    was_nudged: bool,
    in_collision_input: bool,
    S_nodes: pd.DataFrame,
    sea_kdt,
    sea_ok_set,
    k_near: int,
    r_max_km: float,
    prefer_ok_set: bool,
    allow_fallback_non_ok: bool,
    allow_radius_fallback: bool,
    r_fallback_km: Optional[float],
    # routing-aware trigger target
    target_ll: Optional[LonLat] = None,
    # local entrance augmentation controls
    enable_local_entrance_aug: bool = True,
    aug_dist_trigger_km: float = 60.0,
    aug_delta_end_km: float = 120.0,
    aug_angle_trigger_deg: float = 110.0,
    aug_seed_neighbors_cap: int = 12,
    aug_seed_count: int = 1,
    extra_debug: Optional[Dict[str, Any]] = None,
) -> SnapResult:
    """
    Sea-first candidate selection. Optionally augments candidates by sampling points on adjacent sea edges
    around the nearest sea node when it looks "bad" (often causes V-shape detours).
    """
    dbg = extra_debug if isinstance(extra_debug, dict) else {}
    dbg.setdefault("mode", "sea_first")

    # query nearest sea nodes from p_used_ll
    x, y = _point_ll_to_m(out, p_used_ll)
    kq = max(1, int(k_near))
    try:
        idxs = _kdt_query_indices(sea_kdt, x, y, k=min(kq, len(S_nodes)))
    except Exception as e:
        return SnapResult(
            p_input_ll=p_input_ll,
            p_used_ll=p_used_ll,
            was_nudged=was_nudged,
            in_collision_input=in_collision_input,
            candidates=[],
            reason="kdt_query_failed",
            debug={**dbg, "error": repr(e)},
        )

    def make_candidate(i: int) -> SnapCandidate:
        row = S_nodes.iloc[int(i)]
        ll = (float(row["lon"]), float(row["lat"]))
        dist = float(haversine_km(p_used_ll, ll))
        comp = None
        try:
            comp = int(row["component"])
        except Exception:
            comp = None
        ok = True
        if isinstance(sea_ok_set, set):
            ok = (int(i) in sea_ok_set)
        return SnapCandidate(node_id=(str(S_nodes.loc[int(i), 'node_id']) if isinstance(S_nodes, pd.DataFrame) and 'node_id' in S_nodes.columns and int(i) in S_nodes.index else coord_id(ll[0], ll[1], prefix='S:')),
                           node_idx=int(i), node_ll=ll, dist_km=dist, component=comp, ok=ok)

    cands_all: List[SnapCandidate] = [make_candidate(i) for i in idxs]
    cands_all.sort(key=lambda c: c.dist_km)
    dbg["k_near_returned"] = len(cands_all)

    # -------------------------
    # Local entrance augmentation trigger (routing-aware-ish, but cheap)
    # -------------------------
    aug_dbg = {"enabled": bool(enable_local_entrance_aug), "triggered": False, "reason": None}
    if enable_local_entrance_aug and len(cands_all) > 0:
        # Seed candidates: nearest(s)
        seeds = cands_all[: max(1, int(aug_seed_count))]
        seed0 = seeds[0]

        trigger = False
        reasons: List[str] = []

        # A) too few candidates (sparse net)
        if len(cands_all) <= 2:
            trigger = True
            reasons.append("few_candidates")

        # B) nearest sea node is "far"
        if seed0.dist_km > float(aug_dist_trigger_km):
            trigger = True
            reasons.append(f"dist>{aug_dist_trigger_km:g}km")

        # C) routing-aware: compare to target
        if target_ll is not None and isinstance(target_ll, (list, tuple)) and len(target_ll) >= 2:
            tgt = (float(target_ll[0]), float(target_ll[1]))
            # compare d_end to best among top-K nodes (relative, avoids long-haul false triggers)
            topK = cands_all[: min(len(cands_all), max(5, int(k_near)))]
            d_end_list = [float(haversine_km(c.node_ll, tgt)) for c in topK]
            d_end_min = float(min(d_end_list)) if d_end_list else float(haversine_km(seed0.node_ll, tgt))
            d_end_seed0 = float(haversine_km(seed0.node_ll, tgt))

            # angle: p_used->seed0 vs p_used->target
            try:
                b1 = bearing_deg(p_used_ll, seed0.node_ll)
                b2 = bearing_deg(p_used_ll, tgt)
                a_diff = ang_diff_deg(b1, b2)
            except Exception:
                a_diff = 0.0

            aug_dbg.update(
                {
                    "target_ll": tgt,
                    "d_end_seed0_km": d_end_seed0,
                    "d_end_min_topK_km": d_end_min,
                    "d_end_gap_km": float(d_end_seed0 - d_end_min),
                    "angle_diff_deg": float(a_diff),
                }
            )

            if (d_end_seed0 - d_end_min) > float(aug_delta_end_km):
                trigger = True
                reasons.append(f"end_gap>{aug_delta_end_km:g}km")
            if a_diff > float(aug_angle_trigger_deg):
                trigger = True
                reasons.append(f"angle>{aug_angle_trigger_deg:g}deg")

        if trigger:
            aug_dbg["triggered"] = True
            aug_dbg["reason"] = ",".join(reasons) if reasons else "trigger"

            # Build virtual candidates from seed node(s)
            added_total = 0
            for s in seeds:
                v = _virtual_candidates_from_seed_node(
                    out,
                    seed_idx=int(s.node_idx),
                    p_used_ll=p_used_ll,
                    component=s.component,
                    max_neighbors=int(aug_seed_neighbors_cap),
                )
                if v:
                    cands_all.extend(v)
                    added_total += len(v)

            # Dedup by rounded lon/lat
            seen = set()
            uniq: List[SnapCandidate] = []
            for c in sorted(cands_all, key=lambda cc: cc.dist_km):
                key = (
                    round(float(c.node_ll[0]), 6),
                    round(float(c.node_ll[1]), 6),
                    int(c.component) if c.component is not None else None,
                )
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(c)
            cands_all = uniq
            aug_dbg["virtual_added"] = int(added_total)
            aug_dbg["cands_all_after"] = int(len(cands_all))

    dbg["local_entrance_aug"] = aug_dbg

    # -------------------------
    # selection logic (radius + ok_set + fallbacks)
    # -------------------------
    def filter_by_radius(cands: List[SnapCandidate], r: float) -> List[SnapCandidate]:
        rr = float(r)
        return [c for c in cands if c.dist_km <= rr]

    # decide radius fallback
    if r_fallback_km is None:
        cfg = out.get("cfg", None)
        try:
            r_fallback_km = float(getattr(getattr(cfg, "sea"), "r_max_km"))
        except Exception:
            r_fallback_km = float(r_max_km)

    cands_r = filter_by_radius(cands_all, float(r_max_km))
    dbg["within_r_max_km"] = len(cands_r)

    cands_ok = [c for c in cands_r if c.ok] if prefer_ok_set else cands_r
    dbg["within_r_ok"] = len(cands_ok)

    chosen = cands_ok
    reason = "ok_within_radius" if chosen else "no_ok_within_radius"

    if (not chosen) and allow_fallback_non_ok:
        chosen = cands_r
        reason = "fallback_non_ok_within_radius" if chosen else "no_candidates_within_radius"

    if (not chosen) and allow_radius_fallback and float(r_fallback_km) > float(r_max_km):
        cands_r2 = filter_by_radius(cands_all, float(r_fallback_km))
        dbg["within_r_fallback_km"] = len(cands_r2)
        cands_ok2 = [c for c in cands_r2 if c.ok] if prefer_ok_set else cands_r2
        if cands_ok2:
            chosen = cands_ok2
            reason = "ok_within_radius_fallback"
        elif allow_fallback_non_ok and cands_r2:
            chosen = cands_r2
            reason = "fallback_non_ok_within_radius_fallback"

    chosen = sorted(chosen, key=lambda c: c.dist_km)

    return SnapResult(
        p_input_ll=p_input_ll,
        p_used_ll=p_used_ll,
        was_nudged=was_nudged,
        in_collision_input=in_collision_input,
        candidates=chosen,
        reason=reason,
        debug=dbg,
    )


# -------------------------
# Snap decision tree (Updated: do NOT lock into coastal->GateB)
# -------------------------
def snap_to_sea_candidates(
    out: Dict[str, Any],
    p_ll: LonLat,
    *,
    k_near: int = 30,
    r_max_km: float = 150.0,
    prefer_ok_set: bool = True,
    allow_fallback_non_ok: bool = True,
    allow_radius_fallback: bool = True,
    r_fallback_km: Optional[float] = None,
    do_nudge: bool = True,
    # coastal nudge params
    k_near_coast: int = 80,
    r_max_km_coast: Optional[float] = None,
    # routing-aware trigger target (for local entrance augmentation)
    target_ll: Optional[LonLat] = None,
    # local entrance augmentation knobs
    enable_local_entrance_aug: bool = True,
    aug_dist_trigger_km: float = 60.0,
    aug_delta_end_km: float = 120.0,
    aug_angle_trigger_deg: float = 110.0,
    aug_seed_neighbors_cap: int = 12,
    aug_seed_count: int = 1,
) -> SnapResult:
    """
    Decision tree:
    - If point NOT in collision: normal sea-first candidates at p_ll
    - If point IS in collision and do_nudge:
        1) nudge to nearest coastal node p_coast (legal point just outside land)
        2) sea-first candidates are queried from p_coast (NOT GateB-locked)
    """
    p_ll0 = normalize_lonlat(p_ll)
    dbg: Dict[str, Any] = {}

    S_nodes = out.get("S_nodes", None)
    sea_kdt = out.get("sea_kdt", None)
    sea_ok_set = out.get("sea_ok_set", None)

    if S_nodes is None or sea_kdt is None:
        return SnapResult(
            p_input_ll=p_ll0,
            p_used_ll=p_ll0,
            was_nudged=False,
            in_collision_input=False,
            candidates=[],
            reason="missing_S_nodes_or_sea_kdt",
            debug={"has_S_nodes": S_nodes is not None, "has_kdt": sea_kdt is not None},
        )

    # collision check
    collision_prep = None
    geom_m = _get_collision_geom_m(out)
    if geom_m is not None:
        collision_prep = prep(geom_m)

    in_collision = _is_in_collision(out, p_ll0, collision_prep=collision_prep) if geom_m is not None else False

    # CASE 1) Not in collision -> sea-first
    if (not in_collision) or (not do_nudge) or (geom_m is None):
        dbg["mode"] = "sea_first"
        return _sea_first_candidates(
            out,
            p_input_ll=p_ll0,
            p_used_ll=p_ll0,
            was_nudged=False,
            in_collision_input=in_collision,
            S_nodes=S_nodes,
            sea_kdt=sea_kdt,
            sea_ok_set=sea_ok_set,
            k_near=k_near,
            r_max_km=r_max_km,
            prefer_ok_set=prefer_ok_set,
            allow_fallback_non_ok=allow_fallback_non_ok,
            allow_radius_fallback=allow_radius_fallback,
            r_fallback_km=r_fallback_km,
            target_ll=target_ll,
            enable_local_entrance_aug=enable_local_entrance_aug,
            aug_dist_trigger_km=aug_dist_trigger_km,
            aug_delta_end_km=aug_delta_end_km,
            aug_angle_trigger_deg=aug_angle_trigger_deg,
            aug_seed_neighbors_cap=aug_seed_neighbors_cap,
            aug_seed_count=aug_seed_count,
            extra_debug=dbg,
        )

    # CASE 2) In collision -> coast then sea-first
    dbg["mode"] = "coast_then_sea"

    p_coast, ok_coast, dbg_coast = nudge_to_nearest_coastal_node(
        out,
        p_ll0,
        k_near=int(k_near_coast),
        r_max_km=float(r_max_km if r_max_km_coast is None else r_max_km_coast),
        collision_prep=collision_prep,
    )

    dbg["chosen_coastal"] = {
        "ok": bool(ok_coast),
        "p_coast_ll": (float(p_coast[0]), float(p_coast[1])),
        **(dbg_coast if isinstance(dbg_coast, dict) else {"dbg": str(dbg_coast)}),
    }

    if not ok_coast:
        # fallback: boundary nudge then sea-first
        dbg["fallback_reason"] = "coastal_nudge_failed"
        p_b, ok_b, dbg_b = nudge_out_of_collision(out, p_ll0, collision_prep=collision_prep)
        dbg["boundary_fallback"] = {"ok": bool(ok_b), **(dbg_b if isinstance(dbg_b, dict) else {"dbg": str(dbg_b)})}

        return _sea_first_candidates(
            out,
            p_input_ll=p_ll0,
            p_used_ll=p_b,
            was_nudged=bool(ok_b),
            in_collision_input=True,
            S_nodes=S_nodes,
            sea_kdt=sea_kdt,
            sea_ok_set=sea_ok_set,
            k_near=k_near,
            r_max_km=r_max_km,
            prefer_ok_set=prefer_ok_set,
            allow_fallback_non_ok=allow_fallback_non_ok,
            allow_radius_fallback=allow_radius_fallback,
            r_fallback_km=r_fallback_km,
            target_ll=target_ll,
            enable_local_entrance_aug=enable_local_entrance_aug,
            aug_dist_trigger_km=aug_dist_trigger_km,
            aug_delta_end_km=aug_delta_end_km,
            aug_angle_trigger_deg=aug_angle_trigger_deg,
            aug_seed_neighbors_cap=aug_seed_neighbors_cap,
            aug_seed_count=aug_seed_count,
            extra_debug=dbg,
        )

    return _sea_first_candidates(
        out,
        p_input_ll=p_ll0,
        p_used_ll=p_coast,
        was_nudged=True,
        in_collision_input=True,
        S_nodes=S_nodes,
        sea_kdt=sea_kdt,
        sea_ok_set=sea_ok_set,
        k_near=k_near,
        r_max_km=r_max_km,
        prefer_ok_set=prefer_ok_set,
        allow_fallback_non_ok=allow_fallback_non_ok,
        allow_radius_fallback=allow_radius_fallback,
        r_fallback_km=r_fallback_km,
        target_ll=target_ll,
        enable_local_entrance_aug=enable_local_entrance_aug,
        aug_dist_trigger_km=aug_dist_trigger_km,
        aug_delta_end_km=aug_delta_end_km,
        aug_angle_trigger_deg=aug_angle_trigger_deg,
        aug_seed_neighbors_cap=aug_seed_neighbors_cap,
        aug_seed_count=aug_seed_count,
        extra_debug=dbg,
    )


# -------------------------
# Pair: component-aware pick (passes target_ll to enable routing-aware triggers)
# -------------------------
def snap_pair_component_aware(
    out: Dict[str, Any],
    start_ll: LonLat,
    end_ll: LonLat,
    *,
    start_policy: Optional[str] = None,  # NEW: "R" or "S" or None
    end_policy: Optional[str] = None,    # NEW: "R" or "S" or None
    k_near: int = 30,
    r_max_km: float = 150.0,
    k_inject: int = 4,
    prefer_ok_set: bool = True,
    allow_fallback_non_ok: bool = True,
    allow_radius_fallback: bool = True,
    do_nudge: bool = True,
    k_near_coast: int = 80,
    r_max_km_coast: Optional[float] = None,
    # local entrance augmentation knobs
    enable_local_entrance_aug: bool = True,
    aug_dist_trigger_km: float = 60.0,
    aug_delta_end_km: float = 120.0,
    aug_angle_trigger_deg: float = 110.0,
    aug_seed_neighbors_cap: int = 12,
    aug_seed_count: int = 1,
) -> SnapPairResult:
    """
    Component-aware pair snapping.
    We pass the opposite endpoint as target_ll so snapping can detect "bad nearest entrance"
    and augment virtual candidates around adjacent sea edges.
    """
    sp = (start_policy or "S").upper()
    ep = (end_policy or "S").upper()
        # ---- START ----
    if sp == "R":
        sres = snap_to_ring_candidates(out, start_ll, k_near=k_near, prefer="auto", target_ll=end_ll)
    else:
        sres = snap_to_sea_candidates(
            out,
            start_ll,
            k_near=k_near,
            r_max_km=r_max_km,
            prefer_ok_set=prefer_ok_set,
            allow_fallback_non_ok=allow_fallback_non_ok,
            allow_radius_fallback=allow_radius_fallback,
            do_nudge=do_nudge,
            k_near_coast=k_near_coast,
            r_max_km_coast=r_max_km_coast,
            target_ll=end_ll,
            enable_local_entrance_aug=enable_local_entrance_aug,
            aug_dist_trigger_km=aug_dist_trigger_km,
            aug_delta_end_km=aug_delta_end_km,
            aug_angle_trigger_deg=aug_angle_trigger_deg,
            aug_seed_neighbors_cap=aug_seed_neighbors_cap,
            aug_seed_count=aug_seed_count,
        )

    # ---- END ----
    if ep == "R":
        eres = snap_to_ring_candidates(out, end_ll, k_near=k_near, prefer="auto", target_ll=start_ll)
    else:
        eres = snap_to_sea_candidates(
            out,
            end_ll,
            k_near=k_near,
            r_max_km=r_max_km,
            prefer_ok_set=prefer_ok_set,
            allow_fallback_non_ok=allow_fallback_non_ok,
            allow_radius_fallback=allow_radius_fallback,
            do_nudge=do_nudge,
            k_near_coast=k_near_coast,
            r_max_km_coast=r_max_km_coast,
            target_ll=start_ll,
            enable_local_entrance_aug=enable_local_entrance_aug,
            aug_dist_trigger_km=aug_dist_trigger_km,
            aug_delta_end_km=aug_delta_end_km,
            aug_angle_trigger_deg=aug_angle_trigger_deg,
            aug_seed_neighbors_cap=aug_seed_neighbors_cap,
            aug_seed_count=aug_seed_count,
        )

    dbg: Dict[str, Any] = {
        "start_reason": sres.reason,
        "end_reason": eres.reason,
        "start_mode": sres.debug.get("mode") if isinstance(sres.debug, dict) else None,
        "end_mode": eres.debug.get("mode") if isinstance(eres.debug, dict) else None,
        "start_fallback": sres.debug.get("fallback_reason") if isinstance(sres.debug, dict) else None,
        "end_fallback": eres.debug.get("fallback_reason") if isinstance(eres.debug, dict) else None,
    }

    if len(sres.candidates) == 0 or len(eres.candidates) == 0:
        return SnapPairResult(
            start=sres,
            end=eres,
            chosen_common_component=None,
            largest_component=None,
            start_pick=[],
            end_pick=[],
            reason="snap_failed",
            debug=dbg,
        )
    
    # NEW: ring-world bypass component logic
    if sp == "R" or ep == "R":
        spick = sorted(sres.candidates, key=lambda c: c.dist_km)[: max(1, int(k_inject))]
        epick = sorted(eres.candidates, key=lambda c: c.dist_km)[: max(1, int(k_inject))]
        return SnapPairResult(
            start=sres,
            end=eres,
            chosen_common_component=None,
            largest_component=None,
            start_pick=spick,
            end_pick=epick,
            reason="ring_policy_bypass_component",
            debug=dbg,
        )


    start_comps = {c.component for c in sres.candidates if c.component is not None}
    end_comps = {c.component for c in eres.candidates if c.component is not None}
    common = sorted(list(start_comps.intersection(end_comps)))

    chosen_common = None
    if common:
        best = None
        for comp in common:
            ds = min([c.dist_km for c in sres.candidates if c.component == comp], default=np.inf)
            de = min([c.dist_km for c in eres.candidates if c.component == comp], default=np.inf)
            score = ds + de
            if best is None or score < best[0]:
                best = (score, comp)
        chosen_common = best[1] if best else common[0]

    # infer largest component from sea_ok_set distribution
    largest_comp = None
    try:
        S_nodes_df = out.get("S_nodes")
        sea_ok_set = out.get("sea_ok_set")
        if (
            isinstance(S_nodes_df, pd.DataFrame)
            and isinstance(sea_ok_set, set)
            and len(sea_ok_set) > 0
            and "component" in S_nodes_df.columns
        ):
            comps = S_nodes_df.loc[list(sea_ok_set), "component"].value_counts()
            if len(comps):
                largest_comp = int(comps.index[0])
    except Exception:
        largest_comp = None

    dbg.update(
        {
            "common_components": common,
            "chosen_common_component": chosen_common,
            "largest_component": largest_comp,
        }
    )

    def order_candidates(cands: List[SnapCandidate]) -> List[SnapCandidate]:
        def key(c: SnapCandidate):
            pri = 2
            if chosen_common is not None and c.component == chosen_common:
                pri = 0
            elif largest_comp is not None and c.component == largest_comp:
                pri = 1
            return (pri, c.dist_km)

        return sorted(cands, key=key)

    spick = order_candidates(sres.candidates)[: max(1, int(k_inject))]
    epick = order_candidates(eres.candidates)[: max(1, int(k_inject))]

    reason = "common_component_preferred" if chosen_common is not None else "largest_component_preferred"

    return SnapPairResult(
        start=sres,
        end=eres,
        chosen_common_component=chosen_common,
        largest_component=largest_comp,
        start_pick=spick,
        end_pick=epick,
        reason=reason,
        debug=dbg,
    )


# -------------------------
# Inject edges into graph
# -------------------------
def inject_point_edges(
    G,
    p_id: str,
    p_ll: LonLat,
    candidates: Sequence[SnapCandidate],
    *,
    k_inject: int = 4,
    etype: str = "inject",
    weight_attr: str = "weight",
):
    """Inject a query point into the graph using node_id keys.

    - Adds node `p_id` with lon/lat attrs.
    - Adds edges from `p_id` to top-k candidate nodes.
    - If a candidate is a virtual point (has `_virtual_bridge`), also connects that
      virtual node back to the sea graph endpoints (by node_id).
    """
    p_ll = normalize_lonlat(p_ll)
    if not hasattr(G, "add_edge"):
        raise TypeError("G must be a networkx-like graph with add_edge.")

    # Ensure query node exists with attrs
    if hasattr(G, "add_node"):
        G.add_node(str(p_id), lon=float(p_ll[0]), lat=float(p_ll[1]), kind="query")

    use = list(candidates)[: max(1, int(k_inject))]
    for c in use:
        u = str(p_id)
        v = str(c.node_id)

        # Ensure candidate node exists with attrs (best-effort)
        if hasattr(G, "add_node") and (v not in G):
            G.add_node(v, lon=float(c.node_ll[0]), lat=float(c.node_ll[1]), kind="candidate")

        w = float(haversine_km(p_ll, c.node_ll))
        lat_max = float(max(abs(float(p_ll[1])), abs(float(c.node_ll[1]))))
        ban = int(B_HIGH_LAT) if lat_max > 70.0 else 0
        G.add_edge(u, v, **{weight_attr: w, "length_km": w, "etype": etype, "layer_mask": int(L_INJECT), "ban_mask": ban, "lat_max_abs": lat_max})

        # If virtual: connect it back to sea graph so A* can traverse it
        bridge = getattr(c, "_virtual_bridge", None)
        if isinstance(bridge, dict) and "u_id" in bridge and "v_id" in bridge:
            vu = str(bridge["u_id"])
            vv = str(bridge["v_id"])
            u_ll = tuple(map(float, bridge.get("u_ll", c.node_ll)))
            v_ll = tuple(map(float, bridge.get("v_ll", c.node_ll)))

            # connect virtual -> endpoints (weights in km)
            w1 = float(haversine_km(c.node_ll, u_ll))
            w2 = float(haversine_km(c.node_ll, v_ll))
            G.add_edge(v, vu, **{weight_attr: w1, "length_km": w1, "etype": "sea_virtual"})
            G.add_edge(v, vv, **{weight_attr: w2, "length_km": w2, "etype": "sea_virtual"})



def _get_ring_nodes(out: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    rg = out.get("ring_graph", {}) or {}
    E_nodes = rg.get("E_nodes", None)
    T_nodes = rg.get("T_nodes", None)
    if not isinstance(E_nodes, pd.DataFrame):
        E_nodes = None
    if not isinstance(T_nodes, pd.DataFrame):
        T_nodes = None
    return E_nodes, T_nodes


def _build_kdt_for_nodes(
    out: Dict[str, Any],
    df: pd.DataFrame,
    *,
    cache_key: str,
) -> Optional["Any"]:
    """Build (and cache) a KDTree for a node dataframe in metric coords.

    Uses existing x_m/y_m columns if present, else projects lon/lat using out["proj"].
    """
    if df is None or len(df) == 0:
        return None

    # cache
    kdt = out.get(cache_key, None)
    if kdt is not None:
        return kdt

    if "x_m" in df.columns and "y_m" in df.columns:
        xy = df[["x_m", "y_m"]].to_numpy(dtype=float, copy=False)
    else:
        # project lon/lat to metric
        lon = df["lon"].to_numpy(dtype=float, copy=False)
        lat = df["lat"].to_numpy(dtype=float, copy=False)
        xy = np.zeros((len(df), 2), dtype=float)
        for i in range(len(df)):
            xy[i, :] = _point_ll_to_m(out, (float(lon[i]), float(lat[i])))
    
    try:
        from sklearn.neighbors import KDTree  # type: ignore
        kdt = KDTree(xy, leaf_size=40)
    except Exception:
        try:
            from scipy.spatial import cKDTree  # type: ignore
            kdt = cKDTree(xy)
        except Exception:
            return None

    out[cache_key] = kdt
    return kdt


def _nearest_node_dist_km(out: Dict[str, Any], df: Optional[pd.DataFrame], kdt: Any, p_ll: LonLat) -> float:
    if df is None or kdt is None or len(df) == 0:
        return float("inf")

    x, y = _point_ll_to_m(out, p_ll)
    try:
        idxs = _kdt_query_indices(kdt, float(x), float(y), k=1)
        if not idxs:
            return float("inf")
        row = df.iloc[int(idxs[0])]

        # 用 meters 算距離（因為 x_m/y_m 就是 meters）
        if "x_m" in df.columns and "y_m" in df.columns:
            dx = float(x) - float(row["x_m"])
            dy = float(y) - float(row["y_m"])
            return (dx * dx + dy * dy) ** 0.5 / 1000.0

        # fallback: haversine
        ll = (float(row["lon"]), float(row["lat"]))
        return float(haversine_km(p_ll, ll))
    except Exception:
        # fallback: haversine brute
        best = float("inf")
        for _, r in df.iterrows():
            ll = (float(r["lon"]), float(r["lat"]))
            best = min(best, float(haversine_km(p_ll, ll)))
        return best



def snap_to_ring_candidates(
    out: Dict[str, Any],
    p_ll: LonLat,
    *,
    k_near: int = 30,
    prefer: str = "auto",
    target_ll: Optional[LonLat] = None,
) -> SnapResult:
    """Return ring-node candidates near p_ll.

    prefer:
      - "E": only E ring nodes
      - "T": only T ring nodes
      - "auto": choose closer between E and T (by nearest-node distance)
    """
    p_ll0 = normalize_lonlat(p_ll)
    dbg: Dict[str, Any] = {"mode": "ring", "prefer": prefer}

    E_nodes, T_nodes = _get_ring_nodes(out)
    if E_nodes is None and T_nodes is None:
        return SnapResult(
            p_input_ll=p_ll0,
            p_used_ll=p_ll0,
            was_nudged=False,
            in_collision_input=False,
            candidates=[],
            reason="missing_ring_nodes",
            debug=dbg,
        )

    # KD trees
    e_kdt = _build_kdt_for_nodes(out, E_nodes, cache_key="e_nodes_kdt") if E_nodes is not None else None
    t_kdt = _build_kdt_for_nodes(out, T_nodes, cache_key="t_nodes_kdt") if T_nodes is not None else None

    # choose set
    if prefer.upper() == "E":
        chosen_df, chosen_kdt, chosen_tag = E_nodes, e_kdt, "E"
    elif prefer.upper() == "T":
        chosen_df, chosen_kdt, chosen_tag = T_nodes, t_kdt, "T"
    else:
        dE = _nearest_node_dist_km(out, E_nodes, e_kdt, p_ll0) if E_nodes is not None else float("inf")
        dT = _nearest_node_dist_km(out, T_nodes, t_kdt, p_ll0) if T_nodes is not None else float("inf")
        chosen_tag = "E" if dE <= dT else "T"
        chosen_df, chosen_kdt = (E_nodes, e_kdt) if chosen_tag == "E" else (T_nodes, t_kdt)
        dbg.update({"dE_km": float(dE), "dT_km": float(dT), "chosen": chosen_tag})

    if chosen_df is None or chosen_kdt is None or len(chosen_df) == 0:
        return SnapResult(
            p_input_ll=p_ll0,
            p_used_ll=p_ll0,
            was_nudged=False,
            in_collision_input=False,
            candidates=[],
            reason="no_ring_candidates",
            debug=dbg,
        )

    # query nearest
    x, y = _point_ll_to_m(out, p_ll0)
    kq = max(1, min(int(k_near), len(chosen_df)))
    try:
        idxs = _kdt_query_indices(chosen_kdt, float(x), float(y), k=kq)
    except Exception as e:
        return SnapResult(
            p_input_ll=p_ll0,
            p_used_ll=p_ll0,
            was_nudged=False,
            in_collision_input=False,
            candidates=[],
            reason="ring_kdt_query_failed",
            debug={**dbg, "error": repr(e)},
        )

    cands: List[SnapCandidate] = []
    for i in idxs:
        try:
            row = chosen_df.iloc[int(i)]
            ll = (float(row["lon"]), float(row["lat"]))

            # meters distance (preferred)
            if "x_m" in chosen_df.columns and "y_m" in chosen_df.columns:
                dx = float(x) - float(row["x_m"])
                dy = float(y) - float(row["y_m"])
                dist_km = (dx * dx + dy * dy) ** 0.5 / 1000.0
            else:
                dist_km = float(haversine_km(p_ll0, ll))

            node_idx = int(row["node_id"]) if "node_id" in chosen_df.columns else int(i)
            node_key = str(row['node_key']) if 'node_key' in chosen_df.columns else f"{chosen_tag}:{int(row['node_id'])}"
            cands.append(SnapCandidate(node_id=node_key, node_idx=int(node_idx), node_ll=ll, dist_km=dist_km, component=None, ok=True))
        except Exception:
            continue

    cands.sort(key=lambda c: c.dist_km)

    return SnapResult(
        p_input_ll=p_ll0,
        p_used_ll=p_ll0,
        was_nudged=False,
        in_collision_input=False,
        candidates=cands,
        reason=f"ring_{chosen_tag}_ok",
        debug=dbg,
    )


def compute_multiworld_policies_for_point(
    out: Dict[str, Any],
    p_ll: LonLat,
    *,
    R_NEAR_COAST_KM: float = 120.0,
    S_MAX_SNAP_KM: float = 200.0,
) -> Dict[str, Any]:
    """Compute which worlds (R,S) should be attempted for this endpoint.

    Heuristics:
    - If in collision -> must include R
    - If within R_NEAR_COAST_KM of ring nodes -> include R, else may prune R
    - If nearest sea node distance > S_MAX_SNAP_KM -> prune S
    """
    p_ll0 = normalize_lonlat(p_ll)

    # collision
    collision_prep = None
    geom_m = _get_collision_geom_m(out)
    if geom_m is not None:
        try:
            collision_prep = prep(geom_m)
        except Exception:
            collision_prep = None

    in_collision = bool(_is_in_collision(out, p_ll0, collision_prep=collision_prep)) if geom_m is not None else False

    # distances
    E_nodes, T_nodes = _get_ring_nodes(out)
    e_kdt = _build_kdt_for_nodes(out, E_nodes, cache_key="e_nodes_kdt") if E_nodes is not None else None
    t_kdt = _build_kdt_for_nodes(out, T_nodes, cache_key="t_nodes_kdt") if T_nodes is not None else None
    dE = _nearest_node_dist_km(out, E_nodes, e_kdt, p_ll0) if E_nodes is not None else float("inf")
    dT = _nearest_node_dist_km(out, T_nodes, t_kdt, p_ll0) if T_nodes is not None else float("inf")
    d_ring = float(min(dE, dT))

    # sea nearest dist (robust across sklearn KDTree / scipy cKDTree)
    d_sea = float("inf")
    try:
        S_nodes = out.get("S_nodes", None)
        sea_kdt = out.get("sea_kdt", None)
        if isinstance(S_nodes, pd.DataFrame) and sea_kdt is not None and len(S_nodes) > 0:
            x, y = _point_ll_to_m(out, p_ll0)

            # Use shared helper to tolerate KDTree API differences (shape, return types)
            idxs = _kdt_query_indices(sea_kdt, x, y, k=1)
            if idxs:
                sea_idx = int(idxs[0])

                # Prefer metric coords if available (fast and consistent)
                if ("x_m" in S_nodes.columns) and ("y_m" in S_nodes.columns):
                    sx = float(S_nodes.iloc[sea_idx]["x_m"])
                    sy = float(S_nodes.iloc[sea_idx]["y_m"])
                    d_sea = float(np.hypot(sx - float(x), sy - float(y)) / 1000.0)
                else:
                    sll = (float(S_nodes.iloc[sea_idx]["lon"]), float(S_nodes.iloc[sea_idx]["lat"]))
                    d_sea = float(haversine_km(p_ll0, sll))
    except Exception:
        d_sea = float("inf")

    policies = set(["R", "S"])

    # prune ring if far (unless collision)
    if (not in_collision) and (d_ring > float(R_NEAR_COAST_KM)):
        policies.discard("R")

    # prune sea if too far
    if np.isfinite(d_sea) and d_sea > float(S_MAX_SNAP_KM):
        policies.discard("S")

    # always keep at least one
    if not policies:
        policies = set(["R"]) if in_collision else set(["S"])

    return {
        "p_ll": p_ll0,
        "in_collision": in_collision,
        "d_ring_km": d_ring,
        "dE_km": float(dE),
        "dT_km": float(dT),
        "d_sea_km": d_sea,
        "policies": sorted(list(policies)),
    }



__all__ = [
    "SnapCandidate",
    "SnapResult",
    "SnapPairResult",
    "nudge_out_of_collision",
    "nudge_to_nearest_coastal_node",
    "snap_to_sea_candidates",
    "snap_pair_component_aware",
    "inject_point_edges",
    # multi-world
    "snap_to_ring_candidates",
    "compute_multiworld_policies_for_point",
]
