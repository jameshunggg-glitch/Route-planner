"""
routing_map.t_gate_connectors

Build connectors from T-ring gate candidates to sea nodes.

Design goals:
- Follow the same pattern as existing connector modules:
  * build_* -> returns a DataFrame
  * add_*   -> writes edges into a networkx graph

This module intentionally keeps a few knobs (sector filtering, repair) optional and safe.
It does NOT assume any particular node-key scheme in the final networkx graph; instead
`add_*` accepts mapper callables so you can avoid ID collisions (e.g., ring node_id vs sea idx).

Expected inputs:
- out: dict from build_aoi(cfg)
  - out["ring_graph"]["T_nodes"] (DataFrame) with columns:
        node_id, ring_id, seq, x_m, y_m, lon, lat, is_gate_candidate (bool)
    or out["ring_graph"]["T_gate_candidates"] (optional)
  - out["S_nodes"] (DataFrame) with columns: lon, lat (x_m,y_m optional)
  - out["proj"] provides lon/lat <-> metric (m) projection via one of:
        ll_to_xy/xy_to_ll, ll_to_m/m_to_ll, to_m/to_ll, fwd/inv
  - out["layers"]["COLLISION_M"] or out["COLLISION_M"] or out["collision_m"] provides collision geometry in metric CRS.

Output DataFrame schema (recommended):
- ring_id
- t_node_id
- sea_idx
- dist_km
- reason: "direct" | "repair" | "fallback"
- sector_ok (bool)
- repaired (bool)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from shapely.geometry import LineString


# ---------------------------
# params
# ---------------------------

@dataclass
class TGateSeaConnectorParams:
    k_connect: int = 2
    topN: int = 60
    r_connect_km: float = 120

    # sector filter
    enable_sector_filter: bool = True
    sector_deg: float = 110
    sector_mode: str = "centroid"  # "centroid" or "tangent" (tangent requires full ring context; centroid is robust)

    # collision / repair
    do_collision_check: bool = True
    do_repair: bool = True
    repair_mid_frac: float = 0.6          # where to put an intermediate point along the connector length
    repair_side_push_m: float = 25_000.0  # lateral push for the intermediate point

    # output
    etype: str = "T_S_GATE"


# ---------------------------
# projection & collision helpers (robust)
# ---------------------------

def _apply_proj_fn(fn, a, b):
    if hasattr(fn, "transform") and callable(getattr(fn, "transform")):
        return fn.transform(a, b)
    if callable(fn):
        try:
            return fn(a, b)
        except TypeError:
            return fn((a, b))
    raise TypeError(f"Invalid projector function type: {type(fn)}")

def get_ll2m_m2ll(out) -> Tuple[Callable[[float, float], Tuple[float, float]], Callable[[float, float], Tuple[float, float]]]:
    """Infer lon/lat <-> metric projectors from out['proj']."""
    proj = out.get("proj", None)
    if proj is None:
        raise ValueError("out['proj'] not found")

    candidates = [("ll_to_xy", "xy_to_ll"), ("ll_to_m", "m_to_ll"), ("to_m", "to_ll"), ("fwd", "inv")]
    for a, b in candidates:
        if hasattr(proj, a) and hasattr(proj, b):
            f, g = getattr(proj, a), getattr(proj, b)
            ll2m = lambda lon, lat: tuple(map(float, _apply_proj_fn(f, float(lon), float(lat))))
            m2ll = lambda x, y: tuple(map(float, _apply_proj_fn(g, float(x), float(y))))
            return ll2m, m2ll

    # dict-like fallback
    if isinstance(proj, dict):
        for a, b in candidates:
            if a in proj and b in proj:
                f, g = proj[a], proj[b]
                ll2m = lambda lon, lat: tuple(map(float, _apply_proj_fn(f, float(lon), float(lat))))
                m2ll = lambda x, y: tuple(map(float, _apply_proj_fn(g, float(x), float(y))))
                return ll2m, m2ll

    raise ValueError("Cannot infer projection methods from out['proj'].")

def get_collision_metric(out):
    layers = out.get("layers", None)
    if isinstance(layers, dict):
        for k in ("COLLISION_PREP_M", "collision_prep_m", "collision_prep"):
            if layers.get(k) is not None:
                return layers[k]
        if layers.get("COLLISION_M") is not None:
            return layers["COLLISION_M"]

    for k in ("COLLISION_PREP_M", "collision_prep_m", "collision_prep"):
        v = out.get(k, None)
        if v is not None:
            return v

    for k in ("COLLISION_M", "collision_m", "collision"):
        v = out.get(k, None)
        if v is not None:
            return v

    return None

def _collision_intersects(collision_obj, seg: LineString) -> bool:
    if collision_obj is None:
        return False
    # prepared geometry
    if hasattr(collision_obj, "intersects") and callable(getattr(collision_obj, "intersects")):
        try:
            return bool(collision_obj.intersects(seg))
        except Exception:
            pass
    # prepared geometry stored as .context (shapely.prepared.prep)
    ctx = getattr(collision_obj, "context", None)
    if ctx is not None and hasattr(ctx, "intersects"):
        try:
            return bool(ctx.intersects(seg))
        except Exception:
            pass
    return False


# ---------------------------
# geometry helpers
# ---------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    if n <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return v / n

def _deg_between(u: np.ndarray, v: np.ndarray) -> float:
    u = _unit(u); v = _unit(v)
    c = float(np.clip(u[0]*v[0] + u[1]*v[1], -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def _compute_ring_centroids_xy(T_nodes: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    cent = {}
    for rid, df in T_nodes.groupby("ring_id", sort=False):
        cent[rid] = (float(df["x_m"].mean()), float(df["y_m"].mean()))
    return cent

def _sector_ok_centroid(g_xy: np.ndarray, s_xy: np.ndarray, centroid_xy: np.ndarray, sector_deg: float) -> bool:
    outward = g_xy - centroid_xy
    vec = s_xy - g_xy
    ang = _deg_between(outward, vec)
    return ang <= float(sector_deg)

def _attempt_simple_repair(g_xy: np.ndarray, s_xy: np.ndarray, centroid_xy: np.ndarray, params: TGateSeaConnectorParams, collision_obj) -> Optional[List[Tuple[float, float]]]:
    """
    Try a very cheap 2-hop connector: g -> mid -> s
    mid is pushed outward and sideways to avoid immediate land collision.
    Returns polyline in metric [(x,y), ...] if successful, else None.
    """
    # base direction outward
    outward = _unit(g_xy - centroid_xy)
    # side direction (rotate 90 deg)
    side = np.array([-outward[1], outward[0]], dtype=float)

    # pick a mid point along the segment then push
    mid = g_xy + params.repair_mid_frac * (s_xy - g_xy)
    mid = mid + outward * (0.5 * params.repair_side_push_m) + side * (0.5 * params.repair_side_push_m)

    # check two segments
    seg1 = LineString([tuple(g_xy), tuple(mid)])
    seg2 = LineString([tuple(mid), tuple(s_xy)])
    if _collision_intersects(collision_obj, seg1):
        # try opposite side
        mid2 = g_xy + params.repair_mid_frac * (s_xy - g_xy)
        mid2 = mid2 + outward * (0.5 * params.repair_side_push_m) - side * (0.5 * params.repair_side_push_m)
        seg1 = LineString([tuple(g_xy), tuple(mid2)])
        seg2 = LineString([tuple(mid2), tuple(s_xy)])
        if _collision_intersects(collision_obj, seg1) or _collision_intersects(collision_obj, seg2):
            return None
        return [tuple(g_xy), tuple(mid2), tuple(s_xy)]

    if _collision_intersects(collision_obj, seg2):
        return None

    return [tuple(g_xy), tuple(mid), tuple(s_xy)]


# ---------------------------
# core API
# ---------------------------

def build_tgate_sea_connectors(
    out: dict,
    *,
    params: Optional[TGateSeaConnectorParams] = None,
    S_nodes: Optional[pd.DataFrame] = None,
    T_nodes: Optional[pd.DataFrame] = None,
    gate_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build T-gate -> Sea connectors as a DataFrame.
    """
    params = params or TGateSeaConnectorParams()
    rg = out.get("ring_graph", {}) or {}

    if T_nodes is None:
        T_nodes = rg.get("T_nodes", None)
    if gate_df is None:
        gate_df = rg.get("T_gate_candidates", None)

    if S_nodes is None:
        S_nodes = out.get("S_nodes", None)

    if not isinstance(T_nodes, pd.DataFrame) or len(T_nodes) == 0:
        return pd.DataFrame(columns=["ring_id","t_node_id","sea_idx","dist_km","reason","sector_ok","repaired","etype"])
    if not isinstance(S_nodes, pd.DataFrame) or len(S_nodes) == 0:
        return pd.DataFrame(columns=["ring_id","t_node_id","sea_idx","dist_km","reason","sector_ok","repaired","etype"])

    # gate list
    if isinstance(gate_df, pd.DataFrame) and len(gate_df) > 0:
        gates = gate_df.copy()
    else:
        if "is_gate_candidate" not in T_nodes.columns:
            raise ValueError("T_nodes must contain 'is_gate_candidate' or provide gate_df.")
        gates = T_nodes[T_nodes["is_gate_candidate"] == True].copy()

    if len(gates) == 0:
        return pd.DataFrame(columns=["ring_id","t_node_id","sea_idx","dist_km","reason","sector_ok","repaired","etype"])

    # build metric coords for sea nodes
    ll2m, _ = get_ll2m_m2ll(out)

    if "x_m" in S_nodes.columns and "y_m" in S_nodes.columns and np.isfinite(S_nodes["x_m"]).all():
        S_xy = np.column_stack([S_nodes["x_m"].astype(float).values, S_nodes["y_m"].astype(float).values])
    else:
        S_xy = np.array([ll2m(lon, lat) for lon, lat in zip(S_nodes["lon"].values, S_nodes["lat"].values)], dtype=float)

    # ring centroids for sector filter + repair
    if "x_m" not in T_nodes.columns or "y_m" not in T_nodes.columns:
        raise ValueError("T_nodes must include x_m/y_m in metric CRS.")
    ring_cent = _compute_ring_centroids_xy(T_nodes)

    collision_obj = get_collision_metric(out)

    rows = []
    r2_max = (float(params.r_connect_km) * 1000.0) ** 2

    # precompute for speed
    S_x = S_xy[:, 0]; S_y = S_xy[:, 1]

    for r in gates.itertuples(index=False):
        rid = int(getattr(r, "ring_id"))
        tid = int(getattr(r, "node_id")) if hasattr(r, "node_id") else int(getattr(r, "t_node_id"))

        g_xy = np.array([float(getattr(r, "x_m")), float(getattr(r, "y_m"))], dtype=float)
        centroid_xy = np.array(ring_cent.get(rid, (float(T_nodes["x_m"].mean()), float(T_nodes["y_m"].mean()))), dtype=float)

        # distance squared in metric
        dx = S_x - g_xy[0]
        dy = S_y - g_xy[1]
        d2 = dx*dx + dy*dy

        # apply radius
        cand_idx = np.where(d2 <= r2_max)[0]
        if cand_idx.size == 0:
            continue

        # take topN by distance
        if cand_idx.size > int(params.topN):
            # partial sort
            local = d2[cand_idx]
            take = np.argpartition(local, int(params.topN) - 1)[: int(params.topN)]
            cand_idx = cand_idx[take]

        # order by distance
        cand_idx = cand_idx[np.argsort(d2[cand_idx])]

        accepted = 0
        for si in cand_idx:
            s_xy = S_xy[int(si)]
            sector_ok = True

            if params.enable_sector_filter and params.sector_mode == "centroid":
                sector_ok = _sector_ok_centroid(g_xy, s_xy, centroid_xy, float(params.sector_deg))
                if not sector_ok:
                    continue

            # collision check
            repaired = False
            reason = "direct"
            dist_km = float(np.hypot(s_xy[0] - g_xy[0], s_xy[1] - g_xy[1]) / 1000.0)

            if params.do_collision_check and collision_obj is not None:
                seg = LineString([tuple(g_xy), (float(s_xy[0]), float(s_xy[1]))])
                if _collision_intersects(collision_obj, seg):
                    if not params.do_repair:
                        continue
                    poly_m = _attempt_simple_repair(g_xy, s_xy, centroid_xy, params, collision_obj)
                    if poly_m is None:
                        continue
                    repaired = True
                    reason = "repair"
                    # recompute length from polyline
                    L = 0.0
                    for (x1, y1), (x2, y2) in zip(poly_m, poly_m[1:]):
                        L += float(np.hypot(x2 - x1, y2 - y1))
                    dist_km = float(L / 1000.0)

            rows.append(
                {
                    "etype": params.etype,
                    "ring_id": rid,
                    "t_node_id": tid,
                    "t_node_key": f"T:{int(tid)}",
                    "sea_idx": int(si),
                    "sea_node_id": str(S_nodes.loc[int(si), "node_id"]) if isinstance(S_nodes, pd.DataFrame) and "node_id" in S_nodes.columns and int(si) in S_nodes.index else "",
                    "dist_km": float(dist_km),
                    "reason": reason,
                    "sector_ok": bool(sector_ok),
                    "repaired": bool(repaired),
                }
            )
            accepted += 1
            if accepted >= int(params.k_connect):
                break

    return pd.DataFrame(rows)


def add_tgate_sea_connectors_to_graph(
    G: nx.Graph,
    df_conn: pd.DataFrame,
    *,
    t_node_key_fn: Optional[Callable[[int], object]] = None,
    sea_node_key_fn: Optional[Callable[[int], object]] = None,
    weight_col: str = "dist_km",
    etype_col: str = "etype",
) -> int:
    """
    Add connectors to an existing networkx graph.

    Important: ring node IDs may collide with sea node indices. Use mappers!
    - t_node_key_fn: maps t_node_id -> graph node key (default: identity)
    - sea_node_key_fn: maps sea_idx  -> graph node key (default: identity)
    """
    if t_node_key_fn is None:
        # default to node-id scheme
        t_node_key_fn = lambda tid: f"T:{int(tid)}"
    if sea_node_key_fn is None:
        # WARNING: without sea_node_id column, caller must provide a mapper.
        sea_node_key_fn = lambda sid: int(sid)

    if not isinstance(df_conn, pd.DataFrame) or len(df_conn) == 0:
        return 0

    added = 0
    for r in df_conn.itertuples(index=False):
        tid = int(getattr(r, "t_node_id"))
        sid = int(getattr(r, "sea_idx"))
        w = float(getattr(r, weight_col))
        et = str(getattr(r, etype_col)) if hasattr(r, etype_col) else "T_S_GATE"

        u = str(getattr(r, "t_node_key")) if hasattr(r, "t_node_key") and getattr(r, "t_node_key") else t_node_key_fn(tid)
        v = str(getattr(r, "sea_node_id")) if hasattr(r, "sea_node_id") and getattr(r, "sea_node_id") else sea_node_key_fn(sid)

        # Add nodes if missing (no attributes here; caller can add node attrs elsewhere)
        if u not in G:
            G.add_node(u)
        if v not in G:
            G.add_node(v)

        # Add edge
        G.add_edge(u, v, weight=w, etype=et)
        added += 1

    return added
