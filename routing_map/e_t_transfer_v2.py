
"""
routing_map.e_t_transfer_v2

Build E↔T transfer edges for the ring graph:
- Shared transfer edges: E node ↔ T node for merged (tol) pairs
- Ramp transfer edges (fallback): sparse E anchors connect to nearby T nodes when shared coverage is sparse

Designed to plug into your existing "ring_graph" dict produced by ring_graph.build_ring_nodes_edges(..)
(where T_nodes are generated from the *final* taut polyline after simplification).

Key guarantees:
- Ramp T endpoints are chosen ONLY from T_nodes (i.e., nodes that exist in the T ring graph).
- Collision shrink/buffer is NOT performed in the inner loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree as KDTree  # fast
except Exception:  # pragma: no cover
    KDTree = None

from shapely.geometry import LineString
from shapely.prepared import prep


# -----------------------------
# Config
# -----------------------------
@dataclass
class ETRampConfig:
    # baseline ramp anchors along E ring
    ramp_spacing_km: float = 60.0
    min_ramp_per_ring: int = 2
    near_shared_km: float = 15.0  # skip anchors too close to any shared transfer (along ring distance)
    # T search
    topK_T: int = 12
    k_ramp_per_anchor: int = 1
    ramp_max_km: float = 40.0
    # cost tweak (optional)
    ramp_penalty: float = 0.10
    # geometry checks
    enable_collision_check: bool = True


# -----------------------------
# Small helpers
# -----------------------------
def _euclid_km(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(math.hypot(x2 - x1, y2 - y1) / 1000.0)


def _ring_len_km_from_nodes(df_nodes: pd.DataFrame) -> float:
    """
    Approx ring length from ordered seq nodes: sum consecutive + closure.
    Expects df_nodes includes x_m,y_m and seq.
    """
    if df_nodes.empty:
        return 0.0
    d = df_nodes.sort_values("seq")
    xs = d["x_m"].to_numpy(dtype=float)
    ys = d["y_m"].to_numpy(dtype=float)
    if len(xs) < 2:
        return 0.0
    dx = np.diff(xs)
    dy = np.diff(ys)
    seg = np.hypot(dx, dy).sum()
    close = math.hypot(xs[0] - xs[-1], ys[0] - ys[-1])
    return float((seg + close) / 1000.0)


def _circ_dist_km(s1: float, s2: float, L: float) -> float:
    """Circular distance on [0,L)."""
    if L <= 0:
        return abs(s1 - s2)
    d = abs(s1 - s2)
    return float(min(d, L - d))


def _pick_anchor_nodes_by_spacing(df_e: pd.DataFrame, *, spacing_km: float, min_count: int) -> List[int]:
    """
    Choose E node_ids roughly every spacing_km along s_km.
    Uses nearest node by s_km to target positions.
    """
    if df_e.empty:
        return []
    d = df_e.sort_values("s_km")
    s = d["s_km"].to_numpy(dtype=float)
    ids = d["node_id"].to_numpy(dtype=int)

    L = _ring_len_km_from_nodes(d)
    if L <= 1e-9:
        return [int(ids[0])]

    n_target = max(int(math.ceil(L / max(1e-6, spacing_km))), int(min_count))
    # include 0 and do n_target anchors
    target_s = np.linspace(0.0, L, num=n_target, endpoint=False)

    anchors: List[int] = []
    for ts in target_s:
        # nearest by absolute difference in s (good enough because s is monotonic)
        j = int(np.argmin(np.abs(s - ts)))
        anchors.append(int(ids[j]))

    # de-dup while preserving order
    out: List[int] = []
    seen = set()
    for a in anchors:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


def _is_near_any_shared(e_row: pd.Series, shared_s_list: np.ndarray, ring_len_km: float, near_km: float) -> bool:
    """True if this E point is within near_km (circular) to any shared transfer along the ring."""
    if shared_s_list.size == 0:
        return False
    s0 = float(e_row["s_km"])
    # vectorized circ distance
    d = np.abs(shared_s_list - s0)
    d = np.minimum(d, ring_len_km - d) if ring_len_km > 0 else d
    return bool(np.any(d <= float(near_km)))


def _build_kdtree_xy(df: pd.DataFrame) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (tree, node_id_arr, x_arr, y_arr)
    """
    node_id = df["node_id"].to_numpy(dtype=int)
    x = df["x_m"].to_numpy(dtype=float)
    y = df["y_m"].to_numpy(dtype=float)
    if KDTree is None:
        raise RuntimeError("scipy is required for KDTree-based ramp building (scipy.spatial.cKDTree).")
    tree = KDTree(np.c_[x, y])
    return tree, node_id, x, y


def _segment_hits_collision(x1: float, y1: float, x2: float, y2: float, collision_prep: Any) -> bool:
    seg = LineString([(float(x1), float(y1)), (float(x2), float(y2))])
    return bool(collision_prep.intersects(seg))


# -----------------------------
# Public API
# -----------------------------
def build_e_t_transfer_edges(
    ring_graph: Dict[str, pd.DataFrame],
    *,
    collision_hard_m: Optional[Any] = None,
    cfg: Optional[ETRampConfig] = None,
    shared_edge_cost: float = 0.0,
) -> Dict[str, pd.DataFrame]:
    """
    Build:
      - ET_edges_shared: E_T_SHARED edges from Shared_nodes (0-cost by default)
      - ET_edges_ramp:   E_T_RAMP edges as sparse fallback anchors

    Returns dict with:
      - ET_edges (combined)
      - ET_edges_shared
      - ET_edges_ramp
      - ET_stats (per ring summary)
    """
    if cfg is None:
        cfg = ETRampConfig()

    E_nodes = ring_graph.get("E_nodes", pd.DataFrame()).copy()
    T_nodes = ring_graph.get("T_nodes", pd.DataFrame()).copy()
    Shared_nodes = ring_graph.get("Shared_nodes", pd.DataFrame()).copy()

    if E_nodes.empty or T_nodes.empty:
        return {
            "ET_edges": pd.DataFrame(columns=["etype", "ring_id", "u", "v", "length_km", "cost_km", "reason"]),
            "ET_edges_shared": pd.DataFrame(),
            "ET_edges_ramp": pd.DataFrame(),
            "ET_stats": pd.DataFrame(),
        }

    # Prepare collision once (critical)
    collision_prep = None
    if cfg.enable_collision_check and collision_hard_m is not None and (not getattr(collision_hard_m, "is_empty", True)):
        collision_prep = prep(collision_hard_m)

    # -----------------
    # Shared edges
    # -----------------
    shared_rows: List[Dict[str, Any]] = []
    if not Shared_nodes.empty:
        for r in Shared_nodes.itertuples(index=False):
            ring_id = int(getattr(r, "ring_id"))
            u = int(getattr(r, "e_node_id"))
            v = int(getattr(r, "t_node_id"))
            shared_rows.append(
                dict(
                    etype="E_T_SHARED",
                    ring_id=ring_id,
                    u=u,
                    v=v,
                    length_km=0.0,
                    cost_km=float(shared_edge_cost),
                    reason="shared",
                )
            )

    ET_edges_shared = pd.DataFrame(shared_rows)

    # -----------------
    # Ramp edges (baseline E-driven)
    # -----------------
    ramp_rows: List[Dict[str, Any]] = []
    stats_rows: List[Dict[str, Any]] = []

    # Quick access maps
    e_by_id = E_nodes.set_index("node_id")
    # group by ring
    for ring_id, df_e in E_nodes.groupby("ring_id"):
        ring_id = int(ring_id)
        df_t = T_nodes[T_nodes["ring_id"] == ring_id]
        if df_t.empty or df_e.empty:
            continue

        ring_len_km = _ring_len_km_from_nodes(df_e)

        # shared s positions on E
        shared_e_ids = set()
        if not Shared_nodes.empty:
            shared_e_ids = set(Shared_nodes.loc[Shared_nodes["ring_id"] == ring_id, "e_node_id"].astype(int).tolist())
        shared_s = df_e[df_e["node_id"].isin(shared_e_ids)]["s_km"].to_numpy(dtype=float) if shared_e_ids else np.array([], dtype=float)

        # anchor selection
        anchors = _pick_anchor_nodes_by_spacing(
            df_e,
            spacing_km=float(cfg.ramp_spacing_km),
            min_count=int(cfg.min_ramp_per_ring),
        )

        # build KDTree on T nodes
        tree, t_ids, t_x, t_y = _build_kdtree_xy(df_t)

        n_attempt = 0
        n_ok = 0

        for e_id in anchors:
            if e_id in shared_e_ids:
                continue
            e_row = e_by_id.loc[e_id]
            if cfg.near_shared_km > 0 and _is_near_any_shared(e_row, shared_s, ring_len_km, float(cfg.near_shared_km)):
                continue

            ex = float(e_row["x_m"])
            ey = float(e_row["y_m"])

            # query candidates
            n_attempt += 1
            kq = max(1, int(cfg.topK_T))
            dist, idx = tree.query([ex, ey], k=min(kq, len(t_ids)))
            # normalize outputs
            if np.isscalar(dist):
                dist = np.array([dist], dtype=float)
                idx = np.array([idx], dtype=int)
            else:
                dist = np.asarray(dist, dtype=float)
                idx = np.asarray(idx, dtype=int)

            chosen = 0
            for j in np.argsort(dist):
                tid = int(t_ids[int(idx[j])])
                tx = float(t_x[int(idx[j])])
                ty = float(t_y[int(idx[j])])
                d_km = _euclid_km(ex, ey, tx, ty)
                if d_km > float(cfg.ramp_max_km):
                    continue
                if collision_prep is not None:
                    if _segment_hits_collision(ex, ey, tx, ty, collision_prep):
                        continue

                # accept
                length_km = float(d_km)
                cost_km = float(length_km * (1.0 + float(cfg.ramp_penalty)))
                ramp_rows.append(
                    dict(
                        etype="E_T_RAMP",
                        ring_id=ring_id,
                        u=int(e_id),
                        v=int(tid),
                        length_km=length_km,
                        cost_km=cost_km,
                        reason="anchor_spacing",
                    )
                )
                n_ok += 1
                chosen += 1
                if chosen >= int(cfg.k_ramp_per_anchor):
                    break

        stats_rows.append(
            dict(
                ring_id=ring_id,
                ring_len_km=float(ring_len_km),
                n_shared=int(len(shared_e_ids)),
                n_anchor=int(len(anchors)),
                n_attempt=int(n_attempt),
                n_ramps=int(n_ok),
                ramp_spacing_km=float(cfg.ramp_spacing_km),
                ramp_max_km=float(cfg.ramp_max_km),
            )
        )

    ET_edges_ramp = pd.DataFrame(ramp_rows)
    ET_stats = pd.DataFrame(stats_rows)

    # combine
    if not ET_edges_shared.empty and not ET_edges_ramp.empty:
        ET_edges = pd.concat([ET_edges_shared, ET_edges_ramp], ignore_index=True)
    elif not ET_edges_shared.empty:
        ET_edges = ET_edges_shared.copy()
    else:
        ET_edges = ET_edges_ramp.copy()

    
    # --- Global graph string keys (keep int ids for internal computations)
    for _df in [ET_edges_shared, ET_edges_ramp]:
        if isinstance(_df, pd.DataFrame) and len(_df) and "u" in _df.columns and "v" in _df.columns:
            _df["u_key"] = _df["u"].map(lambda i: f"E:{int(i)}")
            _df["v_key"] = _df["v"].map(lambda i: f"T:{int(i)}")
    return {
        "ET_edges": ET_edges,
        "ET_edges_shared": ET_edges_shared,
        "ET_edges_ramp": ET_edges_ramp,
        "ET_stats": ET_stats,
    }