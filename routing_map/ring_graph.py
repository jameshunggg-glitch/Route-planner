
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import math
import numpy as np
import pandas as pd

from .ring_types import RingBuildConfig, RingResult, XY


# -----------------------------
# Projector helpers (robust)
# -----------------------------
def _get_proj_fn(proj: Any, name: str) -> Optional[Callable]:
    """
    Robust projector function getter.
    Expected names: "ll2m", "m2ll".
    Also supports legacy transformer names: to_m/to_ll (Transformer.transform).
    """
    if proj is None:
        return None

    # dict-like
    if isinstance(proj, dict):
        # preferred
        if name in proj and callable(proj[name]):
            return proj[name]
        # legacy transformers
        if name == "m2ll" and "to_ll" in proj and hasattr(proj["to_ll"], "transform"):
            return proj["to_ll"].transform
        if name == "ll2m" and "to_m" in proj and hasattr(proj["to_m"], "transform"):
            return proj["to_m"].transform
        return None

    # object-like (AOIProjector)
    if hasattr(proj, name) and callable(getattr(proj, name)):
        return getattr(proj, name)

    # legacy: transformers on object
    if name == "m2ll" and hasattr(proj, "to_ll") and hasattr(proj.to_ll, "transform"):
        return proj.to_ll.transform
    if name == "ll2m" and hasattr(proj, "to_m") and hasattr(proj.to_m, "transform"):
        return proj.to_m.transform

    return None



def _m2ll_safe(proj: Any, x: float, y: float) -> Tuple[float, float]:
    fn = _get_proj_fn(proj, "m2ll")
    if fn is None:
        return (float("nan"), float("nan"))
    try:
        # try signature fn(x,y) first (AOIProjector.m2ll, Transformer.transform both work)
        lon, lat = fn(float(x), float(y))
        return (float(lon), float(lat))
    except TypeError:
        try:
            # alternate signature fn((x,y))
            lon, lat = fn((float(x), float(y)))
            return (float(lon), float(lat))
        except Exception:
            return (float("nan"), float("nan"))
    except Exception:
        return (float("nan"), float("nan"))




# -----------------------------
# Geometry helpers (metric)
# -----------------------------
def _dist_m(a: XY, b: XY) -> float:
    return float(math.hypot(b[0] - a[0], b[1] - a[1]))


def _cum_s_km_closed(pts: List[XY]) -> List[float]:
    """
    For a closed ring represented by UNIQUE vertices (NOT duplicated last),
    compute cumulative distance s_km per vertex (starting at 0).
    """
    n = len(pts)
    if n == 0:
        return []
    s = [0.0]
    acc = 0.0
    for i in range(1, n):
        acc += _dist_m(pts[i-1], pts[i]) / 1000.0
        s.append(acc)
    return s


def _turn_angle_deg(prev: XY, cur: XY, nxt: XY) -> float:
    """
    Signed turning angle at 'cur' when walking prev->cur->nxt.
    Returns degrees in [-180, 180]. We often use abs(angle).
    """
    ax, ay = cur[0] - prev[0], cur[1] - prev[1]
    bx, by = nxt[0] - cur[0], nxt[1] - cur[1]
    # protect zero-length
    an = math.hypot(ax, ay)
    bn = math.hypot(bx, by)
    if an < 1e-9 or bn < 1e-9:
        return 0.0
    ax, ay = ax / an, ay / an
    bx, by = bx / bn, by / bn
    dot = max(-1.0, min(1.0, ax * bx + ay * by))
    cross = ax * by - ay * bx
    ang = math.degrees(math.atan2(cross, dot))
    return float(ang)


def _unique_closed_pts(pts_m_closed: List[XY]) -> List[XY]:
    """
    Convert possibly-closed (last==first) list into UNIQUE vertices (no duplicate last).
    """
    if not pts_m_closed:
        return []
    pts = [(float(x), float(y)) for x, y in pts_m_closed]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def _densify_segment(a: XY, b: XY, max_gap_km: float) -> List[XY]:
    """
    Return interior points (excluding endpoints) so that resulting gaps <= max_gap_km.
    """
    max_gap_m = float(max_gap_km) * 1000.0
    d = _dist_m(a, b)
    if d <= max_gap_m or max_gap_m <= 0:
        return []
    k = int(math.ceil(d / max_gap_m))  # number of segments
    # need k segments => add (k-1) interior points
    out: List[XY] = []
    for i in range(1, k):
        t = i / k
        out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out

# -----------------------------
# T node angle degs
# -----------------------------

def add_t_turn_angles(T_nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Add/overwrite T_nodes['t_angle_deg'] using metric xy and seq order (cycle-aware).
    """
    if T_nodes is None or len(T_nodes) == 0:
        return T_nodes

    T_nodes = T_nodes.copy()
    if "t_angle_deg" not in T_nodes.columns:
        T_nodes["t_angle_deg"] = 0.0

    for rid, df_ring in T_nodes.groupby("ring_id", sort=False):
        df_ring = df_ring.sort_values("seq")
        idx = df_ring.index.to_list()
        pts = list(zip(df_ring["x_m"].astype(float).values, df_ring["y_m"].astype(float).values))
        n = len(pts)
        if n < 3:
            T_nodes.loc[idx, "t_angle_deg"] = 0.0
            continue

        angs = []
        for i in range(n):
            prev = pts[(i - 1) % n]
            cur  = pts[i]
            nxt  = pts[(i + 1) % n]
            angs.append(_turn_angle_deg(prev, cur, nxt))

        T_nodes.loc[idx, "t_angle_deg"] = np.array(angs, dtype=float)

    return T_nodes

# -----------------------------
# T gate candidate selection
# -----------------------------
def _circular_dist_km(s1: float, s2: float, L: float) -> float:
    """Distance along a closed ring between two arc-length positions s1,s2."""
    d = abs(float(s1) - float(s2))
    return float(min(d, max(0.0, L - d))) if L > 0 else float(d)


def _ring_length_km_from_s(s_km: np.ndarray) -> float:
    if len(s_km) == 0:
        return 0.0
    return float(np.nanmax(s_km))


def _pick_nearest_by_s(
    candidates: pd.DataFrame,
    target_s: float,
    selected_s: List[float],
    *,
    L: float,
    min_sep_km: float,
) -> Optional[int]:
    """Pick candidate node_id nearest to target_s, respecting min separation to selected."""
    if candidates.empty:
        return None
    s_vals = candidates["s_km"].astype(float).values
    d_raw = np.abs(s_vals - float(target_s))
    d_to_target = np.minimum(d_raw, np.maximum(0.0, float(L) - d_raw))
    order = np.argsort(d_to_target)
    for idx in order:
        nid = int(candidates.iloc[int(idx)]["node_id"])
        s = float(candidates.iloc[int(idx)]["s_km"])
        if all(_circular_dist_km(s, ss, L) >= float(min_sep_km) for ss in selected_s):
            return nid
    return None


def _pick_farthest_point(
    candidates: pd.DataFrame,
    selected_s: List[float],
    *,
    L: float,
) -> Optional[int]:
    """Pick candidate maximizing its minimum distance to selected points."""
    if candidates.empty:
        return None
    if not selected_s:
        # choose one near s=0 for determinism
        i0 = int(np.nanargmin(candidates["s_km"].astype(float).values))
        return int(candidates.iloc[i0]["node_id"])
    best_nid = None
    best_score = -1.0
    for r in candidates.itertuples(index=False):
        s = float(getattr(r, "s_km"))
        mind = min(_circular_dist_km(s, ss, L) for ss in selected_s)
        if mind > best_score:
            best_score = mind
            best_nid = int(getattr(r, "node_id"))
    return best_nid


def select_t_gate_candidates(
    E_nodes: pd.DataFrame,
    T_nodes: pd.DataFrame,
    Shared_nodes: pd.DataFrame,
    *,
    params: RingGraphBuildParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mark T_nodes['is_gate_candidate']=True following selection v1.

    Tier-0 (hard): shared nodes whose corresponding E-node abs(angle_deg) >= t_gate_ang_shared_deg
    Tier-1 (soft): add more along ring by spacing (prefer vertex), respecting min sep
    Tier-2 (fallback): ensure at least t_gate_min_per_ring per ring by farthest-point picks
    Soft cap: keep all hard gates, downsample soft ones if above t_gate_max_per_ring

    Returns: (updated T_nodes, T_gate_candidates df)
    """
    if T_nodes is None or len(T_nodes) == 0:
        T_nodes = (T_nodes.copy() if T_nodes is not None else pd.DataFrame())
        if "is_gate_candidate" not in T_nodes.columns:
            T_nodes["is_gate_candidate"] = False
        if "gate_reason" not in T_nodes.columns:
            T_nodes["gate_reason"] = ""
        return T_nodes, T_nodes.iloc[0:0].copy()

    T_nodes = T_nodes.copy()
    if "is_gate_candidate" not in T_nodes.columns:
        T_nodes["is_gate_candidate"] = False
    if "gate_reason" not in T_nodes.columns:
        T_nodes["gate_reason"] = ""

    # map e_node_id -> abs(angle_deg)
    e_ang: Dict[int, float] = {}
    if E_nodes is not None and len(E_nodes) > 0 and "angle_deg" in E_nodes.columns:
        for r in E_nodes.itertuples(index=False):
            e_ang[int(getattr(r, "node_id"))] = abs(float(getattr(r, "angle_deg")))

    # shared mapping per ring: t_node_id -> e_angle
    hard_by_ring: Dict[int, Dict[int, float]] = {}
    if Shared_nodes is not None and len(Shared_nodes) > 0:
        for r in Shared_nodes.itertuples(index=False):
            rid = int(getattr(r, "ring_id"))
            tid = int(getattr(r, "t_node_id"))
            eid = int(getattr(r, "e_node_id"))
            ang = e_ang.get(eid, 0.0)
            if ang >= float(params.t_gate_ang_shared_deg):
                hard_by_ring.setdefault(rid, {})[tid] = ang

    # per ring selection
    for rid, df_ring in T_nodes.groupby("ring_id", sort=False):
        ring_idx = df_ring.index
        L = _ring_length_km_from_s(df_ring["s_km"].astype(float).values)

        if params.t_gate_prefer_vertex and "kind" in df_ring.columns:
            cand_vertex = df_ring[df_ring["kind"].astype(str) == "vertex"]
            cand_any = df_ring
        else:
            cand_vertex = df_ring
            cand_any = df_ring

        selected: Dict[int, str] = {}

        # Tier-0 hard
        for tid in hard_by_ring.get(int(rid), {}).keys():
            selected[int(tid)] = "shared_angle"

        def selected_s_list() -> List[float]:
            if not selected:
                return []
            sub = df_ring[df_ring["node_id"].isin(list(selected.keys()))]
            return sub["s_km"].astype(float).tolist()

        # Tier-1 spacing
        spacing = float(params.t_gate_spacing_km)
        min_sep = float(params.t_gate_min_sep_km)
        if L > 0 and spacing > 0:
            n_targets = int(math.floor(L / spacing))
            targets = [i * spacing for i in range(n_targets + 1)]
            for ts in targets:
                sel_s = selected_s_list()
                nid = _pick_nearest_by_s(cand_vertex, ts, sel_s, L=L, min_sep_km=min_sep)
                if nid is None:
                    nid = _pick_nearest_by_s(cand_any, ts, sel_s, L=L, min_sep_km=min_sep)
                if nid is not None and nid not in selected:
                    selected[int(nid)] = "spacing"

        # Tier-2 fallback min per ring
        minN = int(params.t_gate_min_per_ring)
        guard = 0
        while len(selected) < minN and guard < 10_000:
            guard += 1
            sel_s = selected_s_list()
            nid = _pick_farthest_point(cand_vertex, sel_s, L=L)
            if nid is None:
                nid = _pick_farthest_point(cand_any, sel_s, L=L)
            if nid is None:
                break
            if nid not in selected:
                selected[int(nid)] = "fallback"
            else:
                break

        # Soft cap (keep all hard)
        cap = int(params.t_gate_max_per_ring)
        if cap > 0 and len(selected) > cap:
            hard = [nid for nid, reason in selected.items() if reason == "shared_angle"]
            soft = [nid for nid, reason in selected.items() if reason != "shared_angle"]
            keep = set(hard)
            if len(keep) < cap and soft:
                soft_df = df_ring[df_ring["node_id"].isin(soft)].sort_values("s_km")
                k = max(0, cap - len(keep))
                if k > 0 and len(soft_df) > 0:
                    idxs = np.linspace(0, len(soft_df) - 1, num=min(k, len(soft_df)), dtype=int)
                    keep.update(soft_df.iloc[idxs]["node_id"].astype(int).tolist())
            selected = {nid: selected[nid] for nid in list(selected.keys()) if nid in keep}

        sel_ids = set(selected.keys())
        T_nodes.loc[ring_idx, "is_gate_candidate"] = T_nodes.loc[ring_idx, "node_id"].isin(sel_ids)
        T_nodes.loc[ring_idx, "gate_reason"] = ""
        for nid, reason in selected.items():
            T_nodes.loc[T_nodes["node_id"] == nid, "gate_reason"] = reason

    T_gate_candidates = T_nodes[T_nodes["is_gate_candidate"]].copy()
    return T_nodes, T_gate_candidates

# -----------------------------
# Builders
# -----------------------------
@dataclass
class RingGraphBuildParams:
    # ---- E nodes ----
    e_angle_feature_deg: float = 25.0   # mark as feature if abs(turn) >= this

    # ---- T nodes ----
    t_max_gap_km: float = 20.0          # densify taut edges if longer than this

    # ---- Shared merge ----
    shared_tol_m: float = 25.0          # merge tol in meters

    # ---- T gate candidates (selection v1) ----
    # Hard include: shared nodes whose corresponding E-node has abs(angle_deg) >= threshold.
    t_gate_ang_shared_deg: float = 15.0

    # Soft include: add more gates along ring by spacing on T s_km.
    t_gate_spacing_km: float = 60.0     # target spacing between gates along ring
    t_gate_min_sep_km: float = 20.0     # minimum separation between selected gates

    # Per-ring safety / caps
    t_gate_min_per_ring: int = 2        # ensure at least N gates per ring
    t_gate_max_per_ring: int = 25       # (soft) cap for gates per ring

    # Candidate preference: choose among T nodes with kind=="vertex" first
    t_gate_prefer_vertex: bool = True


def build_ring_nodes_edges(
    rings: List[RingResult],
    *,
    proj: Any,
    cfg: RingBuildConfig,
    params: Optional[RingGraphBuildParams] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build E/T/shared nodes and E/T edges from ring results.

    Returns dict of DataFrames:
      - E_nodes, T_nodes, Shared_nodes
      - E_edges, T_edges
    """
    if params is None:
        params = RingGraphBuildParams()

    e_nodes_rows: List[Dict[str, Any]] = []
    t_nodes_rows: List[Dict[str, Any]] = []
    shared_rows: List[Dict[str, Any]] = []
    e_edges_rows: List[Dict[str, Any]] = []
    t_edges_rows: List[Dict[str, Any]] = []

    # temporary structures per ring for shared matching
    all_e_xy: List[Tuple[int, int, XY]] = []  # (ring_id, e_node_id, (x,y))
    all_t_xy: List[Tuple[int, int, XY]] = []  # (ring_id, t_node_id, (x,y))

    e_node_id = 0
    t_node_id = 0

    for r in rings:
        ring_id = int(r.ring_id)

        # ---- E nodes (from envelope pts as-is: already approx equal spacing cfg.ring_sample_km)
        e_pts = _unique_closed_pts(r.envelope_pts_m)
        nE = len(e_pts)
        if nE >= 3:
            s_km = _cum_s_km_closed(e_pts)
            for i, (x, y) in enumerate(e_pts):
                prev = e_pts[(i - 1) % nE]
                nxt = e_pts[(i + 1) % nE]
                ang = _turn_angle_deg(prev, (x, y), nxt)
                kind = "feature" if abs(ang) >= float(params.e_angle_feature_deg) else "eq"
                lon, lat = _m2ll_safe(proj, x, y)
                row = dict(
                    node_id=e_node_id,
                    ring_id=ring_id,
                    seq=int(i),
                    x_m=float(x),
                    y_m=float(y),
                    lon=float(lon),
                    lat=float(lat),
                    s_km=float(s_km[i]),
                    angle_deg=float(ang),
                    kind=str(kind),
                    is_shared=False,
                )
                e_nodes_rows.append(row)
                all_e_xy.append((ring_id, e_node_id, (float(x), float(y))))
                e_node_id += 1

            # E edges along seq + closure
            # edges reference node_id; we need ring-local mapping seq->node_id
            ring_e_node_ids = [row["node_id"] for row in e_nodes_rows[-nE:]]
            for i in range(nE):
                u = ring_e_node_ids[i]
                v = ring_e_node_ids[(i + 1) % nE]
                e_edges_rows.append(dict(
                    etype="E_RING",
                    ring_id=ring_id,
                    u=int(u),
                    v=int(v),
                    seq=int(i),
                ))

        # ---- T nodes (from taut pts, densified)
        t_pts0 = _unique_closed_pts(r.taut_pts_m if r.taut_pts_m else r.envelope_pts_m)
        nT0 = len(t_pts0)
        if nT0 >= 3:
            # build densified sequence
            t_pts: List[Tuple[XY, str]] = []  # (pt, kind)
            for i in range(nT0):
                a = t_pts0[i]
                b = t_pts0[(i + 1) % nT0]
                t_pts.append((a, "vertex"))
                fills = _densify_segment(a, b, max_gap_km=float(params.t_max_gap_km))
                for p in fills:
                    t_pts.append((p, "fill"))

            # remove possible duplicates due to densify edge cases
            pts_only = [p for p, _ in t_pts]
            kinds = [k for _, k in t_pts]
            # if last equals first, drop last
            if len(pts_only) >= 2 and pts_only[0] == pts_only[-1]:
                pts_only = pts_only[:-1]
                kinds = kinds[:-1]

            nT = len(pts_only)
            s_km_t = _cum_s_km_closed(pts_only)
            ring_t_node_ids: List[int] = []
            for i, (pt, kind) in enumerate(zip(pts_only, kinds)):
                x, y = pt
                lon, lat = _m2ll_safe(proj, x, y)
                row = dict(
                    node_id=t_node_id,
                    ring_id=ring_id,
                    seq=int(i),
                    x_m=float(x),
                    y_m=float(y),
                    lon=float(lon),
                    lat=float(lat),
                    s_km=float(s_km_t[i]),
                    kind=str(kind),
                    is_gate_candidate=False,  # gate selection later
                    gate_reason="",
                    is_shared=False,
                )
                t_nodes_rows.append(row)
                all_t_xy.append((ring_id, t_node_id, (float(x), float(y))))
                ring_t_node_ids.append(t_node_id)
                t_node_id += 1

            # T edges along seq + closure
            for i in range(nT):
                u = ring_t_node_ids[i]
                v = ring_t_node_ids[(i + 1) % nT]
                t_edges_rows.append(dict(
                    etype="T_RING",
                    ring_id=ring_id,
                    u=int(u),
                    v=int(v),
                    seq=int(i),
                ))

    # ---- Shared nodes: merge E and T within tol per ring
    tol = float(params.shared_tol_m)
    # index T nodes by quantized grid per ring
    grid = max(1.0, tol)
    t_index: Dict[Tuple[int, int, int], List[Tuple[int, XY]]] = {}
    for ring_id, tid, (x, y) in all_t_xy:
        gx = int(round(x / grid))
        gy = int(round(y / grid))
        key = (int(ring_id), gx, gy)
        t_index.setdefault(key, []).append((tid, (x, y)))

    # for each E, search in neighboring grid cells
    shared_id = 0
    e_is_shared = set()
    t_is_shared = set()

    for ring_id, eid, (x, y) in all_e_xy:
        gx = int(round(x / grid))
        gy = int(round(y / grid))
        found = None
        best_d = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (int(ring_id), gx + dx, gy + dy)
                for tid, (tx, ty) in t_index.get(key, []):
                    d = math.hypot(tx - x, ty - y)
                    if d <= tol and (best_d is None or d < best_d):
                        best_d = d
                        found = (tid, tx, ty)
        if found is not None:
            tid, tx, ty = found
            e_is_shared.add(eid)
            t_is_shared.add(tid)
            lon, lat = _m2ll_safe(proj, x, y)
            shared_rows.append(dict(
                shared_id=int(shared_id),
                ring_id=int(ring_id),
                e_node_id=int(eid),
                t_node_id=int(tid),
                x_m=float(x),
                y_m=float(y),
                lon=float(lon),
                lat=float(lat),
                tol_m=float(best_d if best_d is not None else 0.0),
            ))
            shared_id += 1

    # mark flags in node tables
    if e_nodes_rows:
        for row in e_nodes_rows:
            if row["node_id"] in e_is_shared:
                row["is_shared"] = True
    if t_nodes_rows:
        for row in t_nodes_rows:
            if row["node_id"] in t_is_shared:
                row["is_shared"] = True

    E_nodes = pd.DataFrame(e_nodes_rows)
    T_nodes = pd.DataFrame(t_nodes_rows)
    Shared_nodes = pd.DataFrame(shared_rows)

    T_nodes = add_t_turn_angles(T_nodes)

    # ---- Select T gate candidates (updates T_nodes + returns a convenient subset table)
    T_nodes, T_gate_candidates = select_t_gate_candidates(
        E_nodes, T_nodes, Shared_nodes, params=params
    )
    E_edges = pd.DataFrame(e_edges_rows)
    T_edges = pd.DataFrame(t_edges_rows)


    # --- Node/edge keys for global graph (keep int ids internally, expose string keys)
    if isinstance(E_nodes, pd.DataFrame) and "node_id" in E_nodes.columns:
        E_nodes["node_key"] = E_nodes["node_id"].map(lambda i: f"E:{int(i)}")
    if isinstance(T_nodes, pd.DataFrame) and "node_id" in T_nodes.columns:
        T_nodes["node_key"] = T_nodes["node_id"].map(lambda i: f"T:{int(i)}")
    if isinstance(Shared_nodes, pd.DataFrame):
        if "e_node_id" in Shared_nodes.columns:
            Shared_nodes["e_key"] = Shared_nodes["e_node_id"].map(lambda i: f"E:{int(i)}")
        if "t_node_id" in Shared_nodes.columns:
            Shared_nodes["t_key"] = Shared_nodes["t_node_id"].map(lambda i: f"T:{int(i)}")

    if isinstance(E_edges, pd.DataFrame) and "u" in E_edges.columns and "v" in E_edges.columns:
        E_edges["u_key"] = E_edges["u"].map(lambda i: f"E:{int(i)}")
        E_edges["v_key"] = E_edges["v"].map(lambda i: f"E:{int(i)}")
    if isinstance(T_edges, pd.DataFrame) and "u" in T_edges.columns and "v" in T_edges.columns:
        T_edges["u_key"] = T_edges["u"].map(lambda i: f"T:{int(i)}")
        T_edges["v_key"] = T_edges["v"].map(lambda i: f"T:{int(i)}")

    # If gate candidates exist, add t_node_key for downstream connectors
    if isinstance(T_gate_candidates, pd.DataFrame) and "t_node_id" in T_gate_candidates.columns:
        T_gate_candidates["t_node_key"] = T_gate_candidates["t_node_id"].map(lambda i: f"T:{int(i)}")

    
    return {
        "E_nodes": E_nodes,
        "T_nodes": T_nodes,
        "Shared_nodes": Shared_nodes,
        "E_edges": E_edges,
        "T_edges": T_edges,
        "T_gate_candidates": T_gate_candidates,
    }
