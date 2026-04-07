from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LonLat = Tuple[float, float]
BBoxLL = Tuple[float, float, float, float]


def _in_bbox(p: LonLat, bbox_ll: Optional[BBoxLL]) -> bool:
    if bbox_ll is None:
        return True
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)
    lon, lat = float(p[0]), float(p[1])
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)


def _safe_df(out: Dict[str, Any], key: str):
    obj = out.get(key)
    if isinstance(obj, pd.DataFrame) and len(obj):
        return obj
    return None


def _lonlat_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    for lc, la in [("lon", "lat"), ("Long", "Lat"), ("longitude", "latitude"), ("x", "y")]:
        if lc in df.columns and la in df.columns:
            return lc, la
    return None, None


def collect_candidates(
    out: Dict[str, Any],
    *,
    bbox_ll: Optional[BBoxLL] = None,
    max_per_kind: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, List[LonLat]]:
    """Collect candidate pools from out.

    Returns dict with keys: "E", "T", "TGATE", "S".
    - E/T from out["ring_graph"]["E_nodes"/"T_nodes"] with required column "node_id".
    - TGATE from out["tgate_sea_connectors"] mapping t_node_id -> T_nodes node_ll.
    - S from out["S_nodes"] (sea nodes df).
    """
    rng = np.random.default_rng(seed)

    pools: Dict[str, List[LonLat]] = {"E": [], "T": [], "TGATE": [], "S": []}

    rg = out.get("ring_graph") or {}
    E_nodes = rg.get("E_nodes")
    T_nodes = rg.get("T_nodes")

    def _collect_from_df(df, key):
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return
        lon_col, lat_col = _lonlat_cols(df)
        if not lon_col or not lat_col:
            return
        pts = [(float(r[lon_col]), float(r[lat_col])) for _, r in df.iterrows()]
        if bbox_ll:
            pts = [p for p in pts if _in_bbox(p, bbox_ll)]
        if max_per_kind and len(pts) > max_per_kind:
            idx = rng.choice(len(pts), size=int(max_per_kind), replace=False)
            pts = [pts[i] for i in idx]
        pools[key] = pts

    _collect_from_df(E_nodes, "E")
    _collect_from_df(T_nodes, "T")

    # Sea nodes
    S_nodes = _safe_df(out, "S_nodes")
    if S_nodes is not None:
        lon_col, lat_col = _lonlat_cols(S_nodes)
        if lon_col and lat_col:
            pts = [(float(r[lon_col]), float(r[lat_col])) for _, r in S_nodes.iterrows()]
            if bbox_ll:
                pts = [p for p in pts if _in_bbox(p, bbox_ll)]
            if max_per_kind and len(pts) > max_per_kind:
                idx = rng.choice(len(pts), size=int(max_per_kind), replace=False)
                pts = [pts[i] for i in idx]
            pools["S"] = pts

    # TGATE pool: t_node_id from connectors -> t node lonlat
    dfTG = _safe_df(out, "tgate_sea_connectors")
    if isinstance(T_nodes, pd.DataFrame) and len(T_nodes) and dfTG is not None:
        if "node_id" in T_nodes.columns:
            lon_col, lat_col = _lonlat_cols(T_nodes)
            if lon_col and lat_col:
                tmap = {int(r["node_id"]): (float(r[lon_col]), float(r[lat_col])) for _, r in T_nodes.iterrows()}
                tcol = "t_node_id" if "t_node_id" in dfTG.columns else ("t_id" if "t_id" in dfTG.columns else None)
                if tcol:
                    tg_pts = []
                    for _, r in dfTG.iterrows():
                        try:
                            tid = int(r[tcol])
                        except Exception:
                            continue
                        if tid in tmap:
                            tg_pts.append(tmap[tid])
                    # unique
                    seen = set()
                    uniq = []
                    for p in tg_pts:
                        key = (round(p[0], 7), round(p[1], 7))
                        if key in seen:
                            continue
                        seen.add(key)
                        uniq.append(p)
                    if bbox_ll:
                        uniq = [p for p in uniq if _in_bbox(p, bbox_ll)]
                    if max_per_kind and len(uniq) > max_per_kind:
                        idx = rng.choice(len(uniq), size=int(max_per_kind), replace=False)
                        uniq = [uniq[i] for i in idx]
                    pools["TGATE"] = uniq

    return pools


def filter_candidates_in_graph(
    G,
    pools: Dict[str, List[LonLat]],
    *,
    min_degree: int = 1,
) -> Dict[str, List[LonLat]]:
    """Filter pools by whether the point exists as a node in G (and has degree >= min_degree)."""
    out: Dict[str, List[LonLat]] = {}
    for k, pts in pools.items():
        kept: List[LonLat] = []
        for p in pts:
            if p in G:
                try:
                    if G.degree[p] >= int(min_degree):
                        kept.append(p)
                except Exception:
                    kept.append(p)
        out[k] = kept
    return out


def pick_pair(
    pools: Dict[str, List[LonLat]],
    route_mode: str,
    *,
    seed: Optional[int] = None,
) -> Tuple[Optional[LonLat], Optional[LonLat]]:
    """Pick (start,end) based on route_mode.

    route_mode examples:
      - "e_to_e", "e_to_t", "t_to_t", "t_to_tgate", "tgate_to_sea", "sea_to_sea", "sea_to_tgate"
    """
    rng = np.random.default_rng(seed)
    a_kind, b_kind = None, None

    mode = (route_mode or "").lower().strip()
    if "_to_" in mode:
        a_kind, b_kind = mode.split("_to_", 1)
        a_kind = a_kind.upper()
        b_kind = b_kind.upper()
        if a_kind == "SEY":
            a_kind = "S"
    else:
        # default
        a_kind, b_kind = "E", "TGATE"

    if a_kind not in pools or b_kind not in pools:
        return None, None

    A = pools[a_kind]
    B = pools[b_kind]
    if not A or not B:
        return None, None

    a = A[int(rng.integers(len(A)))]
    b = B[int(rng.integers(len(B)))]
    return a, b


__all__ = [
    "collect_candidates",
    "filter_candidates_in_graph",
    "pick_pair",
]
