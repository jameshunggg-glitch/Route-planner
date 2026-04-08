# routing_map/routing_graph.py
from __future__ import annotations

"""
routing_map.routing_graph (Rings-focused Edition - Fixed)

使用字串 node_id 作為 NetworkX 圖資的 Key。
修正了 ET_edges 缺少 u_key/v_key 欄位時的報錯問題。
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import math
import networkx as nx
import pandas as pd

from routing_map.geom_utils import coord_id, wrap_lon

LonLat = Tuple[float, float]

# -------------------------
# Edge layer / ban masks
# -------------------------
L_BASE_SEA      = 1 << 0
L_RING_E        = 1 << 1
L_RING_T        = 1 << 2
L_ET_TRANSFER   = 1 << 3
L_TGATE_SEA     = 1 << 4
L_GATEWAY       = 1 << 5
L_NE_CORRIDOR   = 1 << 6
L_NW_CORRIDOR   = 1 << 7
L_INJECT        = 1 << 8

B_HIGH_LAT      = 1 << 0

def infer_layer_mask_from_etype(etype: str) -> int:
    e = (etype or "").upper()
    if e in ("E_RING", "E-RING"): return L_RING_E
    if e in ("T_RING", "T-RING"): return L_RING_T
    if e in ("E_T", "ET", "E_T_RAMP", "E_T_TRANSFER"): return L_ET_TRANSFER
    if e in ("T_S_GATE", "TGATE_SEA", "T_GATE_SEA", "T_S"): return L_TGATE_SEA
    if e in ("INJECT",): return L_INJECT
    return L_BASE_SEA

def edge_lat_max_abs(G: nx.Graph, u: str, v: str) -> Optional[float]:
    try:
        lat_u = float(G.nodes[u].get("lat"))
        lat_v = float(G.nodes[v].get("lat"))
        return float(max(abs(lat_u), abs(lat_v)))
    except:
        return None

def compute_edge_masks(G: nx.Graph, u: str, v: str, *, etype: str, hard_lat_cap_deg: float = 70.0) -> Tuple[int, int, Optional[float]]:
    layer = infer_layer_mask_from_etype(etype)
    lat_max = edge_lat_max_abs(G, u, v)
    ban = 0
    if lat_max is not None and lat_max > float(hard_lat_cap_deg):
        ban |= B_HIGH_LAT
    return int(layer), int(ban), lat_max

# -------------------------
# Helpers
# -------------------------
def haversine_km(a: LonLat, b: LonLat) -> float:
    lon1, lat1 = float(a[0]), float(a[1])
    lon2, lat2 = float(b[0]), float(b[1])
    r = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians((lon2 - lon1 + 180) % 360 - 180)
    s = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * r * math.asin(min(1.0, math.sqrt(s)))

def _df(out: Dict[str, Any], key: str) -> Optional[pd.DataFrame]:
    x = out.get(key)
    return x if isinstance(x, pd.DataFrame) else None

def _ring_df(out: Dict[str, Any], key: str) -> Optional[pd.DataFrame]:
    rg = out.get("ring_graph", {}) or {}
    x = rg.get(key)
    return x if isinstance(x, pd.DataFrame) else None

@dataclass
class GraphBuildStats:
    sea_edges_added: int = 0
    e_edges_added: int = 0
    t_edges_added: int = 0
    et_edges_added: int = 0
    tgate_sea_edges_added: int = 0

# -------------------------
# Main Builder
# -------------------------
def build_base_graph(
    out: Dict[str, Any],
    *,
    include_sea: bool = True,
    include_rings: bool = True,
    include_et: bool = True,
    include_tgate_sea: bool = True,
    max_sea_edges: Optional[int] = None,
    max_ring_edges: Optional[int] = None,
    weight_unit: str = "km",
    bbox_ll: Optional[Tuple[float, float, float, float]] = None,
    hard_lat_cap_deg: float = 70.0,
) -> Tuple[nx.Graph, GraphBuildStats]:
    stats = GraphBuildStats()
    G = nx.Graph()

    # 1. Sea
    if include_sea:
        S_nodes = _df(out, "S_nodes")
        S_edges = out.get("S_edges")
        if isinstance(S_nodes, pd.DataFrame) and len(S_nodes):
            for r in S_nodes.itertuples(index=False):
                nid = str(getattr(r, "node_id"))
                G.add_node(nid, lon=wrap_lon(getattr(r, "lon")), lat=getattr(r, "lat"), kind="sea")
                if hasattr(r, "x_m"): G.nodes[nid]["x_m"] = getattr(r, "x_m")
                if hasattr(r, "y_m"): G.nodes[nid]["y_m"] = getattr(r, "y_m")
        if S_edges:
            take = list(S_edges)[:max_sea_edges] if max_sea_edges else S_edges
            for a, b in take:
                u, v = coord_id(a[0], a[1], prefix="S:"), coord_id(b[0], b[1], prefix="S:")
                w = haversine_km(a, b)
                layer, ban, lat_max = compute_edge_masks(G, u, v, etype="sea", hard_lat_cap_deg=hard_lat_cap_deg)
                G.add_edge(u, v, weight=w, length_km=w, etype="sea", layer_mask=layer, ban_mask=ban, lat_max_abs=lat_max)
                stats.sea_edges_added += 1

    # 2. Rings
    if include_rings:
        for prefix, tag in [("E", "E_nodes"), ("T", "T_nodes")]:
            df = _ring_df(out, tag)
            if isinstance(df, pd.DataFrame):
                for r in df.itertuples(index=False):
                    nid = str(getattr(r, "node_key"))
                    G.add_node(nid, lon=wrap_lon(getattr(r, "lon")), lat=getattr(r, "lat"), kind=f"{prefix}_ring")
                    if hasattr(r, "x_m"): G.nodes[nid]["x_m"] = getattr(r, "x_m")
                    if hasattr(r, "y_m"): G.nodes[nid]["y_m"] = getattr(r, "y_m")

        for prefix, tag, etype, stat_attr in [("E", "E_edges", "E_RING", "e_edges_added"), ("T", "T_edges", "T_RING", "t_edges_added")]:
            df = _ring_df(out, tag)
            if isinstance(df, pd.DataFrame):
                take = df.head(max_ring_edges) if max_ring_edges else df
                for r in take.itertuples(index=False):
                    # 容錯處理：檢查是否有 u_key 欄位
                    u = str(getattr(r, "u_key")) if hasattr(r, "u_key") else f"{prefix}:{int(getattr(r, 'u'))}"
                    v = str(getattr(r, "v_key")) if hasattr(r, "v_key") else f"{prefix}:{int(getattr(r, 'v'))}"
                    w = getattr(r, "length_km", None) or haversine_km((G.nodes[u]["lon"], G.nodes[u]["lat"]), (G.nodes[v]["lon"], G.nodes[v]["lat"]))
                    layer, ban, lat_max = compute_edge_masks(G, u, v, etype=etype, hard_lat_cap_deg=hard_lat_cap_deg)
                    G.add_edge(u, v, weight=w, length_km=w, etype=etype, layer_mask=layer, ban_mask=ban, lat_max_abs=lat_max)
                    setattr(stats, stat_attr, getattr(stats, stat_attr) + 1)

    # 3. E <-> T Transfer
    if include_et:
        ET = _ring_df(out, "ET_edges")
        if isinstance(ET, pd.DataFrame):
            for r in ET.itertuples(index=False):
                # 關鍵修正點：
                u = str(getattr(r, "u_key")) if hasattr(r, "u_key") else f"E:{int(getattr(r, 'u'))}"
                v = str(getattr(r, "v_key")) if hasattr(r, "v_key") else f"T:{int(getattr(r, 'v'))}"
                w = getattr(r, "cost_km", 0.0)
                etype = getattr(r, "etype", "E_T")
                layer, ban, lat_max = compute_edge_masks(G, u, v, etype=etype, hard_lat_cap_deg=hard_lat_cap_deg)
                G.add_edge(u, v, weight=w, length_km=w, etype=etype, layer_mask=layer, ban_mask=ban, lat_max_abs=lat_max)
                stats.et_edges_added += 1

    # 4. T-gate -> Sea
    if include_tgate_sea:
        dfTG = out.get("tgate_sea_connectors")
        if isinstance(dfTG, pd.DataFrame):
            for r in dfTG.itertuples(index=False):
                u = str(getattr(r, "t_node_key")) if hasattr(r, "t_node_key") else f"T:{int(getattr(r, 't_node_id'))}"
                v = str(getattr(r, "sea_node_id"))
                w = getattr(r, "dist_km", 0.0)
                etype = getattr(r, "etype", "T_S_GATE")
                layer, ban, lat_max = compute_edge_masks(G, u, v, etype=etype, hard_lat_cap_deg=hard_lat_cap_deg)
                G.add_edge(u, v, weight=w, length_km=w, etype=etype, layer_mask=layer, ban_mask=ban, lat_max_abs=lat_max)
                stats.tgate_sea_edges_added += 1

    return G, stats