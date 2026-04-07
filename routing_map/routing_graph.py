from __future__ import annotations

"""routing_map.routing_graph (node_id edition)

This version uses **string node_id** keys everywhere in the assembled NetworkX graph.

Node key scheme:
- Sea nodes:  "S:{lon:.6f},{lat:.6f}"  (lon wrapped to [-180,180))
- Envelope ring nodes: "E:{int_id}"
- Taut ring nodes:     "T:{int_id}"
- Injected query nodes: "Q:START", "Q:END" (in pipeline)

Edges store:
- weight (km)
- length_km (km)
- etype (string)
- layer_mask (bitmask)
- ban_mask (bitmask)
- lat_max_abs (float)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import math
import networkx as nx
import pandas as pd

from .geom_utils import coord_id, wrap_lon


LonLat = Tuple[float, float]
# -------------------------
# Edge layer / ban masks (bitmask)
# -------------------------

# Layer bits: an edge can belong to multiple layers (OR together).
L_BASE_SEA      = 1 << 0  # normal sea edges
L_RING_E        = 1 << 1  # E-ring edges
L_RING_T        = 1 << 2  # T-ring edges
L_ET_TRANSFER   = 1 << 3  # E<->T transfer edges
L_TGATE_SEA     = 1 << 4  # T-gate -> sea connectors
L_GATEWAY       = 1 << 5  # reserved: canals/straits corridors (Suez/Panama/etc.)
L_NE_CORRIDOR   = 1 << 6  # reserved: Northeast / NSR corridors
L_NW_CORRIDOR   = 1 << 7  # reserved: Northwest corridors
L_INJECT        = 1 << 8  # injected query edges (Q:START/Q:END -> picks)

# Ban bits: edge carries "potential ban reasons". Policy chooses which bans are active.
B_HIGH_LAT      = 1 << 0  # max(|lat_u|,|lat_v|) > hard cap
B_ECA           = 1 << 1  # reserved: Emission Control Areas
B_ICE           = 1 << 2  # reserved: ice/seasonal restrictions
B_SHALLOW       = 1 << 3  # reserved: bathymetry/draft limits
B_TSS           = 1 << 4  # reserved: traffic separation schemes
B_USER_NO_GO    = 1 << 5  # reserved: user-defined no-go polygons


def infer_layer_mask_from_etype(etype: str) -> int:
    """Best-effort mapping from `etype` to layer bits.
    Unknown types fall back to `L_BASE_SEA`.
    """
    e = (etype or "").upper()
    if e in ("E_RING", "E-RING"):
        return L_RING_E
    if e in ("T_RING", "T-RING"):
        return L_RING_T
    if e in ("E_T", "ET", "E_T_RAMP", "E_T_TRANSFER"):
        return L_ET_TRANSFER
    if e in ("T_S_GATE", "TGATE_SEA", "T_GATE_SEA", "T_S"):
        return L_TGATE_SEA
    if e in ("INJECT",):
        return L_INJECT
    # default: treat as base sea
    return L_BASE_SEA


def edge_lat_max_abs(G: nx.Graph, u: str, v: str) -> Optional[float]:
    """Return max(|lat_u|, |lat_v|) if node attrs exist."""
    try:
        lat_u = float(G.nodes[u].get("lat"))
        lat_v = float(G.nodes[v].get("lat"))
        return float(max(abs(lat_u), abs(lat_v)))
    except Exception:
        return None


def compute_edge_masks(G: nx.Graph, u: str, v: str, *, etype: str, hard_lat_cap_deg: float = 70.0) -> Tuple[int, int, Optional[float]]:
    """Compute (layer_mask, ban_mask, lat_max_abs) for an edge."""
    layer = infer_layer_mask_from_etype(etype)
    lat_max = edge_lat_max_abs(G, u, v)
    ban = 0
    if lat_max is not None and float(lat_max) > float(hard_lat_cap_deg):
        ban |= B_HIGH_LAT
    return int(layer), int(ban), (float(lat_max) if lat_max is not None else None)




# -------------------------
# Helpers
# -------------------------

def _wrap_dlon_deg(dlon: float) -> float:
    # shortest signed delta in degrees
    return (float(dlon) + 180.0) % 360.0 - 180.0


def haversine_km(a: LonLat, b: LonLat) -> float:
    """Haversine distance (km) with dateline-safe delta-lon."""
    lon1, lat1 = float(a[0]), float(a[1])
    lon2, lat2 = float(b[0]), float(b[1])

    r = 6371.0088  # mean earth radius (km)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(_wrap_dlon_deg(lon2 - lon1))

    s = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return float(2.0 * r * math.asin(min(1.0, math.sqrt(s))))


def _df(out: Dict[str, Any], key: str) -> Optional[pd.DataFrame]:
    x = out.get(key)
    return x if isinstance(x, pd.DataFrame) else None


def _ring_df(out: Dict[str, Any], key: str) -> Optional[pd.DataFrame]:
    rg = out.get("ring_graph", {}) or {}
    x = rg.get(key)
    return x if isinstance(x, pd.DataFrame) else None


# -------------------------
# Stats
# -------------------------

@dataclass
class GraphBuildStats:
    sea_edges_added: int = 0
    cc_edges_added: int = 0
    gateb_sea_edges_added: int = 0
    c_gateb_edges_added: int = 0

    e_edges_added: int = 0
    t_edges_added: int = 0
    et_edges_added: int = 0
    tgate_sea_edges_added: int = 0


# -------------------------
# Main
# -------------------------

def build_base_graph(
    out: Dict[str, Any],
    *,
    include_sea: bool = True,
    include_cc: bool = False,
    include_gateb_sea: bool = False,
    include_c_gateb: bool = False,
    include_rings: bool = True,
    include_et: bool = True,
    include_tgate_sea: bool = True,
    max_sea_edges: Optional[int] = None,
    max_cc_edges: Optional[int] = None,
    max_ring_edges: Optional[int] = None,
    weight_unit: str="km",
    bbox_ll: Optional[Tuple[float, float, float, float]] = None,
    hard_lat_cap_deg: float = 70.0,
) -> Tuple[nx.Graph, GraphBuildStats]:
    """Assemble a NetworkX undirected graph from `out` using node_id keys."""

    stats = GraphBuildStats()
    G = nx.Graph()

    # -------- Sea nodes + edges --------
    if include_sea:
        S_nodes = _df(out, "S_nodes")
        S_edges = out.get("S_edges")

        if isinstance(S_nodes, pd.DataFrame) and len(S_nodes):
            # add sea nodes with attrs
            for r in S_nodes.itertuples(index=False):
                nid = str(getattr(r, "node_id"))
                lon = float(getattr(r, "lon"))
                lat = float(getattr(r, "lat"))
                G.add_node(nid, lon=wrap_lon(lon), lat=lat, kind="sea")
                if hasattr(r, "x_m") and hasattr(r, "y_m"):
                    G.nodes[nid]["x_m"] = float(getattr(r, "x_m"))
                    G.nodes[nid]["y_m"] = float(getattr(r, "y_m"))

        if S_edges is not None and isinstance(S_edges, (list, tuple)) and len(S_edges):
            take = list(S_edges)
            if max_sea_edges is not None and len(take) > int(max_sea_edges):
                take = take[: int(max_sea_edges)]

            for e in take:
                if not (isinstance(e, (tuple, list)) and len(e) == 2):
                    continue
                a, b = e
                if not (isinstance(a, (tuple, list)) and len(a) == 2 and isinstance(b, (tuple, list)) and len(b) == 2):
                    continue
                a_ll = (float(a[0]), float(a[1]))
                b_ll = (float(b[0]), float(b[1]))
                u = coord_id(a_ll[0], a_ll[1], prefix="S:")
                v = coord_id(b_ll[0], b_ll[1], prefix="S:")
                w = haversine_km(a_ll, b_ll)
                layer, ban, lat_max = compute_edge_masks(G, u, v, etype="sea", hard_lat_cap_deg=hard_lat_cap_deg)
                G.add_edge(u, v, weight=w, length_km=w, etype="sea", layer_mask=layer, ban_mask=ban, lat_max_abs=lat_max)
                stats.sea_edges_added += 1

    # -------- Ring nodes + edges (E/T) --------
    if include_rings:
        E_nodes = _ring_df(out, "E_nodes")
        T_nodes = _ring_df(out, "T_nodes")
        E_edges = _ring_df(out, "E_edges")
        T_edges = _ring_df(out, "T_edges")

        def _add_ring_nodes(df: Optional[pd.DataFrame], prefix: str):
            if not isinstance(df, pd.DataFrame) or not len(df):
                return
            if "node_key" not in df.columns and "node_id" in df.columns:
                df = df.copy()
                df["node_key"] = df["node_id"].map(lambda i: f"{prefix}:{int(i)}")
            for r in df.itertuples(index=False):
                nid = str(getattr(r, "node_key"))
                lon = float(getattr(r, "lon"))
                lat = float(getattr(r, "lat"))
                G.add_node(nid, lon=wrap_lon(lon), lat=lat, kind=("E_ring" if prefix == "E" else "T_ring"))
                if hasattr(r, "x_m") and hasattr(r, "y_m"):
                    G.nodes[nid]["x_m"] = float(getattr(r, "x_m"))
                    G.nodes[nid]["y_m"] = float(getattr(r, "y_m"))

        _add_ring_nodes(E_nodes, "E")
        _add_ring_nodes(T_nodes, "T")

        def _add_ring_edges(df: Optional[pd.DataFrame], prefix: str, etype: str, counter_attr: str):
            nonlocal stats
            if not isinstance(df, pd.DataFrame) or not len(df):
                return
            take = df
            if max_ring_edges is not None and len(take) > int(max_ring_edges):
                take = take.head(int(max_ring_edges))
            # allow either u_key/v_key or u/v
            for r in take.itertuples(index=False):
                if hasattr(r, "u_key") and hasattr(r, "v_key") and getattr(r, "u_key") and getattr(r, "v_key"):
                    u = str(getattr(r, "u_key"))
                    v = str(getattr(r, "v_key"))
                else:
                    u = f"{prefix}:{int(getattr(r, 'u'))}"
                    v = f"{prefix}:{int(getattr(r, 'v'))}"
                # edge length
                w = None
                if hasattr(r, "length_km") and getattr(r, "length_km") is not None:
                    try:
                        w = float(getattr(r, "length_km"))
                    except Exception:
                        w = None
                if w is None:
                    # compute from node attrs if possible
                    if u in G.nodes and v in G.nodes:
                        a_ll = (float(G.nodes[u]["lon"]), float(G.nodes[u]["lat"]))
                        b_ll = (float(G.nodes[v]["lon"]), float(G.nodes[v]["lat"]))
                        w = haversine_km(a_ll, b_ll)
                    else:
                        continue
                layer, ban, lat_max = compute_edge_masks(G, u, v, etype=str(etype), hard_lat_cap_deg=hard_lat_cap_deg)
                G.add_edge(u, v, weight=w, length_km=w, etype=etype, layer_mask=layer, ban_mask=ban, lat_max_abs=lat_max)
                setattr(stats, counter_attr, getattr(stats, counter_attr) + 1)

        _add_ring_edges(E_edges, "E", "E_RING", "e_edges_added")
        _add_ring_edges(T_edges, "T", "T_RING", "t_edges_added")

    # -------- E<->T transfer edges --------
    if include_et:
        ET = _ring_df(out, "ET_edges")
        if isinstance(ET, pd.DataFrame) and len(ET):
            take = ET
            for r in take.itertuples(index=False):
                if hasattr(r, "u_key") and hasattr(r, "v_key") and getattr(r, "u_key") and getattr(r, "v_key"):
                    u = str(getattr(r, "u_key"))
                    v = str(getattr(r, "v_key"))
                else:
                    u = f"E:{int(getattr(r, 'u'))}"
                    v = f"T:{int(getattr(r, 'v'))}"
                # cost/length
                w = None
                for col in ["cost_km", "length_km", "dist_km"]:
                    if hasattr(r, col) and getattr(r, col) is not None:
                        try:
                            w = float(getattr(r, col))
                            break
                        except Exception:
                            pass
                if w is None and u in G.nodes and v in G.nodes:
                    a_ll = (float(G.nodes[u]["lon"]), float(G.nodes[u]["lat"]))
                    b_ll = (float(G.nodes[v]["lon"]), float(G.nodes[v]["lat"]))
                    w = haversine_km(a_ll, b_ll)
                if w is None:
                    continue
                etype = str(getattr(r, "etype")) if hasattr(r, "etype") else "E_T"
                layer, ban, lat_max = compute_edge_masks(G, u, v, etype=str(etype), hard_lat_cap_deg=hard_lat_cap_deg)
                G.add_edge(u, v, weight=w, length_km=w, etype=etype, layer_mask=layer, ban_mask=ban, lat_max_abs=lat_max)
                stats.et_edges_added += 1

    # -------- T-gate -> Sea connectors --------
    if include_tgate_sea:
        dfTG = _df(out, "tgate_sea_connectors")
        if isinstance(dfTG, pd.DataFrame) and len(dfTG):
            for r in dfTG.itertuples(index=False):
                # prefer explicit keys
                if hasattr(r, "t_node_key") and getattr(r, "t_node_key"):
                    u = str(getattr(r, "t_node_key"))
                else:
                    u = f"T:{int(getattr(r, 't_node_id'))}"
                if hasattr(r, "sea_node_id") and getattr(r, "sea_node_id"):
                    v = str(getattr(r, "sea_node_id"))
                else:
                    # fallback: use sea_idx to look up node_id
                    sid = int(getattr(r, "sea_idx"))
                    S_nodes = _df(out, "S_nodes")
                    if isinstance(S_nodes, pd.DataFrame) and "node_id" in S_nodes.columns and sid in S_nodes.index:
                        v = str(S_nodes.loc[sid, "node_id"])
                    else:
                        continue
                w = float(getattr(r, "dist_km")) if hasattr(r, "dist_km") else None
                if w is None:
                    # compute
                    if u in G.nodes and v in G.nodes:
                        a_ll = (float(G.nodes[u]["lon"]), float(G.nodes[u]["lat"]))
                        b_ll = (float(G.nodes[v]["lon"]), float(G.nodes[v]["lat"]))
                        w = haversine_km(a_ll, b_ll)
                    else:
                        continue
                etype = str(getattr(r, "etype")) if hasattr(r, "etype") else "T_S_GATE"
                layer, ban, lat_max = compute_edge_masks(G, u, v, etype=str(etype), hard_lat_cap_deg=hard_lat_cap_deg)
                G.add_edge(u, v, weight=w, length_km=w, etype=etype, layer_mask=layer, ban_mask=ban, lat_max_abs=lat_max)
                stats.tgate_sea_edges_added += 1

    return G, stats
