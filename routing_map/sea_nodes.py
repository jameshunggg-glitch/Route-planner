from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from typing import Any, Dict, Tuple, List, Optional
from sklearn.neighbors import KDTree

from .geom_utils import AOIProjector, coord_id, ll_to_xy_m

def build_sea_nodes_from_bundle(
    proj: AOIProjector,
    bundle: Dict[str, Any],
) -> tuple[pd.DataFrame, list[tuple[tuple[float,float],tuple[float,float]]], nx.Graph, KDTree]:
    """Convert scgraph bundle {nodes, edges} into DataFrame + graph + KDTree.

    The notebook stores edges as [((lon,lat),(lon,lat)), ...].
    We normalize to:
      S_nodes: columns [node_id, lon, lat, x_m, y_m, degree, component]
      G: nx.Graph over node_id
      kdt: KDTree over (x_m,y_m)
    """
    S_nodes_raw = bundle.get("nodes", [])
    S_edges_raw = bundle.get("edges", [])
    pts: list[tuple[float,float]] = []

    # nodes may be list of (lon,lat) or dict
    if isinstance(S_nodes_raw, list):
        for p in S_nodes_raw:
            if isinstance(p, (tuple, list)) and len(p) == 2:
                pts.append((float(p[0]), float(p[1])))
    elif isinstance(S_nodes_raw, dict) and "lon" in S_nodes_raw and "lat" in S_nodes_raw:
        pts.extend(list(zip(map(float, S_nodes_raw["lon"]), map(float, S_nodes_raw["lat"]))))

    for e in S_edges_raw:
        if not (isinstance(e, (tuple, list)) and len(e) == 2):
            continue
        a, b = e
        if isinstance(a, (tuple, list)) and len(a) == 2:
            pts.append((float(a[0]), float(a[1])))
        if isinstance(b, (tuple, list)) and len(b) == 2:
            pts.append((float(b[0]), float(b[1])))

    pts_unique = sorted(set(pts))
    S_nodes = pd.DataFrame(pts_unique, columns=["lon","lat"])
    S_nodes["node_id"] = [coord_id(lo, la, prefix="S:") for lo, la in S_nodes[["lon","lat"]].to_numpy()]

    # metric coords
    xy = np.array([ll_to_xy_m(proj, lo, la) for lo, la in S_nodes[["lon","lat"]].to_numpy()], dtype=float)
    S_nodes["x_m"] = xy[:,0]
    S_nodes["y_m"] = xy[:,1]

    # graph
    G = nx.Graph()
    for nid in S_nodes["node_id"]:
        G.add_node(nid)
    # edges as coord pairs
    norm_edges = []
    for e in S_edges_raw:
        if not (isinstance(e, (tuple, list)) and len(e) == 2):
            continue
        a, b = e
        if not (isinstance(a,(tuple,list)) and len(a)==2 and isinstance(b,(tuple,list)) and len(b)==2):
            continue
        na = coord_id(a[0], a[1], prefix="S:")
        nb = coord_id(b[0], b[1], prefix="S:")
        G.add_edge(na, nb)
        norm_edges.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))

    deg = dict(G.degree())
    S_nodes["degree"] = S_nodes["node_id"].map(lambda k: int(deg.get(k, 0)))

    # components
    comp_map = {}
    for ci, comp in enumerate(nx.connected_components(G)):
        for nid in comp:
            comp_map[nid] = ci
    S_nodes["component"] = S_nodes["node_id"].map(lambda k: int(comp_map.get(k, -1)))

    kdt = KDTree(S_nodes[["x_m","y_m"]].to_numpy(dtype=float))
    return S_nodes, norm_edges, G, kdt

def filter_sea_nodes(
    S_nodes: pd.DataFrame,
    G: nx.Graph,
    *,
    deg_min: int,
    use_largest_component_only: bool,
) -> set[int]:
    """Return indices of S_nodes that are eligible for connection (structure filter)."""
    ok = set(S_nodes.index[S_nodes["degree"] >= deg_min].tolist())
    if use_largest_component_only:
        # pick largest component id among ok
        comp_counts = S_nodes.loc[list(ok), "component"].value_counts()
        if len(comp_counts):
            keep_comp = int(comp_counts.index[0])
            ok = set(S_nodes.index[(S_nodes["component"] == keep_comp) & (S_nodes["degree"] >= deg_min)].tolist())
    return ok
