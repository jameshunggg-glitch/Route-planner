# routing_map/scgraph_bridge.py
# -*- coding: utf-8 -*-
"""
Adapter for scgraph (marnet) that exposes stable APIs for routing_map.

Public APIs:
- sc_edges_in_bbox(bbox_ll, ...) -> bundle dict:
    {"nodes":[(lon,lat)], "edges":[((lon,lat),(lon,lat))], "stats":{...}}
- sc_keypoints_in_bbox(bbox_ll, edges=None, ...) -> [(lon,lat), ...]
- sc_shortest_path_lonlat(origin, dest, ...) -> {"track":[(lon,lat), ...], "length_km": float|None} or {}
"""

from __future__ import annotations

from typing import Tuple, List, Dict, Optional, Any
import math
import random

try:
    import numpy as np  # optional
except Exception:
    np = None

# --- Try to import scgraph marnet geograph ---
_MARNET = None
try:
    from scgraph.geographs.marnet import marnet_geograph as _MARNET  # type: ignore
except Exception:
    _MARNET = None


LonLat = Tuple[float, float]                  # (lon, lat)
BboxLL = Tuple[float, float, float, float]    # (min_lon, min_lat, max_lon, max_lat)
Seg = Tuple[LonLat, LonLat]


# ----------------- Small helpers -----------------
def _snap_key(pt: LonLat, decimals: int = 6) -> LonLat:
    return (round(float(pt[0]), decimals), round(float(pt[1]), decimals))


def _bearing_deg(p_from: LonLat, p_to: LonLat) -> float:
    dx = float(p_to[0]) - float(p_from[0])
    dy = float(p_to[1]) - float(p_from[1])
    ang = math.degrees(math.atan2(dy, dx))  # [-180,180]
    return (ang + 360.0) % 360.0


def _angle_deviation_from_straight(b1: float, b2: float) -> float:
    """
    Return deviation from "straight" (180°).
    - If perfectly straight, returns 0
    - The more it bends, the larger the value
    """
    diff = abs(b1 - b2)
    diff = 360.0 - diff if diff > 180.0 else diff  # [0,180]
    return 180.0 - diff  # straight=0, more bend => larger


def _norm_bbox(bbox_ll: BboxLL) -> BboxLL:
    x0, y0, x1, y1 = [float(v) for v in bbox_ll]
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


def _in_bbox(pt: LonLat, bbox_ll: BboxLL) -> bool:
    x0, y0, x1, y1 = bbox_ll
    x, y = float(pt[0]), float(pt[1])
    return (x0 <= x <= x1) and (y0 <= y <= y1)


# ----------------- GeoGraph adjacency-list extraction (preferred) -----------------
def _try_geograph_adjlist_bundle(
    bbox_ll: BboxLL,
    *,
    node_snap_decimals: int = 6,
) -> Optional[Dict[str, Any]]:
    if _MARNET is None:
        return None

    nodes = getattr(_MARNET, "nodes", None)
    graph = getattr(_MARNET, "graph", None)

    if not isinstance(nodes, (list, tuple)) or not isinstance(graph, (list, tuple)):
        return None
    if len(nodes) == 0 or len(graph) == 0:
        return None
    if len(nodes) != len(graph):
        return None

    bbox_ll = _norm_bbox(bbox_ll)

    # ---- decide node order by which interpretation yields more in-bbox points ----
    cnt_latlon = 0  # nodes = [lat, lon]
    cnt_lonlat = 0  # nodes = [lon, lat]
    for p in nodes:
        try:
            a, b = float(p[0]), float(p[1])
        except Exception:
            continue
        if _in_bbox((b, a), bbox_ll):  # (lon,lat)
            cnt_latlon += 1
        if _in_bbox((a, b), bbox_ll):
            cnt_lonlat += 1

    # choose mapping
    #use_lonlat = (cnt_lonlat > cnt_latlon)
    use_lonlat = False  # FORCE: nodes are [lat, lon]

    def to_lonlat(p):
        a, b = float(p[0]), float(p[1])
        return (b, a)  # (lon,lat)

    # ---- build idx2pt for in-bbox nodes ----
    idx2pt: Dict[int, LonLat] = {}
    for i, p in enumerate(nodes):
        try:
            lon, lat = to_lonlat(p)
        except Exception:
            continue
        pt = (lon, lat)
        if _in_bbox(pt, bbox_ll):
            idx2pt[i] = _snap_key(pt, node_snap_decimals)

    if not idx2pt:
        return {"nodes": [], "edges": [], "stats": {"edge_count": 0, "node_count": 0, "source": "geograph_adjlist_empty"}}

    # ---- edges among in-bbox nodes ----
    edges_set = set()
    for i, pti in idx2pt.items():
        nbrs = graph[i]
        if not isinstance(nbrs, dict):
            continue
        for j in nbrs.keys():
            if j in idx2pt:
                a = pti
                b = idx2pt[j]
                if a == b:
                    continue
                edges_set.add((a, b) if a <= b else (b, a))

    nodes_out = list(set(idx2pt.values()))
    edges_out = list(edges_set)

    return {
        "nodes": [(float(x), float(y)) for (x, y) in nodes_out],
        "edges": edges_out,
        "stats": {
            "edge_count": len(edges_out),
            "node_count": len(nodes_out),
            "source": "geograph_adjlist" + (":lonlat" if use_lonlat else ":latlon"),
            "cnt_latlon": cnt_latlon,
            "cnt_lonlat": cnt_lonlat,
        },
    }



# ----------------- networkx-like fallback (less common) -----------------
def _segments_from_edges_list(edges: Any, nodes_lookup: Optional[Dict[Any, LonLat]] = None) -> List[Seg]:
    segs: List[Seg] = []
    for e in edges or []:
        try:
            # tuple/list: (u, v, data) or (u, v)
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                u, v = e[0], e[1]
                if nodes_lookup and (u in nodes_lookup) and (v in nodes_lookup):
                    pu, pv = nodes_lookup[u], nodes_lookup[v]
                    segs.append(((float(pu[0]), float(pu[1])), (float(pv[0]), float(pv[1]))))
                    continue

            # dict / geojson-like
            if isinstance(e, dict):
                coords = None
                if "geometry" in e and hasattr(e["geometry"], "coords"):
                    coords = list(e["geometry"].coords)
                elif "coordinates" in e and isinstance(e["coordinates"], (list, tuple)):
                    coords = list(e["coordinates"])

                if coords and len(coords) >= 2:
                    for i in range(len(coords) - 1):
                        a, b = coords[i], coords[i + 1]
                        segs.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
        except Exception:
            continue
    return segs


def _try_graph_like_and_build_segments() -> Optional[List[Seg]]:
    if _MARNET is None:
        return None

    for attr in ("graph", "_graph", "geograph", "network", "G", "_G"):
        G = getattr(_MARNET, attr, None)
        if G is None:
            continue

        # nodes lookup
        nodes: Optional[Dict[Any, LonLat]] = None
        try:
            # networkx-like: G.nodes(data=True)
            tmp: Dict[Any, LonLat] = {}
            if hasattr(G, "nodes"):
                it = G.nodes(data=True)
                for nid, data in it:
                    if isinstance(data, dict):
                        lon = data.get("longitude") or data.get("lon") or data.get("x")
                        lat = data.get("latitude") or data.get("lat") or data.get("y")
                        if lon is not None and lat is not None:
                            tmp[nid] = (float(lon), float(lat))
            if tmp:
                nodes = tmp
        except Exception:
            nodes = None

        # edges
        try:
            if hasattr(G, "edges"):
                edges = G.edges()  # might return (u,v) pairs
                segs = _segments_from_edges_list(edges, nodes_lookup=nodes)
                if segs:
                    return segs
        except Exception:
            pass

    return None


# ----------------- Last resort: sampling shortest paths -----------------
def _fallback_segments_by_sampling(aoi: Optional[BboxLL], n_paths: int = 120) -> List[Seg]:
    segs: List[Seg] = []
    if _MARNET is None:
        return segs

    if aoi is None:
        aoi = (60.0, -20.0, 150.0, 40.0)

    x0, y0, x1, y1 = _norm_bbox(aoi)

    # sample more densely than 6x5 to avoid too-sparse subnet
    nx, ny = 10, 8
    xs = [x0 + i * (x1 - x0) / (nx - 1) for i in range(nx)]
    ys = [y0 + j * (y1 - y0) / (ny - 1) for j in range(ny)]
    pts = [(float(x), float(y)) for x in xs for y in ys]

    for _ in range(max(1, int(n_paths))):
        (lon1, lat1), (lon2, lat2) = random.sample(pts, 2)
        try:
            out = _MARNET.get_shortest_path(
                origin_node={"longitude": lon1, "latitude": lat1},
                destination_node={"longitude": lon2, "latitude": lat2},
                output_units="km",
            )
            path = out.get("coordinate_path", [])
            coords: List[LonLat] = []

            for p in path:
                if isinstance(p, dict):
                    lon = p.get("longitude") or p.get("lon") or p.get("x")
                    lat = p.get("latitude") or p.get("lat") or p.get("y")
                    if lon is not None and lat is not None:
                        coords.append((float(lon), float(lat)))
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    a, b = p[0], p[1]
                    # scgraph docs often use [lat, lon]
                    if -90 <= a <= 90 and -180 <= b <= 180:
                        coords.append((float(b), float(a)))  # (lon,lat)
                    elif -180 <= a <= 180 and -90 <= b <= 90:
                        coords.append((float(a), float(b)))

            if len(coords) >= 2:
                for i in range(len(coords) - 1):
                    segs.append((coords[i], coords[i + 1]))
        except Exception:
            continue

    return segs


# ----------------- Public API -----------------
def sc_edges_in_bbox(
    bbox_ll: BboxLL,
    edge_sample_ratio: float = 1.0,
    max_sample_routes: int = 120,
    node_snap_decimals: int = 6,
    simplify_epsilon_km: float = 0.0,
    force_sampling: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Get scgraph subnet within AOI bbox (lon/lat).

    Returns bundle:
      {
        "nodes": [(lon,lat), ...],
        "edges": [ ((lon1,lat1),(lon2,lat2)), ... ],
        "stats": {"edge_count": N, "node_count": M, "source": "..."}
      }
    """
    if _MARNET is None:
        return {"nodes": [], "edges": [], "stats": {"edge_count": 0, "node_count": 0, "source": "scgraph_missing"}}

    bbox_ll = _norm_bbox(bbox_ll)

    # 1) Preferred: GeoGraph adjlist
    if not force_sampling:
        bundle = _try_geograph_adjlist_bundle(bbox_ll, node_snap_decimals=node_snap_decimals)
        if bundle is not None:
            return bundle

    # 2) networkx-like (rare)
    source = "raw_edges"
    segs: Optional[List[Seg]] = None
    if not force_sampling:
        segs = _try_graph_like_and_build_segments()

    # 3) sampling
    if not segs:
        segs = _fallback_segments_by_sampling(bbox_ll, n_paths=max_sample_routes)
        source = "sampled_paths"

    edges_out: List[Seg] = []
    nodes_set = set()

    for (a, b) in segs or []:
        # keep if either endpoint in bbox (simple, stable)
        if not (_in_bbox(a, bbox_ll) or _in_bbox(b, bbox_ll)):
            continue

        if edge_sample_ratio < 1.0 and random.random() > float(edge_sample_ratio):
            continue

        ra = _snap_key(a, node_snap_decimals)
        rb = _snap_key(b, node_snap_decimals)
        if ra == rb:
            continue

        nodes_set.add(ra)
        nodes_set.add(rb)
        edges_out.append((ra, rb))

    _ = simplify_epsilon_km  # reserved

    return {
        "nodes": [(float(x), float(y)) for (x, y) in nodes_set],
        "edges": edges_out,
        "stats": {"edge_count": len(edges_out), "node_count": len(nodes_set), "source": source},
    }


def sc_keypoints_in_bbox(
    bbox_ll: BboxLL,
    edges: Optional[List[Seg]] = None,
    node_snap_decimals: int = 5,
    bend_threshold_deg: float = 12.0,
    **kwargs,
) -> List[LonLat]:
    if edges is None:
        got = sc_edges_in_bbox(bbox_ll, node_snap_decimals=node_snap_decimals, **kwargs)
        edges = got.get("edges", [])

    from collections import defaultdict

    neighbors = defaultdict(set)
    coord_accum = defaultdict(lambda: [0.0, 0.0, 0])

    for (a, b) in edges or []:
        if a == b:
            continue
        neighbors[a].add(b)
        neighbors[b].add(a)

        sx, sy, c = coord_accum[a]
        coord_accum[a] = [sx + a[0], sy + a[1], c + 1]
        sx, sy, c = coord_accum[b]
        coord_accum[b] = [sx + b[0], sy + b[1], c + 1]

    node_xy: Dict[LonLat, LonLat] = {}
    for k, (sx, sy, c) in coord_accum.items():
        node_xy[k] = (sx / c, sy / c) if c > 0 else (k[0], k[1])

    keypoints: List[LonLat] = []
    for k, ns in neighbors.items():
        xy = node_xy[k]
        deg = len(ns)
        if deg != 2:
            keypoints.append(xy)
        else:
            n1, n2 = list(ns)
            b1 = _bearing_deg(xy, node_xy[n1])
            b2 = _bearing_deg(xy, node_xy[n2])
            bend = _angle_deviation_from_straight(b1, b2)
            if bend >= float(bend_threshold_deg):
                keypoints.append(xy)

    seen = set()
    out: List[LonLat] = []
    for p in keypoints:
        sp = _snap_key(p, node_snap_decimals)
        if sp in seen:
            continue
        seen.add(sp)
        out.append(sp)
    return out


def sc_shortest_path_lonlat(origin: LonLat, dest: LonLat, **kwargs) -> Dict[str, Any]:
    if _MARNET is None:
        return {}

    try:
        out = _MARNET.get_shortest_path(
            origin_node={"longitude": float(origin[0]), "latitude": float(origin[1])},
            destination_node={"longitude": float(dest[0]), "latitude": float(dest[1])},
            **({"output_units": "km"} | kwargs),
        )

        coords: List[LonLat] = []
        for p in out.get("coordinate_path", []):
            if isinstance(p, dict):
                lon = p.get("longitude") or p.get("lon") or p.get("x")
                lat = p.get("latitude") or p.get("lat") or p.get("y")
                if lon is not None and lat is not None:
                    coords.append((float(lon), float(lat)))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                # scgraph docs often use [lat, lon]
                a, b = p[0], p[1]
                if -90 <= a <= 90 and -180 <= b <= 180:
                    coords.append((float(b), float(a)))  # (lon,lat)
                elif -180 <= a <= 180 and -90 <= b <= 90:
                    coords.append((float(a), float(b)))

        length_km = None
        for key in ("length_km", "length", "route_length", "distance"):
            if key in out and out[key] is not None:
                try:
                    length_km = float(out[key])
                    break
                except Exception:
                    pass

        return {"track": coords, "length_km": length_km}
    except Exception:
        return {}


__all__ = ["sc_edges_in_bbox", "sc_keypoints_in_bbox", "sc_shortest_path_lonlat"]
