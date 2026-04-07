# routing_map/snap_link_repair.py
from __future__ import annotations

from typing import Callable, List, Tuple, Any

from shapely.geometry import Point, LineString

LonLat = Tuple[float, float]


def _as_lonlat(p) -> LonLat:
    return (float(p[0]), float(p[1]))


def _run_repair_on_two_point_segment(
    a_ll: LonLat,
    b_ll: LonLat,
    *,
    collision_m,
    ll_to_m: Callable,
    m_to_ll: Callable,
    repairer_obj,
) -> List[LonLat]:
    """
    Repair a 2-point lonlat segment using whatever API is available on repairer_obj.

    Preference:
      1) repairer_obj.repair_polyline_ll([...])  (newer repairer)
      2) repairer_obj.repair_path(G, [a_ll, b_ll], ...) with a tiny dummy nx graph (older repairer)
    """
    # 1) If repairer supports repair_polyline_ll, use it directly.
    if hasattr(repairer_obj, "repair_polyline_ll") and callable(getattr(repairer_obj, "repair_polyline_ll")):
        rep = repairer_obj.repair_polyline_ll(
            [a_ll, b_ll],
            collision_m=collision_m,
            ll_to_m=ll_to_m,
            m_to_ll=m_to_ll,
        )
        path_ll = getattr(rep, "path_ll", None)
        if path_ll is None and isinstance(rep, dict):
            path_ll = rep.get("path_ll")
        if path_ll is None:
            # unknown return shape -> fallback to straight
            return [a_ll, b_ll]
        return [_as_lonlat(p) for p in path_ll]

    # 2) Fallback: use repair_path with a minimal graph containing the single edge.
    if hasattr(repairer_obj, "repair_path") and callable(getattr(repairer_obj, "repair_path")):
        try:
            import networkx as nx  # lazy import
        except Exception:
            # can't import networkx -> fallback
            return [a_ll, b_ll]

        G = nx.Graph()
        G.add_node(a_ll)
        G.add_node(b_ll)
        # Mark as 'sea' to match default allow prefixes; also force via edge_should_repair.
        G.add_edge(a_ll, b_ll, etype="sea", weight=0.0)

        try:
            rep = repairer_obj.repair_path(
                G,
                [a_ll, b_ll],
                collision_m=collision_m,
                ll_to_m=ll_to_m,
                m_to_ll=m_to_ll,
                edge_should_repair=lambda u, v, data: True,  # force repair eligibility
            )
        except TypeError:
            # Some implementations may not accept edge_should_repair; try without it.
            try:
                rep = repairer_obj.repair_path(
                    G,
                    [a_ll, b_ll],
                    collision_m=collision_m,
                    ll_to_m=ll_to_m,
                    m_to_ll=m_to_ll,
                )
            except Exception:
                return [a_ll, b_ll]
        except Exception:
            return [a_ll, b_ll]

        path_ll = getattr(rep, "path_ll", None)
        if path_ll is None and isinstance(rep, dict):
            path_ll = rep.get("path_ll")
        if path_ll is None:
            return [a_ll, b_ll]
        return [_as_lonlat(p) for p in path_ll]

    # No usable API -> fallback
    return [a_ll, b_ll]


def repair_snap_link_ll_if_needed(
    a_ll: LonLat,
    b_ll: LonLat,
    *,
    collision_m,
    ll_to_m: Callable,
    m_to_ll: Callable,
    repairer_obj,
    endpoint_eps_m: float = 1.0,   # 判斷端點是否在 collision 內/貼邊的容忍
) -> List[LonLat]:
    """
    Preserve original fallback semantics:
    - If either endpoint is inside (or extremely close to) collision => DO NOT repair; return [a_ll, b_ll].
    - Else if segment does not intersect collision => return [a_ll, b_ll].
    - Else (both endpoints outside AND segment intersects collision) => try repairer; if repair fails it will fallback.
    """
    a_ll = (float(a_ll[0]), float(a_ll[1]))
    b_ll = (float(b_ll[0]), float(b_ll[1]))

    if collision_m is None:
        return [a_ll, b_ll]
    if a_ll == b_ll:
        return [a_ll]

    # metric endpoints
    try:
        a_xy = ll_to_m(a_ll)
        b_xy = ll_to_m(b_ll)
    except Exception:
        # projector failed -> conservative fallback
        return [a_ll, b_ll]

    pa = Point(float(a_xy[0]), float(a_xy[1]))
    pb = Point(float(b_xy[0]), float(b_xy[1]))

    # 1) If either endpoint is in/near collision => keep original straight line (your desired fallback)
    try:
        if collision_m.contains(pa) or collision_m.distance(pa) <= float(endpoint_eps_m):
            return [a_ll, b_ll]
        if collision_m.contains(pb) or collision_m.distance(pb) <= float(endpoint_eps_m):
            return [a_ll, b_ll]
    except Exception:
        # conservative: if collision check fails, do not attempt repair
        return [a_ll, b_ll]

    # 2) If the straight segment doesn't intersect collision => no need to repair
    seg = LineString([pa, pb])
    try:
        if not collision_m.intersects(seg):
            return [a_ll, b_ll]
    except Exception:
        return [a_ll, b_ll]

    # 3) Exact case you care about: both endpoints outside but segment crosses land
    repaired = _run_repair_on_two_point_segment(
        a_ll,
        b_ll,
        collision_m=collision_m,
        ll_to_m=ll_to_m,
        m_to_ll=m_to_ll,
        repairer_obj=repairer_obj,
    )
    # final sanity
    if not repaired or len(repaired) < 2:
        return [a_ll, b_ll]
    return [_as_lonlat(p) for p in repaired]
