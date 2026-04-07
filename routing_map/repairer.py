# routing_map/repairer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import time

import math

try:
    from shapely.geometry import LineString, Point
    from shapely.prepared import prep as _prep
    from shapely.ops import nearest_points
except Exception as e:  # pragma: no cover
    raise ImportError("routing_map.repairer requires shapely") from e


LonLat = Tuple[float, float]
XY = Tuple[float, float]


# ---------------------------
# Config / Result
# ---------------------------

@dataclass
class RepairConfig:
    # Which edges are eligible to repair (by etype)
    etype_allow_prefixes: Tuple[str, ...] = ("sea", "scgraph", "gb_sea", "gateb_sea")

    # Fast patch: local midpoint detour probing
    fast_pad_km: Tuple[float, ...] = (20.0, 40.0, 80.0)
    fast_offset_m_step: float = 2000.0         # 2 km per step
    fast_offset_steps: int = 30                # max offset = step * steps
    fast_angle_degs: Tuple[float, ...] = (0.0, 20.0, 35.0, 50.0, 65.0, 80.0)

    # Rubberband: iterative push-away + smoothing
    rb_n_samples: int = 25                     # 10~40 recommended
    rb_max_iter: int = 60                      # 20~80 recommended
    rb_push_step_m: float = 250.0              # 100~500m recommended
    rb_smooth_lambda: float = 0.35             # 0.2~0.6 typical
    rb_converge_eps_m: float = 1.0             # stop if max move < eps

    # Safety
    allow_multi_hop: bool = True               # allow 2 midpoints if needed (fast patch)
    max_midpoints: int = 2

    # Debug
    debug: bool = True


@dataclass
class RepairStats:
    checked_edges: int = 0
    colliding_edges: int = 0
    repaired_edges: int = 0
    fast_success: int = 0
    rb_success: int = 0
    failed_edges: int = 0

    # timing (seconds)
    prepare_sec: float = 0.0
    total_sec: float = 0.0


@dataclass
class RepairOutcome:
    path_nodes: List[Any]          # node keys (same type as input path)
    path_ll: List[LonLat]          # lonlat polyline (including injected intermediate points)
    stats: RepairStats
    debug: List[Dict[str, Any]]


# ---------------------------
# Projector resolver
# ---------------------------

def _make_projectors(
    proj: Any,
    ll_to_m: Optional[Callable[[LonLat], XY]] = None,
    m_to_ll: Optional[Callable[[XY], LonLat]] = None,
) -> Tuple[Callable[[LonLat], XY], Callable[[XY], LonLat]]:
    if ll_to_m and m_to_ll:
        return ll_to_m, m_to_ll

    candidates = [
        ("ll_to_m", "m_to_ll"),
        ("to_m", "to_ll"),
        ("fwd", "inv"),
        ("forward", "inverse"),
    ]

    def _apply_xy(fn: Any, x: float, y: float, xy_tuple: Tuple[float, float]):
        if hasattr(fn, "transform") and callable(getattr(fn, "transform")):
            return fn.transform(x, y)

        if callable(fn):
            try:
                return fn(x, y)
            except TypeError:
                return fn(xy_tuple)

        raise TypeError(f"Projector of type {type(fn)} is not callable and has no .transform")

    for a, b in candidates:
        if hasattr(proj, a) and hasattr(proj, b):
            f = getattr(proj, a)
            g = getattr(proj, b)

            def _ll2m(p: LonLat) -> XY:
                lon, lat = p
                r = _apply_xy(f, float(lon), float(lat), (float(lon), float(lat)))
                return (float(r[0]), float(r[1]))

            def _m2ll(q: XY) -> LonLat:
                x, y = q
                r = _apply_xy(g, float(x), float(y), (float(x), float(y)))
                return (float(r[0]), float(r[1]))

            return _ll2m, _m2ll

    if hasattr(proj, "transform"):
        tr = getattr(proj, "transform")

        def _ll2m(p: LonLat) -> XY:
            lon, lat = p
            x, y = tr(float(lon), float(lat))
            return (float(x), float(y))

        def _m2ll(q: XY) -> LonLat:
            x, y = q
            try:
                from pyproj.enums import TransformDirection  # type: ignore
                lon, lat = tr(float(x), float(y), direction=TransformDirection.INVERSE)
                return (float(lon), float(lat))
            except Exception:
                raise ValueError(
                    "proj.transform found but inverse is unavailable; please pass m_to_ll explicitly."
                )

        return _ll2m, _m2ll

    raise ValueError("Cannot infer projection functions. Pass ll_to_m and m_to_ll, or provide a proj with ll_to_m/m_to_ll.")



# ---------------------------
# Geometry helpers
# ---------------------------

def _norm(vx: float, vy: float) -> float:
    return math.hypot(vx, vy)

def _unit(vx: float, vy: float) -> Tuple[float, float]:
    n = _norm(vx, vy)
    if n <= 1e-12:
        return 0.0, 0.0
    return vx / n, vy / n

def _line_intersects(prepared_collision, line: LineString) -> bool:
    # prepared geometry supports intersects/contains fast
    return bool(prepared_collision.intersects(line))

def _polyline_intersects(prepared_collision, pts_xy: Sequence[XY]) -> bool:
    if len(pts_xy) < 2:
        return False
    return _line_intersects(prepared_collision, LineString(pts_xy))

def _point_inside(collision_geom, p_xy: XY) -> bool:
    return bool(collision_geom.contains(Point(p_xy)))


# ---------------------------
# Fast patch
# ---------------------------

def _fast_patch_one_midpoint(
    u_xy: XY,
    v_xy: XY,
    collision_geom,
    prepared_collision,
    cfg: RepairConfig,
) -> Optional[List[XY]]:
    """
    Try find a single midpoint p such that segments u->p and p->v do not intersect collision.
    Returns [u, p, v] in XY if success else None.
    """
    ux, uy = u_xy
    vx, vy = v_xy
    mx, my = (ux + vx) * 0.5, (uy + vy) * 0.5

    dx, dy = vx - ux, vy - uy
    tx, ty = _unit(dx, dy)
    # perpendicular
    px, py = -ty, tx

    # if degenerate
    if abs(tx) + abs(ty) < 1e-12:
        return None

    # probe offsets on both sides, with angle variations
    for ang_deg in cfg.fast_angle_degs:
        ang = math.radians(ang_deg)
        ca, sa = math.cos(ang), math.sin(ang)

        # rotate (px,py) around (0,0) by +/-ang in the frame of (perp, tangential)
        # We create two basis directions: d1 and d2
        # d = perp*cos + tangential*sin
        d1x, d1y = px * ca + tx * sa, py * ca + ty * sa
        d2x, d2y = px * ca - tx * sa, py * ca - ty * sa

        for side in (+1.0, -1.0):
            for k in range(1, cfg.fast_offset_steps + 1):
                off = cfg.fast_offset_m_step * k
                # pick one of the rotated directions
                for (bx, by) in ((d1x, d1y), (d2x, d2y)):
                    pxk, pyk = mx + side * off * bx, my + side * off * by
                    p = (pxk, pyk)
                    # quick reject if point is inside collision
                    if _point_inside(collision_geom, p):
                        continue
                    # check two segments
                    if _line_intersects(prepared_collision, LineString([u_xy, p])):
                        continue
                    if _line_intersects(prepared_collision, LineString([p, v_xy])):
                        continue
                    return [u_xy, p, v_xy]
    return None


def _fast_patch_multi_midpoints(
    u_xy: XY,
    v_xy: XY,
    collision_geom,
    prepared_collision,
    cfg: RepairConfig,
) -> Optional[List[XY]]:
    """
    Allow up to 2 midpoints: u->p1->p2->v (greedy).
    This is still "fast" but gives extra success chance near complex coastlines.
    """
    if not cfg.allow_multi_hop or cfg.max_midpoints < 2:
        return None

    # First midpoint
    one = _fast_patch_one_midpoint(u_xy, v_xy, collision_geom, prepared_collision, cfg)
    if one is not None:
        return one

    # Try split at a coarse midpoint candidate, then patch each half with 1 midpoint.
    ux, uy = u_xy
    vx, vy = v_xy
    mx, my = (ux + vx) * 0.5, (uy + vy) * 0.5
    # Generate some coarse candidates around midpoint (bigger jumps)
    dx, dy = vx - ux, vy - uy
    tx, ty = _unit(dx, dy)
    px, py = -ty, tx

    coarse_steps = max(6, cfg.fast_offset_steps // 4)
    coarse_step_m = cfg.fast_offset_m_step * 3.0

    for side in (+1.0, -1.0):
        for k in range(1, coarse_steps + 1):
            off = coarse_step_m * k
            p0 = (mx + side * off * px, my + side * off * py)
            if _point_inside(collision_geom, p0):
                continue
            # require both halves to be potentially feasible (not strictly necessary)
            if _line_intersects(prepared_collision, LineString([u_xy, p0])):
                continue
            if _line_intersects(prepared_collision, LineString([p0, v_xy])):
                continue
            # Already clean with p0 as a midpoint
            return [u_xy, p0, v_xy]

    # If cannot find a clean single coarse midpoint, try two-step:
    # pick p0 even if u->p0 collides, then patch u->p0 and p0->v separately
    for side in (+1.0, -1.0):
        for k in range(1, coarse_steps + 1):
            off = coarse_step_m * k
            p0 = (mx + side * off * px, my + side * off * py)
            if _point_inside(collision_geom, p0):
                continue

            left = _fast_patch_one_midpoint(u_xy, p0, collision_geom, prepared_collision, cfg)
            if left is None:
                continue
            right = _fast_patch_one_midpoint(p0, v_xy, collision_geom, prepared_collision, cfg)
            if right is None:
                continue
            # merge: left = [u, pL, p0], right = [p0, pR, v]
            return [left[0], left[1], left[2], right[1], right[2]]

    return None


# ---------------------------
# Rubberband patch
# ---------------------------

def _rubberband_patch(
    u_xy: XY,
    v_xy: XY,
    collision_geom,
    prepared_collision,
    cfg: RepairConfig,
) -> Optional[List[XY]]:
    """
    Resample u->v into N points, iteratively push points out of collision and smooth.
    Success when entire polyline does NOT intersect collision.
    Returns list of XY points if success else None.
    """
    n = int(cfg.rb_n_samples)
    n = max(10, min(60, n))

    ux, uy = u_xy
    vx, vy = v_xy

    pts = []
    for i in range(n):
        t = i / (n - 1)
        pts.append((ux * (1 - t) + vx * t, uy * (1 - t) + vy * t))

    # if already ok, return as-is
    if not _polyline_intersects(prepared_collision, pts):
        return pts

    boundary = collision_geom.boundary

    for it in range(int(cfg.rb_max_iter)):
        max_move = 0.0
        new_pts = [pts[0]]

        for i in range(1, n - 1):
            x, y = pts[i]
            p = Point(x, y)

            moved_x, moved_y = x, y

            # push if inside collision OR local segments collide
            inside = collision_geom.contains(p)
            local_bad = (
                _line_intersects(prepared_collision, LineString([pts[i - 1], (x, y)]))
                or _line_intersects(prepared_collision, LineString([(x, y), pts[i + 1]]))
            )

            if inside or local_bad:
                # nearest point on boundary
                try:
                    # nearest_points returns (p_on_geom1, p_on_geom2)
                    q1, q2 = nearest_points(p, boundary)
                    bx, by = float(q2.x), float(q2.y)
                except Exception:
                    # fallback: skip pushing
                    bx, by = x, y

                dx, dy = x - bx, y - by
                uxv, uyv = _unit(dx, dy)
                if abs(uxv) + abs(uyv) < 1e-12:
                    # degenerate: use perpendicular of global direction
                    gx, gy = _unit(vx - ux, vy - uy)
                    uxv, uyv = -gy, gx

                step = float(cfg.rb_push_step_m)
                moved_x = x + uxv * step
                moved_y = y + uyv * step

            new_pts.append((moved_x, moved_y))
            max_move = max(max_move, math.hypot(moved_x - x, moved_y - y))

        new_pts.append(pts[-1])

        # smoothing (Laplacian)
        lam = float(cfg.rb_smooth_lambda)
        if lam > 0:
            sm = [new_pts[0]]
            for i in range(1, n - 1):
                x, y = new_pts[i]
                x0, y0 = new_pts[i - 1]
                x1, y1 = new_pts[i + 1]
                sx = x + lam * ((x0 + x1) * 0.5 - x)
                sy = y + lam * ((y0 + y1) * 0.5 - y)
                sm.append((sx, sy))
            sm.append(new_pts[-1])
            new_pts = sm

        pts = new_pts

        if not _polyline_intersects(prepared_collision, pts):
            return pts

        if max_move < float(cfg.rb_converge_eps_m):
            # converged but still intersecting => give up
            break

    return None


# ---------------------------
# Main Repairer
# ---------------------------

class PathRepairer:
    """
    Repair ONLY colliding sea/scgraph edges along an A* path, by replacing bad edges with
    a small polyline detour that avoids collision geometry (in metric CRS, e.g., EPSG:3857).

    Typical use:
      repairer = PathRepairer(cfg)
      out2 = repairer.repair_path(G, path, collision_m=out["collision_m"], proj=out["proj"])
    """


    def __init__(self, cfg: Optional[RepairConfig] = None):
        self.cfg = cfg or RepairConfig()
        # cache prepared collision geometry per-process
        self._prepared_cache: Dict[int, Any] = {}


    def repair_path(
        self,
        G,
        path_nodes,
        *,
        collision_m,
        proj=None,
        ll_to_m=None,
        m_to_ll=None,
        edge_ll=None,
        edge_should_repair=None,
    ) -> RepairOutcome:
        """
        Inputs:
          - G: networkx-like graph
          - path_nodes: A* node list (node keys are usually (lon,lat) tuples in your notebook)
          - collision_m: shapely Polygon/MultiPolygon in meters (NOT prepared) OR prepared geom.
          - proj or ll_to_xy/xy_to_ll: used to convert lonlat -> meters for collision checks/repair.
          - edge_ll: optional function to get endpoints lonlat from an edge; default assumes nodes are lonlat.
          - edge_should_repair: predicate; default uses cfg.etype_allow_prefixes.

        Output:
          - path_nodes: same type as input nodes (keeps original nodes; injected midpoints become lonlat tuples)
          - path_ll: repaired polyline lonlat (for drawing / simplification)
        """
        cfg = self.cfg
        stats = RepairStats()
        dbg: List[Dict[str, Any]] = []

        t0_total = time.perf_counter()

        ll2m, m2ll = _make_projectors(proj, ll_to_m=ll_to_m, m_to_ll=m_to_ll)

        # collision can be prepared or raw; normalize + cache prepared geom
        t0_prep = time.perf_counter()
        collision_geom = getattr(collision_m, "context", None)
        if collision_geom is None:
            collision_geom = collision_m

        key = id(collision_geom)
        prepared_collision = self._prepared_cache.get(key)
        if prepared_collision is None:
            try:
                prepared_collision = _prep(collision_geom)
            except Exception:
                # maybe already prepared
                prepared_collision = collision_m
            self._prepared_cache[key] = prepared_collision

        stats.prepare_sec += time.perf_counter() - t0_prep

        def _default_edge_ll(u, v, data) -> Tuple[LonLat, LonLat]:
            # Prefer graph node attrs (node_id edition), fallback to parsing lon/lat from node_id.
            def _node_ll(n) -> LonLat:
                try:
                    if hasattr(G, "nodes") and n in G.nodes:
                        nd = G.nodes[n]
                        if isinstance(nd, dict) and "lon" in nd and "lat" in nd:
                            return (float(nd["lon"]), float(nd["lat"]))
                except Exception:
                    pass

                # parse "...lon,lat" from node id
                s = str(n)
                if ":" in s:
                    s2 = s.split(":", 1)[1]
                else:
                    s2 = s
                if "," in s2:
                    a, b = s2.split(",", 1)
                    try:
                        return (float(a), float(b))
                    except Exception:
                        pass

                # tuple fallback
                if isinstance(n, (tuple, list)) and len(n) == 2:
                    return (float(n[0]), float(n[1]))
                raise KeyError(f"cannot resolve lon/lat for node {n}")

            return _node_ll(u), _node_ll(v)

        def _default_should_repair(u, v, data) -> bool:
            et = str(data.get("etype", ""))
            return any(et.startswith(pfx) for pfx in cfg.etype_allow_prefixes)

        edge_ll = edge_ll or _default_edge_ll
        edge_should_repair = edge_should_repair or _default_should_repair

        # Build repaired lonlat polyline by walking edges
        repaired_nodes: List[Any] = list(path_nodes)
        repaired_ll: List[LonLat] = []

        if not path_nodes:
            stats.total_sec = time.perf_counter() - t0_total
            return RepairOutcome(path_nodes=[], path_ll=[], stats=stats, debug=dbg)

        # seed first point (node_id-safe)
        ll0, _ll0 = edge_ll(path_nodes[0], path_nodes[0], {})
        repaired_ll.append(ll0)
        for u, v in zip(path_nodes, path_nodes[1:]):
            stats.checked_edges += 1
            data = G[u][v] if hasattr(G, "__getitem__") else {}
            # networkx adjacency: G[u][v] is dict of attrs
            if isinstance(data, dict) and "etype" not in data and isinstance(G, object):
                # some nx setups: G[u][v] already attr dict; keep
                pass

            # if not eligible => just append v
            if not edge_should_repair(u, v, data):
                _, ll_v = edge_ll(u, v, data)
                repaired_ll.append(ll_v)
                continue

            ll_u, ll_v = edge_ll(u, v, data)
            u_xy = ll2m((ll_u[0], ll_u[1]))
            v_xy = ll2m((ll_v[0], ll_v[1]))

            seg = LineString([u_xy, v_xy])
            if not _line_intersects(prepared_collision, seg):
                # ok
                repaired_ll.append(ll_v)
                continue

            stats.colliding_edges += 1
            rec: Dict[str, Any] = {
                "u": u, "v": v, "etype": data.get("etype"),
                "status": "colliding",
            }

            # ---- try fast patch ----
            patched_xy: Optional[List[XY]] = None
            patched_xy = _fast_patch_one_midpoint(u_xy, v_xy, collision_geom, prepared_collision, cfg)
            if patched_xy is None:
                patched_xy = _fast_patch_multi_midpoints(u_xy, v_xy, collision_geom, prepared_collision, cfg)

            if patched_xy is not None:
                stats.repaired_edges += 1
                stats.fast_success += 1
                rec["status"] = "fast_ok"
                rec["n_pts"] = len(patched_xy)

                # append intermediate points (excluding first, include last)
                for p_xy in patched_xy[1:]:
                    llp = m2ll((p_xy[0], p_xy[1]))
                    # make a lonlat tuple node key (consistent with your notebook)
                    node = (float(llp[0]), float(llp[1]))
                    repaired_ll.append(node)
                dbg.append(rec)
                continue

                # ---- rubberband patch (DISABLED) ----
                # 原本這裡會做 _rubberband_patch()（昂貴，且在運河/窄水道常失敗拖慢）。
                # 依你的決策：先只保留 fast patch；fast 也失敗就直接走 fallback（保留原 edge）。
                # rb_xy = _rubberband_patch(u_xy, v_xy, collision_geom, prepared_collision, cfg)
                # if rb_xy is not None and len(rb_xy) >= 2:
                #     repaired_edges.append(rb_xy)
                #     stats.repaired += 1
                #     stats.rb_success += 1
                #     continue

            # ---- fail ----
            stats.failed_edges += 1
            rec["status"] = "fail"
            dbg.append(rec)

            # fall back to original v (even though colliding) so path stays continuous
            repaired_ll.append(ll_v)

        stats.total_sec = time.perf_counter() - t0_total
        return RepairOutcome(path_nodes=repaired_nodes, path_ll=repaired_ll, stats=stats, debug=dbg)
