from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math
from shapely.geometry import LineString
from shapely.prepared import prep
from .ring_types import RingBuildConfig, XY


def _segment_intersects_collision(
    a: XY,
    b: XY,
    collision_taut_prep: Any,
    collision_hard_prep: Any = None,
) -> bool:
    """
    Return True if segment a->b intersects taut collision OR hard collision.
    Inputs are PREPARED geometries (prep(...)) for speed.
    """
    seg = LineString([a, b])

    # hard guardrail (must never cross land)
    if collision_hard_prep is not None:
        try:
            if collision_hard_prep.intersects(seg):
                return True
        except Exception:
            # fall back: if prepared fails for any reason, be conservative
            return True

    # taut collision
    if collision_taut_prep is None:
        return False
    try:
        return bool(collision_taut_prep.intersects(seg))
    except Exception:
        # conservative fallback
        return True





def choose_cut_indices_closed_ring(pts_m_closed: List[XY], *, max_candidates: int = 16) -> List[int]:
    """
    Choose cut indices for a closed ring based on largest gaps (best_gap).
    Returns a list of candidate indices i meaning cut between i and i+1.
    """
    if len(pts_m_closed) < 4:
        return [0]
    # Ensure closure
    pts = pts_m_closed
    if pts[0] != pts[-1]:
        pts = pts + [pts[0]]

    gaps: List[Tuple[float, int]] = []
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        d = math.hypot(x2 - x1, y2 - y1)
        gaps.append((d, i))

    gaps.sort(reverse=True, key=lambda t: t[0])
    idxs = [i for _, i in gaps[:max_candidates]]
    return idxs or [0]


def _rotate_open_from_cut(pts_m_closed: List[XY], cut_i: int) -> List[XY]:
    """
    Convert closed pts into an open polyline by cutting between cut_i and cut_i+1.
    Returned list is NOT closed (first != last).
    """
    pts = pts_m_closed
    if pts[0] != pts[-1]:
        pts = pts + [pts[0]]
    # remove the duplicated last for easier handling
    pts = pts[:-1]
    n = len(pts)
    if n < 3:
        return pts

    j = (cut_i + 1) % n
    return pts[j:] + pts[:j]


def greedy_visibility_simplify_open(
    pts_open: List[XY],
    *,
    collision_taut_prep: Any,
    collision_hard_prep: Any,
    window_size: int,
) -> List[XY]:
    """
    Greedy simplifier for an open polyline:
    from i, jump to farthest j within window such that segment(i,j) is collision-free.
    Keeps endpoints.
    """
    n = len(pts_open)
    if n <= 2:
        return pts_open

    w = max(2, int(window_size))
    kept: List[XY] = [pts_open[0]]
    i = 0
    while i < n - 1:
        # cycle-aware guardrail:
        max_jump = max(2, n // 2)
        j_hi = min(n - 1, i + min(w, max_jump))
        chosen = i + 1  # always move at least 1 step
        for j in range(j_hi, i, -1):
            if not _segment_intersects_collision(pts_open[i], pts_open[j], collision_taut_prep, collision_hard_prep):
                chosen = j
                break
        kept.append(pts_open[chosen])
        i = chosen

    return kept


def taut_simplify_closed_ring(
    envelope_pts_m_closed: List[XY],
    *,
    collision_taut_m: Any,
    collision_hard_m: Any,
    cfg: RingBuildConfig,
) -> Tuple[List[XY], Dict[str, Any]]:
    """
    Cycle-aware taut simplification:
    - Try several cut positions (largest gaps first)
    - Run greedy visibility simplifier on the opened polyline
    - Close the result and validate closing segment

    Returns:
      taut_pts_closed, stats
    """
    if not envelope_pts_m_closed or len(envelope_pts_m_closed) < 4:
        return envelope_pts_m_closed, {"ok": False, "reason": "too_few_points"}

    # ensure closure
    pts_closed = envelope_pts_m_closed
    if pts_closed[0] != pts_closed[-1]:
        pts_closed = pts_closed + [pts_closed[0]]

    cut_candidates = choose_cut_indices_closed_ring(pts_closed, max_candidates=max(4, cfg.taut_max_tries * 2))
    tries = 0
    best = None
    best_stats = None

    # ------------------------------------------------------------
    # PREP collisions ONCE (critical for performance)
    # ------------------------------------------------------------
    eps = 1.0  # meters; boundary-touch tolerance
    collision_hard_prep = None
    if collision_hard_m is not None and (not getattr(collision_hard_m, "is_empty", True)):
        collision_hard_prep = prep(collision_hard_m)

    collision_taut_geom = collision_taut_m
    if collision_taut_geom is None or getattr(collision_taut_geom, "is_empty", True):
        collision_taut_prep = None
    else:
        # allow boundary touch by shrinking ONCE (not in inner-loop)
        try:
            g_shrink = collision_taut_geom.buffer(-eps)
            if g_shrink is not None and (not getattr(g_shrink, "is_empty", True)):
                collision_taut_geom = g_shrink
        except Exception:
            pass
        collision_taut_prep = prep(collision_taut_geom)


    for cut_i in cut_candidates:
        if tries >= int(cfg.taut_max_tries):
            break
        tries += 1

        open_pts = _rotate_open_from_cut(pts_closed, cut_i)
        simplified_open = greedy_visibility_simplify_open(
            open_pts, collision_taut_prep=collision_taut_prep, collision_hard_prep=collision_hard_prep, window_size=int(cfg.taut_window_size)
        )

        # close
        if simplified_open and (simplified_open[0] != simplified_open[-1]):
            simplified_closed = simplified_open + [simplified_open[0]]
        else:
            simplified_closed = simplified_open

        ok = True
        reason = "ok"

        # validate all segments (including closing)
        for k in range(len(simplified_closed) - 1):
            if _segment_intersects_collision(simplified_closed[k], simplified_closed[k + 1],  collision_taut_prep, collision_hard_prep,):
                ok = False
                reason = f"segment_collision@{k}"
                break

        stats = {
            "ok": ok,
            "reason": reason,
            "cut_i": int(cut_i),
            "n_pts_in": int(len(pts_closed)),
            "n_pts_out": int(len(simplified_closed)),
            "tries_used": int(tries),
        }

        if ok:
            return simplified_closed, stats

        # keep best attempt (max compression) as fallback
        if best is None or len(simplified_closed) < len(best):
            best = simplified_closed
            best_stats = stats

    # fallback: return envelope if nothing works
    if best is None:
        return pts_closed, {"ok": False, "reason": "no_candidates"}
    return best, {
        "ok": False,
        "reason": "fallback_to_envelope",
        "best_attempt": best_stats,
    }

