from __future__ import annotations
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, LineString, GeometryCollection
from shapely.ops import transform

from .geom_utils import AOIProjector

def _iter_polygons_any(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        polys = []
        for g in geom.geoms:
            polys.extend(_iter_polygons_any(g))
        return polys
    return []

def _sample_line_xy(line: LineString, step_m: float) -> np.ndarray:
    if line.is_empty or line.length <= 0:
        return np.zeros((0,2), dtype=float)
    n = max(2, int(math.ceil(line.length / step_m)) + 1)
    pts = [line.interpolate(i*step_m) for i in range(n)]
    return np.array([[float(p.x), float(p.y)] for p in pts], dtype=float)

def _turn_angles_deg(xy: np.ndarray) -> np.ndarray:
    # angle at each interior point using segments (p[i]-p[i-1]) and (p[i+1]-p[i])
    n = len(xy)
    ang = np.zeros(n, dtype=float)
    if n < 3:
        return ang
    for i in range(1, n-1):
        a = xy[i] - xy[i-1]
        b = xy[i+1] - xy[i]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            ang[i] = 0.0
            continue
        cosv = float(np.clip(np.dot(a, b)/(na*nb), -1.0, 1.0))
        ang[i] = math.degrees(math.acos(cosv))
    return ang

def _nms_points_xy(xy: np.ndarray, score: np.ndarray, radius_km: float, max_keep: int):
    if len(xy) == 0:
        return []
    rad2 = (radius_km*1000.0)**2
    order = np.argsort(-score)
    keep = []
    suppressed = np.zeros(len(xy), dtype=bool)
    for j in order:
        if suppressed[j]:
            continue
        keep.append(int(j))
        if len(keep) >= max_keep:
            break
        d2 = ((xy - xy[j])**2).sum(axis=1)
        suppressed |= (d2 <= rad2)
        suppressed[j] = False
    return keep

def extract_F_nodes_from_union_smooth(
    union_smooth_m,
    proj: AOIProjector,
    *,
    sample_step_km: float,
    angle_deg_min: float,
    nms_radius_km: float,
    max_keep: int,
) -> pd.DataFrame:
    """Extract 'feature nodes' (F) from smoothed land union boundary.

    Heuristic:
      - sample exterior rings every sample_step_km
      - compute turning angles; keep those above angle_deg_min
      - apply NMS to enforce spatial spread

    Returns DataFrame with lon/lat and metric coords and a 'score' column.
    """
    step_m = sample_step_km * 1000.0
    polys = _iter_polygons_any(union_smooth_m)
    cand_xy = []
    cand_score = []
    cand_meta = []

    for poly in polys:
        ext = poly.exterior
        xy = _sample_line_xy(ext, step_m)
        if len(xy) == 0:
            continue
        ang = _turn_angles_deg(xy)
        # score = angle (bigger is sharper)
        mask = ang >= angle_deg_min
        idxs = np.where(mask)[0]
        for i in idxs:
            cand_xy.append(xy[i])
            cand_score.append(float(ang[i]))
    if not cand_xy:
        return pd.DataFrame(columns=["f_id","lon","lat","x_m","y_m","score"])

    cand_xy = np.asarray(cand_xy, dtype=float)
    cand_score = np.asarray(cand_score, dtype=float)
    keep = _nms_points_xy(cand_xy, cand_score, radius_km=nms_radius_km, max_keep=max_keep)

    rows = []
    for k, j in enumerate(keep):
        x, y = float(cand_xy[j,0]), float(cand_xy[j,1])
        lon, lat = proj.to_ll.transform(x, y)
        rows.append({
            "f_id": int(k),
            "lon": float(lon), "lat": float(lat),
            "x_m": x, "y_m": y,
            "score": float(cand_score[j]),
        })
    return pd.DataFrame(rows)
