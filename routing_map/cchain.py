from __future__ import annotations
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from .geom_utils import AOIProjector, geom_to_ll, linestring_sample_points

def build_C_chain_from_rings(
    rings_m: List[LineString],
    proj: AOIProjector,
    *,
    c_step_km: float,
    round_decimals: int,
):
    """Sample C nodes along each ring and build adjacent edges."""
    c_step_m = c_step_km * 1000.0
    C_nodes_rows = []
    C_edges_rows = []
    c_id = 0

    for ring_id, ring in enumerate(rings_m):
        if ring is None or ring.is_empty or ring.length <= 0:
            continue

        pts = linestring_sample_points(ring, c_step_m)
        ring_c_ids = []
        for p in pts:
            x, y = float(p.x), float(p.y)
            lon, lat = proj.to_ll.transform(x, y)
            lon_r = round(float(lon), round_decimals)
            lat_r = round(float(lat), round_decimals)
            s_km = float(ring.project(p)) / 1000.0
            C_nodes_rows.append({
                "c_id": c_id,
                "ring_id": ring_id,
                "lon": lon_r, "lat": lat_r,
                "x_m": x, "y_m": y,
                "s_km": s_km,
            })
            ring_c_ids.append(c_id)
            c_id += 1

        for i in range(len(ring_c_ids) - 1):
            u = ring_c_ids[i]
            v = ring_c_ids[i + 1]
            xu, yu = C_nodes_rows[u]["x_m"], C_nodes_rows[u]["y_m"]
            xv, yv = C_nodes_rows[v]["x_m"], C_nodes_rows[v]["y_m"]
            length_km = math.hypot(xv - xu, yv - yu) / 1000.0
            C_edges_rows.append({
                "ring_id": ring_id,
                "u": u, "v": v,
                "length_km": float(length_km),
            })

    C_nodes = pd.DataFrame(C_nodes_rows)
    C_edges = pd.DataFrame(C_edges_rows)
    return C_nodes, C_edges

def compute_ring_gap_stats(C_nodes: pd.DataFrame) -> pd.DataFrame:
    """Compute max consecutive distance (km) between sampled C nodes per ring."""
    if C_nodes.empty:
        return pd.DataFrame(columns=["ring_id","n","gap_km_max","gap_km_p95","gap_km_mean"])

    rows = []
    for ring_id, g in C_nodes.groupby("ring_id"):
        g = g.sort_values("s_km")
        xy = g[["x_m","y_m"]].to_numpy()
        if len(xy) < 2:
            rows.append({"ring_id": ring_id, "n": len(xy), "gap_km_max": 0.0, "gap_km_p95": 0.0, "gap_km_mean": 0.0})
            continue
        d = np.sqrt(((xy[1:] - xy[:-1])**2).sum(axis=1)) / 1000.0
        rows.append({
            "ring_id": ring_id,
            "n": int(len(xy)),
            "gap_km_max": float(np.max(d)),
            "gap_km_p95": float(np.percentile(d, 95)),
            "gap_km_mean": float(np.mean(d)),
        })
    return pd.DataFrame(rows).sort_values("gap_km_max", ascending=False).reset_index(drop=True)
