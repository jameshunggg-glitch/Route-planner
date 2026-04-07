from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.prepared import PreparedGeometry
from sklearn.neighbors import KDTree

from .visibility import segment_clear


def gate_to_sea_candidates(
    gate_xy: np.ndarray,
    *,
    S_nodes: pd.DataFrame,
    kdt: KDTree,
    top_n: int,
    r_max_km: float,
) -> list[int]:
    """Return candidate sea node indices near gate (within r_max_km)."""
    n = 0 if S_nodes is None else len(S_nodes)
    if n <= 0:
        return []

    k = min(int(top_n), n)          # ✅ 关键：k 不能超过训练点数
    if k <= 0:
        return []
    r = float(r_max_km) * 1000.0
    dist, idx = kdt.query([gate_xy], k=k, return_distance=True)
    dist = dist[0]
    idx = idx[0]
    keep = [int(i) for d, i in zip(dist, idx) if float(d) <= r]
    return keep


def build_gateB_connectors(
    Gate_all: pd.DataFrame,
    S_nodes: pd.DataFrame,
    *,
    sea_ok_set: set[int],
    kdt: KDTree,
    collision_prep: PreparedGeometry,
    top_n: int,
    r_max_km: float,
    k_connect: int,
) -> pd.DataFrame:
    """
    For each gate, pick up to k_connect sea nodes with collision-free segments.

    Output columns (stable):
      - gate_uid: int   (PRIMARY KEY for the gate row in Gate_all)
      - gate_row: int   (row index during iterrows; for debug)
      - g_id: int|None  (if exists; Gate_F may not have it)
      - source: str|None (if exists)
      - sea_idx: int
      - sea_node_id: str
      - dist_km: float
      - rank: int
    """
    Gate_all = Gate_all.copy()

    # Ensure gate_uid exists (primary key for back-referencing gates)
    if "gate_uid" not in Gate_all.columns:
        Gate_all["gate_uid"] = np.arange(len(Gate_all), dtype=int)

    rows: list[dict] = []

    for gate_row, g in Gate_all.iterrows():
        # required gate coords
        gx, gy = float(g["x_m"]), float(g["y_m"])
        gate_uid = int(g["gate_uid"])

        # optional identifiers
        g_id = None
        if "g_id" in Gate_all.columns and pd.notna(g.get("g_id", None)):
            try:
                g_id = int(g["g_id"])
            except Exception:
                g_id = None

        source = None
        if "source" in Gate_all.columns and pd.notna(g.get("source", None)):
            source = str(g["source"])

        # candidate sea nodes near gate
        cand = gate_to_sea_candidates(
            np.array([gx, gy], dtype=float),
            S_nodes=S_nodes, kdt=kdt, top_n=top_n, r_max_km=r_max_km
        )
        # filter by component/degree etc.
        cand = [i for i in cand if int(i) in sea_ok_set]

        scored: list[tuple[float, int]] = []
        for i in cand:
            s = S_nodes.iloc[int(i)]
            sx, sy = float(s["x_m"]), float(s["y_m"])

            # collision-free check
            if not segment_clear((gx, gy), (sx, sy), collision_prep=collision_prep):
                continue

            d_km = float(np.hypot(sx - gx, sy - gy) / 1000.0)
            scored.append((d_km, int(i)))

        # nearest first
        scored.sort(key=lambda t: t[0])

        for rank, (d_km, i) in enumerate(scored[: int(k_connect)]):
            s = S_nodes.iloc[int(i)]
            rows.append({
                "gate_uid": gate_uid,
                "gate_row": int(gate_row),
                "g_id": g_id,
                "source": source,

                "sea_idx": int(i),
                "sea_node_id": str(s["node_id"]),
                "dist_km": float(d_km),
                "rank": int(rank),
            })

    return pd.DataFrame(rows)
