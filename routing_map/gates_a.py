from __future__ import annotations
import numpy as np
import pandas as pd

def build_gate_A_from_C_and_F_v1(
    C_nodes: pd.DataFrame,
    rings_df: pd.DataFrame,
    F_nodes: pd.DataFrame,
    *,
    min_ring_length_km: float,
    short_ring_no_gate_km: float,
    short_ring_one_gate_km: float,
    snap_to_f_km: float,
) -> pd.DataFrame:
    """Build Gate-A from C coverage; optionally snap to nearest F point.

    This is a straight refactor of the notebook's slim v1 strategy:
      - Decide number of gates per ring based on ring length
      - Place gates by arclength quantiles on C nodes
      - Snap gate location to nearest F if within snap_to_f_km
    """
    # Build quick access arrays for snapping
    Fx = F_nodes["x_m"].to_numpy(dtype=float) if len(F_nodes) else np.array([])
    Fy = F_nodes["y_m"].to_numpy(dtype=float) if len(F_nodes) else np.array([])
    Fr2 = (snap_to_f_km * 1000.0) ** 2

    def nearest_F(x, y):
        if len(Fx) == 0:
            return None
        dx = Fx - x
        dy = Fy - y
        d2 = dx*dx + dy*dy
        j = int(np.argmin(d2))
        if float(d2[j]) <= Fr2:
            return j
        return None

    rows = []
    g_id = 0
    # ring lengths
    ring_len = rings_df.set_index("ring_id")["length_km"].to_dict() if not rings_df.empty else {}
    for ring_id, g in C_nodes.groupby("ring_id"):
        L = float(ring_len.get(ring_id, g["s_km"].max() if len(g) else 0.0))
        if L < short_ring_no_gate_km:
            continue
        if L < short_ring_one_gate_km:
            n_gate = 1
        elif L < min_ring_length_km:
            n_gate = 2
        else:
            n_gate = max(2, int(round(L / min_ring_length_km)))

        gg = g.sort_values("s_km")
        s = gg["s_km"].to_numpy(dtype=float)
        for k in range(n_gate):
            q = (k + 0.5) / n_gate
            target = q * s[-1]
            j = int(np.argmin(np.abs(s - target)))
            row = gg.iloc[j].to_dict()
            x, y = float(row["x_m"]), float(row["y_m"])
            snap_j = nearest_F(x, y)
            if snap_j is not None:
                # snap to F
                fx, fy = float(F_nodes.iloc[snap_j]["x_m"]), float(F_nodes.iloc[snap_j]["y_m"])
                row["x_m"], row["y_m"] = fx, fy
                row["lon"], row["lat"] = float(F_nodes.iloc[snap_j]["lon"]), float(F_nodes.iloc[snap_j]["lat"])
                source = "A_COVERAGE_SNAP_F"
            else:
                source = "A_COVERAGE"

            rows.append({
                "g_id": int(g_id),
                "ring_id": int(ring_id),
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "x_m": float(row["x_m"]),
                "y_m": float(row["y_m"]),
                "source": source,
            })
            g_id += 1

    return pd.DataFrame(rows)
