from __future__ import annotations
import numpy as np
import pandas as pd

def attach_F_to_nearest_C(F_nodes: pd.DataFrame, C_nodes: pd.DataFrame) -> pd.DataFrame:
    """Attach each F to nearest C node (adds ring_id, s_km, nearest_c_id, dist_to_c_km).

    Notebook used brute force; we keep a vectorized-ish version.
    Complexity is fine for O(1e3) F nodes.
    """
    if len(F_nodes) == 0 or len(C_nodes) == 0:
        out = F_nodes.copy()
        out["ring_id"] = -1
        out["s_km"] = np.nan
        out["nearest_c_id"] = -1
        out["dist_to_c_km"] = np.nan
        return out

    Cxy = C_nodes[["x_m","y_m"]].to_numpy(dtype=float)
    Cring = C_nodes["ring_id"].to_numpy(dtype=int)
    Cs = C_nodes["s_km"].to_numpy(dtype=float)
    Cid = C_nodes["c_id"].to_numpy(dtype=int) if "c_id" in C_nodes.columns else C_nodes.index.to_numpy(dtype=int)

    rows = []
    for _, f in F_nodes.iterrows():
        fx, fy = float(f["x_m"]), float(f["y_m"])
        d2 = (Cxy[:,0]-fx)**2 + (Cxy[:,1]-fy)**2
        j = int(np.argmin(d2))
        dist_km = float(np.sqrt(d2[j]) / 1000.0)
        rows.append((int(Cring[j]), float(Cs[j]), int(Cid[j]), dist_km))

    out = F_nodes.copy()
    out["ring_id"] = [t[0] for t in rows]
    out["s_km"] = [t[1] for t in rows]
    out["nearest_c_id"] = [t[2] for t in rows]
    out["dist_to_c_km"] = [t[3] for t in rows]
    return out
