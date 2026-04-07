from __future__ import annotations
import numpy as np
import pandas as pd

def merge_gates(Gate_A: pd.DataFrame, Gate_F: pd.DataFrame, *, round_decimals: int = 5) -> pd.DataFrame:
    cols = ["ring_id","lon","lat","x_m","y_m","source"]
    A = Gate_A.copy() if Gate_A is not None else pd.DataFrame(columns=cols)
    F = Gate_F.copy() if Gate_F is not None else pd.DataFrame(columns=cols)

    if "source" not in A.columns:
        A["source"] = "A_COVERAGE"
    if "source" not in F.columns:
        F["source"] = "F_PRIMARY"

    for df in (A, F):
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df["lon"] = df["lon"].astype(float)
        df["lat"] = df["lat"].astype(float)
        df["lon_r"] = df["lon"].round(round_decimals)
        df["lat_r"] = df["lat"].round(round_decimals)

    allg = pd.concat([F[cols+["lon_r","lat_r"]], A[cols+["lon_r","lat_r"]]], ignore_index=True)
    # prefer F_PRIMARY when duplicate coords
    pref = allg["source"].astype(str).apply(lambda s: 0 if s == "F_PRIMARY" else 1)
    allg = allg.assign(_pref=pref).sort_values(["ring_id","lon_r","lat_r","_pref"])
    allg = allg.drop_duplicates(subset=["ring_id","lon_r","lat_r"], keep="first").drop(columns=["lon_r","lat_r","_pref"]).reset_index(drop=True)
    allg["g_id"] = np.arange(len(allg), dtype=int)
    return allg
