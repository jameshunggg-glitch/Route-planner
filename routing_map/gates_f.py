from __future__ import annotations
import pandas as pd

def build_gate_F_primary(
    F_att: pd.DataFrame,
    rings_df: pd.DataFrame,
    *,
    min_spacing_km: float,
    max_per_ring: int,
    global_max: int,
    round_decimals: int,
) -> pd.DataFrame:
    """Pick primary gates from F points (greedy by score with spacing on s_km)."""
    if F_att.empty:
        return pd.DataFrame(columns=["ring_id","lon","lat","x_m","y_m","source","score","s_km"])

    rows = []
    g_id = 0
    for ring_id, g in F_att.groupby("ring_id"):
        gg = g.sort_values("score", ascending=False).copy()
        picked_s = []
        picked = 0
        for _, r in gg.iterrows():
            s = float(r.get("s_km", 0.0))
            if any(abs(s - ps) < min_spacing_km for ps in picked_s):
                continue
            picked_s.append(s)
            rows.append({
                "ring_id": int(ring_id),
                "lon": float(r["lon"]),
                "lat": float(r["lat"]),
                "x_m": float(r["x_m"]),
                "y_m": float(r["y_m"]),
                "source": "F_PRIMARY",
                "score": float(r.get("score", 0.0)),
                "s_km": float(r.get("s_km", 0.0)),
            })
            g_id += 1
            picked += 1
            if picked >= max_per_ring or g_id >= global_max:
                break
        if g_id >= global_max:
            break

    Gate_F = pd.DataFrame(rows)
    # rounding/dedup friendliness
    if not Gate_F.empty:
        Gate_F["lon"] = Gate_F["lon"].round(round_decimals)
        Gate_F["lat"] = Gate_F["lat"].round(round_decimals)
    return Gate_F
