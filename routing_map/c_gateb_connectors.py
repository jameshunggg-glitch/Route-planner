# c_gateb_connectors.py
from __future__ import annotations

import numpy as np
import pandas as pd


def in_bbox(lon: float, lat: float, bbox_ll) -> bool:
    """
    bbox_ll = (min_lon, min_lat, max_lon, max_lat)
    """
    if bbox_ll is None:
        return True
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)
    return (min_lon <= float(lon) <= max_lon) and (min_lat <= float(lat) <= max_lat)


def build_cnode_gateb_connectors_nearest(
    C_nodes: pd.DataFrame,
    gateB_df: pd.DataFrame,
    *,
    bbox_ll=None,
    max_deg_dist=None,
) -> pd.DataFrame:
    """
    建立 C node ↔ GateB 的橋接邊：每個 GateB 接到 AOI 內「最近的」C node。

    Parameters
    ----------
    C_nodes : DataFrame
        必須含欄位: c_id, lon, lat
    gateB_df : DataFrame
        必須含欄位: g_id, lon, lat
    bbox_ll : tuple(min_lon, min_lat, max_lon, max_lat) | None
        只使用 AOI 內的 C node / GateB 建 connector
    max_deg_dist : float | None
        若設定，距離（經緯度的歐氏距離，單位=度）> max_deg_dist 的 connector 會被丟棄。
        （用來避免把 GateB 接到非常遠的海岸點；debug 建議先 None）

    Returns
    -------
    DataFrame with columns:
        g_id, g_lon, g_lat, c_id, c_lon, c_lat, dist_deg
    """
    if C_nodes is None or gateB_df is None or len(C_nodes) == 0 or len(gateB_df) == 0:
        return pd.DataFrame(columns=["g_id", "g_lon", "g_lat", "c_id", "c_lon", "c_lat", "dist_deg"])

    need_c = {"c_id", "lon", "lat"}
    need_g = {"g_id", "lon", "lat"}
    if not need_c.issubset(set(C_nodes.columns)):
        raise ValueError(f"C_nodes must contain columns {need_c}, got {list(C_nodes.columns)}")
    if not need_g.issubset(set(gateB_df.columns)):
        raise ValueError(f"gateB_df must contain columns {need_g}, got {list(gateB_df.columns)}")

    # AOI filter
    C = C_nodes[["c_id", "lon", "lat"]].copy()
    C = C[C.apply(lambda r: in_bbox(r["lon"], r["lat"], bbox_ll), axis=1)]
    if len(C) == 0:
        return pd.DataFrame(columns=["g_id", "g_lon", "g_lat", "c_id", "c_lon", "c_lat", "dist_deg"])

    G = gateB_df[["g_id", "lon", "lat"]].copy()
    G = G[G.apply(lambda r: in_bbox(r["lon"], r["lat"], bbox_ll), axis=1)]
    if len(G) == 0:
        return pd.DataFrame(columns=["g_id", "g_lon", "g_lat", "c_id", "c_lon", "c_lat", "dist_deg"])

    c_arr = C[["lon", "lat"]].to_numpy(dtype=float)
    c_ids = C["c_id"].to_numpy()

    rows = []
    for _, gr in G.iterrows():
        g_id = int(gr["g_id"])
        g_lon = float(gr["lon"])
        g_lat = float(gr["lat"])

        d2 = (c_arr[:, 0] - g_lon) ** 2 + (c_arr[:, 1] - g_lat) ** 2
        j = int(np.argmin(d2))
        dist = float(np.sqrt(d2[j]))

        if max_deg_dist is not None and dist > float(max_deg_dist):
            continue

        rows.append(
            {
                "g_id": g_id,
                "g_lon": g_lon,
                "g_lat": g_lat,
                "c_id": int(c_ids[j]),
                "c_lon": float(c_arr[j, 0]),
                "c_lat": float(c_arr[j, 1]),
                "dist_deg": dist,
            }
        )

    return pd.DataFrame(rows)


def add_cnode_gateb_connectors_to_graph(
    G,
    connectors_df: pd.DataFrame,
    *,
    etype="c_gb",
    weight_col="dist_deg",
):
    """
    把 connectors_df 加入 networkx Graph。
    節點使用 (lon,lat) tuple，避免 numpy array unhashable 問題。
    """
    if connectors_df is None or len(connectors_df) == 0:
        return 0

    need = {"g_lon", "g_lat", "c_lon", "c_lat", weight_col}
    if not need.issubset(set(connectors_df.columns)):
        raise ValueError(f"connectors_df must contain {need}, got {list(connectors_df.columns)}")

    added = 0
    for _, r in connectors_df.iterrows():
        gb = (float(r["g_lon"]), float(r["g_lat"]))
        c = (float(r["c_lon"]), float(r["c_lat"]))
        w = float(r[weight_col])
        G.add_edge(c, gb, weight=w, etype=etype)
        added += 1
    return added
