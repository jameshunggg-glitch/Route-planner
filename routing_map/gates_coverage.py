# routing_map/gates_coverage.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _ensure_cols(df: pd.DataFrame, cols: List[str], name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{name} missing columns: {miss}")


def attach_gates_to_nearest_C(
    gates: pd.DataFrame,
    C_nodes: pd.DataFrame,
    rings_df: Optional[pd.DataFrame] = None,
    *,
    ring_id_col: str = "ring_id",
    gate_x_col: str = "x_m",
    gate_y_col: str = "y_m",
    c_x_col: str = "x_m",
    c_y_col: str = "y_m",
    c_s_col: str = "s_km",
    out_s_col: str = "s_km",
    out_ringlen_col: str = "ring_length_km",
    debug: bool = True,
) -> pd.DataFrame:
    """
    對所有 gates 補上沿岸序列資訊 s_km（靠近哪個 C node 就用它的 s_km）。
    這樣 Gate_A / Gate_F 都可以一起做 coverage sampling。

    要求：
      - gates 需有 ring_id + x_m/y_m
      - C_nodes 需有 ring_id + x_m/y_m + s_km
    """
    if gates is None or len(gates) == 0:
        return gates.copy()

    _ensure_cols(gates, [ring_id_col, gate_x_col, gate_y_col], "gates")
    _ensure_cols(C_nodes, [ring_id_col, c_x_col, c_y_col, c_s_col], "C_nodes")

    out = gates.copy()

    # ring length（km）優先用 rings_df.length_km
    ringlen_map: Dict[int, float] = {}
    if rings_df is not None and len(rings_df) > 0 and ("ring_id" in rings_df.columns) and ("length_km" in rings_df.columns):
        try:
            ringlen_map = dict(zip(rings_df["ring_id"].astype(int), rings_df["length_km"].astype(float)))
        except Exception:
            ringlen_map = {}

    # 建 ring->C 索引，逐 ring 做最近鄰（gate 數通常很少，所以用 numpy brute-force 夠穩）
    out_s = np.full(len(out), np.nan, dtype=float)
    out_ringlen = np.full(len(out), np.nan, dtype=float)

    # group C by ring
    C_groups = {rid: g for rid, g in C_nodes.groupby(ring_id_col)}

    # gates by ring
    for rid, g_gate in out.groupby(ring_id_col):
        try:
            rid_i = int(rid)
        except Exception:
            continue

        if rid_i not in C_groups:
            continue

        gC = C_groups[rid_i]
        C_xy = gC[[c_x_col, c_y_col]].to_numpy(dtype=float)
        C_s = gC[c_s_col].to_numpy(dtype=float)

        gate_idx = g_gate.index.to_numpy()
        G_xy = g_gate[[gate_x_col, gate_y_col]].to_numpy(dtype=float)

        # brute-force nearest: O(n_gate * n_C) ; n_gate(每 ring) 通常很小
        # dist2 = (G[:,None,:] - C[None,:,:])^2 sum
        diff = G_xy[:, None, :] - C_xy[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        nn = np.argmin(dist2, axis=1)
        out_s[gate_idx] = C_s[nn]

        # ring length km
        if ringlen_map:
            out_ringlen[gate_idx] = ringlen_map.get(rid_i, np.nan)

    out[out_s_col] = out_s
    out[out_ringlen_col] = out_ringlen

    if debug:
        n_tot = len(out)
        n_ok = int(np.isfinite(out[out_s_col]).sum())
        print(f"[coverage] attach gates->C: s_km assigned {n_ok}/{n_tot}")

    return out


def _pick_best_one(df: pd.DataFrame, prefer_source_order: List[str]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    src = df["source"].astype(str) if "source" in df.columns else pd.Series([""] * len(df), index=df.index)
    pri_map = {s: i for i, s in enumerate(prefer_source_order)}
    pri = src.map(lambda x: pri_map.get(x, 9999))

    # 若有 score 用 score；不然就用原排序
    if "score" in df.columns:
        score = pd.to_numeric(df["score"], errors="coerce").fillna(-1e18)
        best_idx = (pd.DataFrame({"pri": pri, "score": -score}, index=df.index)
                    .sort_values(["pri", "score"])
                    .index[0])
    else:
        best_idx = (pd.DataFrame({"pri": pri}, index=df.index)
                    .sort_values(["pri"])
                    .index[0])
    return df.loc[[best_idx]]


def coverage_sample_gates_on_rings(
    gates: pd.DataFrame,
    *,
    gate_spacing_km: float,
    min_per_ring: int,
    prefer_source_order: List[str],
    ring_id_col: str = "ring_id",
    s_col: str = "s_km",
    ringlen_col: str = "ring_length_km",
    debug: bool = True,
) -> pd.DataFrame:
    """
    對 gates 做 coverage sampling（閉合 ring 版本）：
    - 以 ring 分組
    - 依 s_km 排序
    - 找最大 gap 作起點，做 greedy spacing 抽樣
    - 每 ring 至少保留 min_per_ring 個（你指定 1）

    注意：這裡假設 gates 已經有 s_km（用 attach_gates_to_nearest_C 補的）。
    """
    if gates is None or len(gates) == 0:
        return gates.copy()

    _ensure_cols(gates, [ring_id_col], "gates")
    if s_col not in gates.columns:
        raise KeyError(f"gates missing {s_col}. Run attach_gates_to_nearest_C() first.")

    out_rows = []

    before = len(gates)
    spacing = float(gate_spacing_km)

    for rid, g in gates.groupby(ring_id_col):
        gg = g.copy()

        # 沒 s_km（或 NaN）就先全留（不冒險砍掉）
        s = pd.to_numeric(gg[s_col], errors="coerce")
        ok = np.isfinite(s.to_numpy())
        if ok.sum() == 0:
            out_rows.append(gg)
            continue
        gg = gg.loc[ok].copy()
        s = s.loc[ok].to_numpy(dtype=float)

        # ring length
        ring_len = None
        if ringlen_col in gg.columns:
            v = pd.to_numeric(gg[ringlen_col], errors="coerce").dropna()
            if len(v) > 0:
                ring_len = float(v.iloc[0])
        if (ring_len is None) or (not np.isfinite(ring_len)) or (ring_len <= 0):
            # fallback：用 s_km range 近似（較差但比沒有好）
            ring_len = float(s.max() - s.min())
            if ring_len <= 0:
                # 全部在同一點
                out_rows.append(_pick_best_one(gg, prefer_source_order))
                continue

        # sort by s
        order = np.argsort(s)
        gg = gg.iloc[order].copy()
        s = s[order]

        n = len(gg)
        if n == 1:
            out_rows.append(gg)
            continue

        # gaps + wrap gap
        gaps = np.diff(s)
        wrap_gap = ring_len - (s[-1] - s[0])
        gaps_full = np.concatenate([gaps, [wrap_gap]])
        start = int(np.argmax(gaps_full))  # gap index
        start_idx = (start + 1) % n        # first pick index

        # greedy on doubled axis
        s0 = float(s[start_idx])
        s_d = np.concatenate([s, s + ring_len])  # length 2n
        idx_d = np.concatenate([np.arange(n), np.arange(n)])

        picked = []
        last = s0
        picked.append(start_idx)

        # iterate forward from start_idx+1 to start_idx+n-1 (in doubled index space)
        # find indices in doubled ordering: map local indices by shifting
        # easiest: traverse k = start_idx+1 ... start_idx+n-1 in circular
        for step in range(1, n):
            j = (start_idx + step) % n
            # we need "current s" in unwrapped axis:
            # if j <= start_idx -> add ring_len
            cur = float(s[j] + (ring_len if j <= start_idx else 0.0))
            if cur - last >= spacing:
                picked.append(j)
                last = cur

        picked_df = gg.iloc[picked].copy()

        # enforce min_per_ring
        if len(picked_df) < int(min_per_ring):
            # 只補到 1（你目前 min_per_ring=1）
            picked_df = _pick_best_one(gg, prefer_source_order)

        out_rows.append(picked_df)

    out = pd.concat(out_rows, ignore_index=True) if out_rows else gates.iloc[0:0].copy()

    if debug:
        after = len(out)
        print(f"[coverage] spacing sample: before={before} after={after} removed={before-after} "
              f"(spacing_km={gate_spacing_km}, min_per_ring={min_per_ring})")

    return out
