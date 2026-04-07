"""
routing_map.e_t_transfer

Build E<->T transfer edges from Shared_nodes.

Implements "Approach B":
- Keep E_nodes and T_nodes as separate node id spaces.
- Add cheap transfer edges between each shared pair.

The transfer edge allows traversing:
E_RING -> (E_T_SHARED) -> T_RING

IMPORTANT (ID collisions):
Ring node_id often starts at 0, which collides with sea-node indices (also 0..).
When adding to a graph, ALWAYS use mapping functions that put each id-space into
a unique namespace (offset or prefix).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd
import networkx as nx


@dataclass
class ETTransferParams:
    etype: str = "E_T_SHARED"
    weight_km: float = 0.0          # 0 cost transfer (or small epsilon)
    bidirectional: bool = True
    keep_ring_id: bool = True


def build_et_shared_edges(
    out: dict,
    *,
    params: Optional[ETTransferParams] = None,
    Shared_nodes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of transfer edges with schema:
    - etype, ring_id, u, v, u_kind, v_kind, weight_km

    where u_kind/v_kind in {"E","T"}.
    """
    params = params or ETTransferParams()
    rg = out.get("ring_graph", {}) or {}

    if Shared_nodes is None:
        Shared_nodes = rg.get("Shared_nodes", None)

    cols = ["etype", "ring_id", "u", "v", "u_kind", "v_kind", "weight_km"]
    if not isinstance(Shared_nodes, pd.DataFrame) or len(Shared_nodes) == 0:
        return pd.DataFrame(columns=cols)

    rows = []
    for r in Shared_nodes.itertuples(index=False):
        rid = int(getattr(r, "ring_id")) if params.keep_ring_id and hasattr(r, "ring_id") else -1
        e_id = int(getattr(r, "e_node_id"))
        t_id = int(getattr(r, "t_node_id"))

        rows.append({"etype": params.etype, "ring_id": rid, "u": e_id, "v": t_id, "u_kind": "E", "v_kind": "T", "weight_km": float(params.weight_km)})

        if params.bidirectional:
            rows.append({"etype": params.etype, "ring_id": rid, "u": t_id, "v": e_id, "u_kind": "T", "v_kind": "E", "weight_km": float(params.weight_km)})

    return pd.DataFrame(rows, columns=cols)


def add_et_shared_edges_to_graph(
    G: nx.Graph,
    df_et: pd.DataFrame,
    *,
    e_node_key_fn: Optional[Callable[[int], object]] = None,
    t_node_key_fn: Optional[Callable[[int], object]] = None,
    weight_col: str = "weight_km",
    etype_col: str = "etype",
) -> int:
    """
    Add transfer edges into a networkx graph.

    Parameters
    ----------
    e_node_key_fn / t_node_key_fn:
        Map raw E/T node_id into graph node keys (to avoid collisions).
        Example:
            e_node_key_fn = lambda eid: ("E", int(eid))
            t_node_key_fn = lambda tid: ("T", int(tid))
    """
    if e_node_key_fn is None:
        e_node_key_fn = lambda eid: int(eid)
    if t_node_key_fn is None:
        t_node_key_fn = lambda tid: int(tid)

    if not isinstance(df_et, pd.DataFrame) or len(df_et) == 0:
        return 0

    need_cols = {"u", "v", "u_kind", "v_kind"}
    if not need_cols.issubset(set(df_et.columns)):
        raise ValueError(f"df_et missing required columns: {sorted(need_cols - set(df_et.columns))}")

    added = 0
    for r in df_et.itertuples(index=False):
        u_id = int(getattr(r, "u"))
        v_id = int(getattr(r, "v"))
        u_kind = str(getattr(r, "u_kind"))
        v_kind = str(getattr(r, "v_kind"))

        w = float(getattr(r, weight_col))
        et = str(getattr(r, etype_col)) if hasattr(r, etype_col) else "E_T_SHARED"

        u = e_node_key_fn(u_id) if u_kind == "E" else t_node_key_fn(u_id)
        v = e_node_key_fn(v_id) if v_kind == "E" else t_node_key_fn(v_id)

        if u not in G:
            G.add_node(u)
        if v not in G:
            G.add_node(v)

        G.add_edge(u, v, weight=w, etype=et)
        added += 1

    return added
