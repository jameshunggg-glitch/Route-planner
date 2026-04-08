# routing_map/build_aoi.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from shapely.prepared import prep

# --- 這裡改成了絕對匯入 ---
from routing_map.config import RoutingMapConfig
from routing_map.geom_utils import make_aoi_bbox, build_projector_from_bbox, expand_bbox_ll
from routing_map.io_land import load_polys_in_bbox
from routing_map.land_layers import build_land_layers
from routing_map.smooth import smooth_union_for_features_from_union
from routing_map.rings import build_coast_rings_smooth_v2, build_envelope_and_taut_rings_v1
from routing_map.ring_graph import build_ring_nodes_edges, RingGraphBuildParams
from routing_map.cchain import build_C_chain_from_rings
from routing_map.sea_nodes import build_sea_nodes_from_bundle, filter_sea_nodes
from routing_map import scgraph_bridge
from routing_map.e_t_transfer_v2 import build_e_t_transfer_edges, ETRampConfig
from routing_map.t_gate_connectors import build_tgate_sea_connectors, TGateSeaConnectorParams

def _pad_bbox_ll(bbox_ll, pad_deg: float):
    return expand_bbox_ll(bbox_ll, pad_deg)

def _norm_bbox_ll(bbox_ll):
    x0, y0, x1, y1 = map(float, bbox_ll)
    if y0 > y1: y0, y1 = y1, y0
    if x0 > x1:
        span_if_swapped = x0 - x1
        if span_if_swapped < 180.0: x0, x1 = x1, x0
    return (x0, y0, x1, y1)

def _split_bbox_dateline(bbox_ll):
    x0, y0, x1, y1 = map(float, bbox_ll)
    if x0 <= x1: return [(x0, y0, x1, y1)]
    return [(x0, y0, 180.0, y1), (-180.0, y0, x1, y1)]

def _merge_sc_bundles(bundles):
    nodes_set = set()
    edges_set = set()
    stats = {"source": [], "node_count_parts": [], "edge_count_parts": []}
    for b in bundles:
        if not isinstance(b, dict): continue
        nodes = b.get("nodes", []) or []
        edges = b.get("edges", []) or []
        st = b.get("stats", {}) or {}
        for p in nodes:
            if isinstance(p, (tuple, list)) and len(p) == 2:
                nodes_set.add((float(p[0]), float(p[1])))
        for e in edges:
            if not (isinstance(e, (tuple, list)) and len(e) == 2): continue
            a, c = e
            pa, pc = (float(a[0]), float(a[1])), (float(c[0]), float(c[1]))
            if pa == pc: continue
            edges_set.add((pa, pc) if pa <= pc else (pc, pa))
        stats["source"].append(st.get("source"))
    nodes_out = list(nodes_set)
    edges_out = list(edges_set)
    stats["node_count"] = len(nodes_out)
    stats["edge_count"] = len(edges_out)
    stats["source"] = "+".join([s for s in stats["source"] if s])
    return {"nodes": nodes_out, "edges": edges_out, "stats": stats}

def build_aoi(cfg: RoutingMapConfig) -> Dict[str, Any]:
    """
    清理後的 AOI Pipeline：僅保留 E/T Ring, C-Chain(用於Nudge), 及新的 T-Gate Connectors。
    """
    # --- AOI bbox ---
    bbox_ll = cfg.aoi.bbox_ll
    if bbox_ll is None:
        if cfg.aoi.origin_ll is None or cfg.aoi.dest_ll is None:
            raise ValueError("Provide either aoi.bbox_ll or (aoi.origin_ll & aoi.dest_ll)")
        bbox_ll = make_aoi_bbox(cfg.aoi.bbox_ll) # 這裡修正原始碼的小手誤
    
    bbox_ll = _norm_bbox_ll(bbox_ll)
    proj = build_projector_from_bbox(bbox_ll)

    # --- 1. Land & Layers ---
    polys_ll = load_polys_in_bbox(cfg.land.shp_path, bbox_ll)
    layers = build_land_layers(
        polys_ll, proj,
        buffer_km=cfg.land.buffer_km,
        avoid_km=cfg.land.avoid_km,
        collision_safety_km=cfg.land.collision_safety_km,
        grid_size_m=cfg.land.precision_grid_m,
    )
    union_m = layers["UNION_M"]

    # --- 2. Smooth union (用於輔助計算) ---
    union_smooth_m = smooth_union_for_features_from_union(
        union_m,
        a2_smooth_km=cfg.smooth.a2_smooth_km,
        a2_tol_km=cfg.smooth.a2_tol_km,
    )

    # --- 3. Rings 建構 (E-Ring & T-Ring) ---
    ring_cfg = getattr(cfg, "rings", None)
    if ring_cfg is not None:
        collision_hard_m = layers["COLLISION_M"]
        ring_base_m, env_lines_m, taut_lines_m, rings_df, rings = build_envelope_and_taut_rings_v1(
            union_smooth_m,
            collision_hard_m=collision_hard_m,
            cfg=ring_cfg,
        )
        rings_m = taut_lines_m
        ring_graph = build_ring_nodes_edges(
            rings,
            proj=proj,
            cfg=ring_cfg,
            params=RingGraphBuildParams(
                e_angle_feature_deg=25.0, 
                t_max_gap_km=20.0,         
                shared_tol_m=25.0,         
            ),
        )
    else:
        # Fallback legacy
        ring_base_m, rings_m, rings_df = build_coast_rings_smooth_v2(
            union_smooth_m,
            avoid_km=cfg.land.avoid_km,
            island_area_min_km2=5.0,
        )
        ring_graph = None

    # --- 4. ET Ramps (E世界與T世界的橋接) ---
    et_cfg = ETRampConfig(
        ramp_spacing_km=60.0,
        min_ramp_per_ring=2,
        near_shared_km=15.0,
        topK_T=12,
        k_ramp_per_anchor=1,
        ramp_max_km=40.0,
        ramp_penalty=0.10,
        enable_collision_check=True,
    )
    if ring_graph is not None:
        et_out = build_e_t_transfer_edges(
            ring_graph,
            collision_hard_m=layers["COLLISION_M"],
            cfg=et_cfg,
            shared_edge_cost=0.0,
        )
        ring_graph.update(et_out)

    # --- 5. C chain (保留！用於 snap.py 的 nudge 功能) ---
    C_nodes, C_edges = build_C_chain_from_rings(
        rings_m, proj,
        c_step_km=cfg.cchain.c_step_km,
        round_decimals=cfg.cchain.round_decimals,
    )

    # --- 6. Sea Subnet (深海航網) ---
    bbox_ll_sea = _pad_bbox_ll(bbox_ll, pad_deg=float(cfg.sea.aoi_pad_deg))
    sea_bboxes = _split_bbox_dateline(bbox_ll_sea)
    bundles = [scgraph_bridge.sc_edges_in_bbox(bbox_ll=b) for b in sea_bboxes]
    bundle = _merge_sc_bundles(bundles)
    S_nodes, S_edges, G, kdt = build_sea_nodes_from_bundle(proj, bundle)

    sea_ok_set = filter_sea_nodes(
        S_nodes, G,
        deg_min=int(cfg.sea.deg_min),
        use_largest_component_only=bool(cfg.sea.use_largest_component_only),
    )

    # --- 7. T-gate -> Sea Connectors (新的橋接機制) ---
    collision_prep = prep(layers["COLLISION_M"])
    tgate_sea_connectors = None
    try:
        _tmp_out = {
            "ring_graph": ring_graph,
            "S_nodes": S_nodes,
            "proj": proj,
            "layers": layers,
            "collision_prep": collision_prep,
        }
        tgate_sea_connectors = build_tgate_sea_connectors(
            _tmp_out,
            params=TGateSeaConnectorParams(
                k_connect=2, topN=60, r_connect_km=200.0,
                do_collision_check=True, do_repair=True,
            ),
        )
        if tgate_sea_connectors is not None and len(tgate_sea_connectors) > 0 and sea_ok_set is not None:
            tgate_sea_connectors = tgate_sea_connectors[tgate_sea_connectors["sea_idx"].astype(int).isin(sea_ok_set)].reset_index(drop=True)
    except Exception:
        tgate_sea_connectors = None

    # --- 8. ID Mappings (用於路徑搜尋) ---
    id2ll, id2xy = {}, {}
    if isinstance(S_nodes, pd.DataFrame) and len(S_nodes):
        for r in S_nodes.itertuples(index=False):
            nid = str(getattr(r, "node_id"))
            id2ll[nid] = (float(getattr(r, "lon")), float(getattr(r, "lat")))
            id2xy[nid] = (float(getattr(r, "x_m")), float(getattr(r, "y_m")))

    if isinstance(ring_graph, dict):
        for _k, _prefix in [("E_nodes", "E"), ("T_nodes", "T")]:
            df = ring_graph.get(_k)
            if isinstance(df, pd.DataFrame) and len(df):
                for r in df.itertuples(index=False):
                    nid = str(getattr(r, "node_key"))
                    id2ll[nid] = (float(getattr(r, "lon")), float(getattr(r, "lat")))
                    id2xy[nid] = (float(getattr(r, "x_m")), float(getattr(r, "y_m")))

    return {
        "cfg": cfg,
        "bbox_ll": bbox_ll,
        "id2ll": id2ll, "id2xy": id2xy,
        "proj": proj,
        "collision_prep": collision_prep,
        "layers": layers,
        "ring_graph": ring_graph,
        "C_nodes": C_nodes, # 重要：給 Nudge 用
        "C_edges": C_edges,
        "S_nodes": S_nodes,
        "S_edges": S_edges,
        "sea_graph": G,
        "sea_kdt": kdt,
        "sea_ok_set": sea_ok_set,
        "tgate_sea_connectors": tgate_sea_connectors,
    }