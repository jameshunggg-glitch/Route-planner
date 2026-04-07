# routing_map/build_aoi.py
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
from shapely.prepared import prep

from .config import RoutingMapConfig
from .geom_utils import make_aoi_bbox, build_projector_from_bbox
from .io_land import load_polys_in_bbox
from .land_layers import build_land_layers
from .smooth import smooth_union_for_features_from_union
from .rings import build_coast_rings_smooth_v2
from .cchain import build_C_chain_from_rings
from .features import extract_F_nodes_from_union_smooth
from .gates_a import build_gate_A_from_C_and_F_v1
from .gates_merge import merge_gates
from .inject import attach_F_to_nearest_C
from .gates_f import build_gate_F_primary
from .sea_nodes import build_sea_nodes_from_bundle, filter_sea_nodes
from .gates_b import build_gateB_connectors
from . import scgraph_bridge

from .gates_coverage import attach_gates_to_nearest_C, coverage_sample_gates_on_rings
from .rings import build_coast_rings_smooth_v2, build_envelope_and_taut_rings_v1
from .ring_graph import build_ring_nodes_edges, RingGraphBuildParams
from .geom_utils import expand_bbox_ll
from routing_map.e_t_transfer_v2 import build_e_t_transfer_edges, ETRampConfig
from .t_gate_connectors import build_tgate_sea_connectors, TGateSeaConnectorParams



def _pad_bbox_ll(bbox_ll, pad_deg: float):
    return expand_bbox_ll(bbox_ll, pad_deg)


def _norm_bbox_ll(bbox_ll):
    x0, y0, x1, y1 = map(float, bbox_ll)

    # normalize latitude ordering
    if y0 > y1:
        y0, y1 = y1, y0

    # longitude: keep x0>x1 as "dateline crossing" when span is huge
    # but fix obvious reversed input (small span but swapped)
    if x0 > x1:
        span_if_swapped = x0 - x1
        if span_if_swapped < 180.0:
            # likely just reversed input -> swap
            x0, x1 = x1, x0
        else:
            # treat as dateline crossing -> keep
            pass

    return (x0, y0, x1, y1)



def _bbox_crosses_dateline(bbox_ll):
    x0, y0, x1, y1 = map(float, bbox_ll)
    # 規則：允許用 x0 > x1 表達跨日界（例如 170 到 -170）
    return x0 > x1

def _split_bbox_dateline(bbox_ll):
    """
    Split dateline-crossing bbox into two normal bboxes.
    Input bbox format: (min_lon, min_lat, max_lon, max_lat)
    If min_lon <= max_lon: returns [bbox]
    If min_lon > max_lon (crosses dateline): returns two bboxes:
      [ (min_lon, min_lat, 180, max_lat), (-180, min_lat, max_lon, max_lat) ]
    """
    x0, y0, x1, y1 = map(float, bbox_ll)
    if x0 <= x1:
        return [(x0, y0, x1, y1)]
    return [(x0, y0, 180.0, y1), (-180.0, y0, x1, y1)]

def _merge_sc_bundles(bundles):
    """Merge multiple scgraph bundles into one bundle."""
    nodes_set = set()
    edges_set = set()
    stats = {"source": [], "node_count_parts": [], "edge_count_parts": []}

    for b in bundles:
        if not isinstance(b, dict):
            continue
        nodes = b.get("nodes", []) or []
        edges = b.get("edges", []) or []
        st = b.get("stats", {}) or {}

        # nodes
        for p in nodes:
            if isinstance(p, (tuple, list)) and len(p) == 2:
                nodes_set.add((float(p[0]), float(p[1])))

        # edges: canonicalize (a,b) order so duplicates merge
        for e in edges:
            if not (isinstance(e, (tuple, list)) and len(e) == 2):
                continue
            a, c = e
            if not (isinstance(a, (tuple, list)) and len(a) == 2 and isinstance(c, (tuple, list)) and len(c) == 2):
                continue
            pa = (float(a[0]), float(a[1]))
            pc = (float(c[0]), float(c[1]))
            if pa == pc:
                continue
            edges_set.add((pa, pc) if pa <= pc else (pc, pa))

        # stats (optional)
        stats["source"].append(st.get("source"))
        stats["node_count_parts"].append(st.get("node_count"))
        stats["edge_count_parts"].append(st.get("edge_count"))

    nodes_out = list(nodes_set)
    edges_out = list(edges_set)
    stats["node_count"] = len(nodes_out)
    stats["edge_count"] = len(edges_out)
    stats["source"] = "+".join([s for s in stats["source"] if s])

    return {"nodes": nodes_out, "edges": edges_out, "stats": stats}


def _filter_df_in_bbox(df, bbox_ll, lon_col="lon", lat_col="lat"):
    if df is None or len(df) == 0:
        return df
    x0, y0, x1, y1 = [float(v) for v in bbox_ll]

    if x0 <= x1:
        m_lon = df[lon_col].between(x0, x1)
    else:
        # dateline crossing: [x0,180] U [-180,x1]
        m_lon = (df[lon_col] >= x0) | (df[lon_col] <= x1)

    m_lat = df[lat_col].between(y0, y1)
    return df.loc[m_lon & m_lat].reset_index(drop=True)



def build_aoi(cfg: RoutingMapConfig) -> Dict[str, Any]:
    """
    AOI pipeline (Phase 1-ready): build land->rings->C/F/Gates + Sea subnet + Gate-B connectors.
    """
    # --- AOI bbox ---
    bbox_ll = cfg.aoi.bbox_ll
    #print("[land] bbox ll before =", bbox_ll)
    if bbox_ll is None:
        if cfg.aoi.origin_ll is None or cfg.aoi.dest_ll is None:
            raise ValueError("Provide either aoi.bbox_ll or (aoi.origin_ll & aoi.dest_ll)")
        bbox_ll = make_aoi_bbox(cfg.aoi.origin_ll, cfg.aoi.dest_ll, cfg.aoi.pad_deg)
    
    bbox_ll = _norm_bbox_ll(bbox_ll)
    #print("[land] bbox ll after =", bbox_ll)


    proj = build_projector_from_bbox(bbox_ll)

    # --- Land ---
    polys_ll = load_polys_in_bbox(cfg.land.shp_path, bbox_ll)
    print("[land] polys count =", len(polys_ll))
    layers = build_land_layers(
        polys_ll, proj,
        buffer_km=cfg.land.buffer_km,
        avoid_km=cfg.land.avoid_km,
        collision_safety_km=cfg.land.collision_safety_km,
        grid_size_m=cfg.land.precision_grid_m,
    )
    union_m = layers["UNION_M"]

    # --- Smooth union ---
    union_smooth_m = smooth_union_for_features_from_union(
        union_m,
        a2_smooth_km=cfg.smooth.a2_smooth_km,
        a2_tol_km=cfg.smooth.a2_tol_km,
    )

    # --- Rings ---
    ring_cfg = getattr(cfg, "rings", None)

    rings_env_m = None
    rings_taut_m = None
    rings_obj = None

    if ring_cfg is not None:
        # v1: uses clearance_m (meters) from cfg.rings
        collision_hard_m = layers["COLLISION_M"]  # 注意：這是「硬碰撞」的 metric 幾何

        ring_base_m, env_lines_m, taut_lines_m, rings_df, rings = build_envelope_and_taut_rings_v1(
            union_smooth_m,
            collision_hard_m=collision_hard_m,
            cfg=ring_cfg,
        )

        # 你現有下游（C-chain / gates coverage）吃 rings_m，所以先讓它用 taut
        rings_m = taut_lines_m

        # 額外保存（之後 debug / 分層繪圖用）
        rings_env_m = env_lines_m
        rings_taut_m = taut_lines_m
        rings_obj = rings
        ring_graph = None
        if rings_obj is not None:
            ring_graph = build_ring_nodes_edges(
                rings_obj,
                proj=proj,
                cfg=ring_cfg,
                params=RingGraphBuildParams(
                    e_angle_feature_deg=25.0,  # E feature 門檻（先保守）
                    t_max_gap_km=20.0,         # T densify 最大邊長
                    shared_tol_m=25.0,         # E∩T 合併距離
                ),
            )

    else:
        # legacy fallback (uses avoid_km from cfg.land.avoid_km)
        ring_base_m, rings_m, rings_df = build_coast_rings_smooth_v2(
            union_smooth_m,
            avoid_km=cfg.land.avoid_km,
            island_area_min_km2=5.0,
        )
    
    if "length_km" not in rings_df.columns:
        if "length_km_taut" in rings_df.columns:
            rings_df["length_km"] = rings_df["length_km_taut"].astype(float)
        elif "length_km_env" in rings_df.columns:
            rings_df["length_km"] = rings_df["length_km_env"].astype(float)
        elif "length_km_envelope" in rings_df.columns:
            rings_df["length_km"] = rings_df["length_km_envelope"].astype(float)
        else:
            # 最後手段：用幾何自己算（慢但可靠，且 ring 數量通常不大）
            from routing_map.routing_graph import haversine_km  # 你專案裡已有
            def _line_len_km(ls):
                coords = list(ls.coords)
                s = 0.0
                for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
                    s += haversine_km(x1, y1, x2, y2)
                return float(s)

            # rings 通常是 dict/list，這裡假設 rings_df 有 ring_id，且 rings_obj 可取到每個 ring 的線
            ring_len = {}
            # 你 v1 產物 rings 物件我不確定結構，所以做 best-effort
            # 若 rings 是 list[LineString] 且和 rings_df 對齊：
            if isinstance(rings_df, pd.DataFrame):
                if "ring_id" not in rings_df.columns:
                    if "id" in rings_df.columns:
                        rings_df = rings_df.rename(columns={"id": "ring_id"})
                    elif "rid" in rings_df.columns:
                        rings_df = rings_df.rename(columns={"rid": "ring_id"})
                    else:
                        rings_df["ring_id"] = np.arange(len(rings_df), dtype=int)
            if isinstance(rings, (list, tuple)) and isinstance(rings_df, pd.DataFrame) and len(rings) == len(rings_df):
                for rid, ls in zip(rings_df["ring_id"].tolist(), rings):
                    if hasattr(ls, "coords"):
                        ring_len[int(rid)] = _line_len_km(ls)

                for rid, ls in zip(rings_df["ring_id"].tolist(), rings):
                    if hasattr(ls, "coords"):
                        ring_len[int(rid)] = _line_len_km(ls)
            if isinstance(rings_df, pd.DataFrame) and "ring_id" in rings_df.columns:
                rings_df["length_km"] = rings_df["ring_id"].map(lambda rid: ring_len.get(int(rid), np.nan))
                rings_df["length_km"] = rings_df["length_km"].fillna(0.0)

            
    
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
    et_out = build_e_t_transfer_edges(
        ring_graph,
        collision_hard_m=layers["COLLISION_M"],  # 你 metric 的硬碰撞（或你想用 hard+clearance buffer）
        cfg=et_cfg,
        shared_edge_cost=0.0,
    )
    ring_graph.update(et_out)




    # --- C chain ---
    C_nodes, C_edges = build_C_chain_from_rings(
        rings_m, proj,
        c_step_km=cfg.cchain.c_step_km,
        round_decimals=cfg.cchain.round_decimals,
    )

    # --- Feature nodes ---
    F_nodes = extract_F_nodes_from_union_smooth(
        union_smooth_m, proj,
        sample_step_km=cfg.features.f_sample_step_km,
        angle_deg_min=cfg.features.f_angle_deg_min,
        nms_radius_km=cfg.features.f_nms_radius_km,
        max_keep=cfg.features.f_max_keep,
    )

    # Attach F -> nearest C (for Gate-F primary selection)
    F_att = attach_F_to_nearest_C(F_nodes, C_nodes)

    # --- Gates A/F ---
    Gate_A = build_gate_A_from_C_and_F_v1(
        C_nodes, rings_df, F_nodes,
        min_ring_length_km=cfg.gate_a.min_ring_length_km,
        short_ring_no_gate_km=cfg.gate_a.short_ring_no_gate_km,
        short_ring_one_gate_km=cfg.gate_a.short_ring_one_gate_km,
        snap_to_f_km=cfg.gate_a.snap_to_f_km,
    )

    Gate_F = build_gate_F_primary(
        F_att, rings_df,
        min_spacing_km=cfg.gate_f.min_spacing_km,
        max_per_ring=cfg.gate_f.max_per_ring,
        global_max=cfg.gate_f.global_max,
        round_decimals=cfg.cchain.round_decimals,
    )

    Gate_all = merge_gates(Gate_A, Gate_F, round_decimals=cfg.cchain.round_decimals)

    # 僅用 AOI bbox 內的 gates 做 coverage/connectability（避免算到外面）
    Gate_all_aoi = _filter_df_in_bbox(Gate_all, bbox_ll)

    # --- Coverage (Gate-A + Gate-F) ---
    # 1) 補 s_km（靠近哪個 C 就用它的 s_km）
    Gate_all_att = attach_gates_to_nearest_C(
        Gate_all_aoi, C_nodes, rings_df,
        debug=getattr(cfg.coverage, "debug", True),
    )

    # 2) spacing 抽樣（閉合 ring）
    Gate_all_cov = coverage_sample_gates_on_rings(
        Gate_all_att,
        gate_spacing_km=float(cfg.coverage.gate_spacing_km),
        min_per_ring=int(cfg.coverage.min_per_ring),
        prefer_source_order=list(cfg.coverage.prefer_source_order),
        debug=getattr(cfg.coverage, "debug", True),
    )

    # 給 Gate-B 用的 gate_uid（確保 connectors 可以回指 gate）
    Gate_all_cov = Gate_all_cov.copy()
    Gate_all_cov["gate_uid"] = np.arange(len(Gate_all_cov), dtype=int)

    # --- Sea subnet (scgraph) ---
    #bbox_ll_sea = _pad_bbox_ll(bbox_ll, pad_deg=float(cfg.sea.aoi_pad_deg))
    #bundle = scgraph_bridge.sc_edges_in_bbox(bbox_ll=bbox_ll_sea)  # 你現在的 adapter 會回 nodes/edges/stats
    #S_nodes, S_edges, G, kdt = build_sea_nodes_from_bundle(proj, bundle)
    bbox_ll_sea = _pad_bbox_ll(bbox_ll, pad_deg=float(cfg.sea.aoi_pad_deg))

    # --- dateline-safe: split bbox if needed, fetch multiple bundles, then merge ---
    sea_bboxes = _split_bbox_dateline(bbox_ll_sea)
    bundles = [scgraph_bridge.sc_edges_in_bbox(bbox_ll=b) for b in sea_bboxes]
    bundle = _merge_sc_bundles(bundles)

    S_nodes, S_edges, G, kdt = build_sea_nodes_from_bundle(proj, bundle)


    sea_ok_set = filter_sea_nodes(
        S_nodes, G,
        deg_min=int(cfg.sea.deg_min),
        use_largest_component_only=bool(cfg.sea.use_largest_component_only),
    )

    # --- Gate-B connectors (Gate -> Sea) ---
    collision_prep = prep(layers["COLLISION_M"])
    # --- T-gate -> Sea connectors (T-ring gate candidates -> Sea nodes) ---
    # (keep sea_idx as original S_nodes row-index, so viz can do S_nodes.iloc[sea_idx])
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
                k_connect=2,
                topN=60,
                r_connect_km=200.0,
                enable_sector_filter=False,
                sector_deg=110.0,
                do_collision_check=True,
                do_repair=True,
            ),
        )
        # optional: only keep connectors to "ok" sea nodes (largest component / deg_min filtered)
        if tgate_sea_connectors is not None and len(tgate_sea_connectors) > 0 and sea_ok_set is not None:
            ok = set(sea_ok_set)
            tgate_sea_connectors = (
                tgate_sea_connectors[tgate_sea_connectors["sea_idx"].astype(int).isin(ok)]
                .reset_index(drop=True)
            )
    except Exception:
        tgate_sea_connectors = None

    gateB_connectors = build_gateB_connectors(
        Gate_all_cov, S_nodes,
        sea_ok_set=sea_ok_set,
        kdt=kdt,
        collision_prep=collision_prep,
        top_n=cfg.sea.candidate_top_n,
        r_max_km=cfg.sea.r_max_km,
        k_connect=cfg.sea.k_connect,
    )

    # Gate-B kept gates（能連上才算 Gate-B）
    Gate_B_kept_gates = Gate_all_cov.iloc[0:0].copy()
    if gateB_connectors is not None and len(gateB_connectors) > 0:
        if "gate_uid" in gateB_connectors.columns:
            keep_uid = set(gateB_connectors["gate_uid"].astype(int).tolist())
            Gate_B_kept_gates = Gate_all_cov[Gate_all_cov["gate_uid"].isin(keep_uid)].reset_index(drop=True)
        elif "gate_row" in gateB_connectors.columns:
            keep_row = sorted(set(gateB_connectors["gate_row"].astype(int).tolist()))
            Gate_B_kept_gates = Gate_all_cov.iloc[keep_row].reset_index(drop=True)


    # -------------------------
    # node_id mappings (for pipeline / heuristic / viz)
    # -------------------------
    id2ll: Dict[str, Tuple[float, float]] = {}
    id2xy: Dict[str, Tuple[float, float]] = {}

    # Sea nodes (S:lon,lat)
    if isinstance(S_nodes, pd.DataFrame) and len(S_nodes) and "node_id" in S_nodes.columns:
        for r in S_nodes.itertuples(index=False):
            nid = str(getattr(r, "node_id"))
            id2ll[nid] = (float(getattr(r, "lon")), float(getattr(r, "lat")))
            if hasattr(r, "x_m") and hasattr(r, "y_m"):
                id2xy[nid] = (float(getattr(r, "x_m")), float(getattr(r, "y_m")))

    # Ring nodes (E:/T:)
    if isinstance(ring_graph, dict):
        for _k, _prefix in [("E_nodes", "E"), ("T_nodes", "T")]:
            df = ring_graph.get(_k)
            if isinstance(df, pd.DataFrame) and len(df):
                if "node_key" not in df.columns and "node_id" in df.columns:
                    df = df.copy()
                    df["node_key"] = df["node_id"].map(lambda i: f"{_prefix}:{int(i)}")
                for r in df.itertuples(index=False):
                    nid = str(getattr(r, "node_key"))
                    id2ll[nid] = (float(getattr(r, "lon")), float(getattr(r, "lat")))
                    if hasattr(r, "x_m") and hasattr(r, "y_m"):
                        id2xy[nid] = (float(getattr(r, "x_m")), float(getattr(r, "y_m")))


    return {
        "cfg": cfg,
        "bbox_ll": bbox_ll,
        "id2ll": id2ll,
        "id2xy": id2xy,
        "bbox_ll_sea": bbox_ll_sea,
        "bbox_ll_sea_parts": sea_bboxes,
        "proj": proj,
        "collision_prep": collision_prep,

        "polys_ll": polys_ll,
        "layers": layers,
        "union_smooth_m": union_smooth_m,

        "ring_base_m": ring_base_m,
        "rings_m": rings_m,
        "rings_df": rings_df,
        "rings_env_m": rings_env_m,
        "rings_taut_m": rings_taut_m,
        "rings_obj": rings_obj,
        "ring_graph": ring_graph,

        "C_nodes": C_nodes,
        "C_edges": C_edges,

        "F_nodes": F_nodes,
        "F_att": F_att,

        "Gate_A": Gate_A,
        "Gate_F": Gate_F,
        "Gate_all": Gate_all,

        # Coverage outputs
        "Gate_all_aoi": Gate_all_aoi,
        "Gate_all_cov": Gate_all_cov,

        # Sea outputs
        "sc_bundle_stats": bundle.get("stats", {}) if isinstance(bundle, dict) else {},
        "S_nodes": S_nodes,
        "S_edges": S_edges,
        "sea_graph": G,
        "sea_kdt": kdt,
        "sea_ok_set": sea_ok_set,
        "tgate_sea_connectors": tgate_sea_connectors,


        # Gate-B outputs
        "gateB_connectors": gateB_connectors,
        "Gate_B_kept_gates": Gate_B_kept_gates,
    }
