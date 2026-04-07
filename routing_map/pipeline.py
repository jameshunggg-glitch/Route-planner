from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .routing_graph import (
    build_base_graph, haversine_km,
    infer_layer_mask_from_etype,
    L_BASE_SEA, L_RING_E, L_RING_T, L_ET_TRANSFER, L_TGATE_SEA, L_GATEWAY, L_NE_CORRIDOR, L_NW_CORRIDOR, L_INJECT,
    B_HIGH_LAT,
)
from .snap import snap_pair_component_aware, inject_point_edges, compute_multiworld_policies_for_point
from .repairer import PathRepairer, RepairConfig
from .path_simplifier import simplify_path_visibility
from .geom_utils import get_projector, get_collision_metric

LonLat = Tuple[float, float]
BBoxLL = Tuple[float, float, float, float]



@dataclass
class RoutePolicy:
    """Runtime routing policy.

    This is designed to mirror a future UI:
    - layer toggles: enable/disable certain classes of edges (NE/NW corridors, gateways, etc.)
    - ban toggles: activate certain ban reasons (high-lat cap now; ECA/ice/etc later)
    """

    # Hard operational cap (for now: global merchant shipping default)
    hard_lat_cap_deg: float = 70.0

    # Layer toggles (future UI switches)
    enable_gateways: bool = True      # canals/straits corridors (reserved)
    enable_northeast: bool = False    # NSR/NE corridor (reserved)
    enable_northwest: bool = False    # NWP/NW corridor (reserved)

    # Which ban reasons are enforced right now
    active_ban_mask: int = B_HIGH_LAT

    def enabled_layers_mask(self) -> int:
        m = (L_BASE_SEA | L_RING_E | L_RING_T | L_ET_TRANSFER | L_TGATE_SEA | L_INJECT)
        if self.enable_gateways:
            m |= L_GATEWAY
        if self.enable_northeast:
            m |= L_NE_CORRIDOR
        if self.enable_northwest:
            m |= L_NW_CORRIDOR
        return int(m)


def apply_policy_view(G: nx.Graph, policy: RoutePolicy) -> nx.Graph:
    """Return a *view* of G with edges filtered by the policy (no up-front copy cost)."""
    enabled_layers = int(policy.enabled_layers_mask())
    active_bans = int(policy.active_ban_mask)
    hard_lat = float(policy.hard_lat_cap_deg)

    def _edge_ok(u: Any, v: Any) -> bool:
        try:
            attr = G.get_edge_data(u, v) or {}
        except Exception:
            return False

        # layer
        layer = int(attr.get("layer_mask") or 0)
        if layer == 0:
            layer = int(infer_layer_mask_from_etype(str(attr.get("etype", ""))))

        # lat cap (prefer cached attr; fallback to node attrs)
        lat_max = attr.get("lat_max_abs", None)
        if lat_max is None:
            try:
                lat_max = max(abs(float(G.nodes[u].get("lat"))), abs(float(G.nodes[v].get("lat"))))
            except Exception:
                lat_max = None

        ban = int(attr.get("ban_mask") or 0)
        if lat_max is not None and float(lat_max) > hard_lat:
            ban |= B_HIGH_LAT

        if (layer & enabled_layers) == 0:
            return False
        if (ban & active_bans) != 0:
            return False

        # final hard safety check (even if bans are disabled accidentally)
        if lat_max is not None and float(lat_max) > hard_lat:
            return False

        return True

    return nx.subgraph_view(G, filter_edge=_edge_ok)



@dataclass
class GraphConfig:
    bbox_ll: Optional[BBoxLL] = None
    include_sea: bool = True
    include_rings: bool = True
    include_et: bool = True
    include_tgate_sea: bool = True
    # legacy (kept for compatibility)
    include_cc: bool = False
    include_gateb_sea: bool = False
    include_c_gateb: bool = False

    max_sea_edges: Optional[int] = None
    max_ring_edges: Optional[int] = None
    weight_unit: str = "km"


@dataclass
class SnapConfig:
    k_near: int = 30
    r_max_km: float = 150.0
    k_inject: int = 4
    prefer_ok_set: bool = True
    allow_fallback_non_ok: bool = True
    allow_radius_fallback: bool = True
    do_nudge: bool = True
    k_near_coast: int = 80
    r_max_km_coast: Optional[float] = None
    enable_local_entrance_aug: bool = True
    aug_dist_trigger_km: float = 60.0
    aug_delta_end_km: float = 120.0
    aug_angle_trigger_deg: float = 110.0
    aug_seed_neighbors_cap: int = 12
    aug_seed_count: int = 1
    # === multiworld snap ===
    R_NEAR_COAST_KM: float = 120.0
    S_MAX_SNAP_KM: float = 200.0
    # per-run override (new): allow wrapper to force ring/sea worldview
    force_start_policy: Optional[str] = None   # "R" or "S" or None
    force_end_policy: Optional[str] = None     # "R" or "S" or None


@dataclass
class SimplifyConfig:
    enabled: bool = True
    window_size: int = 80
    max_tries: int = 300
    use_prepared_collision: bool = True
    dateline_unwrap: bool = True
    wrap_output_lon: bool = True
    strategy: str = "linear_backscan"


@dataclass
class RunConfig:
    do_repair: bool = True
    do_simplify: bool = True
    debug: bool = True

    # === multiworld selection (safety-aware) ===
    # Switch to rank2 only if it is meaningfully safer (fewer un-fixed collisions)
    select_delta_failed_edges: int = 2   # ΔF >= 2
    select_pct: float = 0.10            # 10%
    select_k_km: float = 150.0          # 150 km


@dataclass
class RouteResult:
    # inputs
    origin_ll: Optional[LonLat] = None
    dest_ll: Optional[LonLat] = None

    # multiworld (if used)
    multiworld_combo: Optional[str] = None  # e.g. 'RS'
    multiworld_rank: Optional[int] = None   # 1=best, 2=runner-up, ...
    multiworld_alternatives: Optional[List['RouteResult']] = None  # rank>=2 results
    multiworld_table: Optional[List[Dict[str, Any]]] = None  # lightweight summary for all combos

    # snap
    start_ll_snap: Optional[LonLat] = None  # nearest picked graph node (for debug/viz)
    end_ll_snap: Optional[LonLat] = None
    start_ll_used: Optional[LonLat] = None  # injected query node lonlat (used by final assembly)
    end_ll_used: Optional[LonLat] = None
    snap_debug: Optional[Dict[str, Any]] = None

    # graph
    G: Optional[Any] = None
    graph_stats: Optional[Any] = None

    # A*
    path_nodes: Optional[List[Any]] = None
    path_ll_raw: Optional[List[LonLat]] = None

    # repair / simplify
    repair_stats: Optional[Any] = None
    repair_debug: Optional[Any] = None
    path_ll_repaired: Optional[List[LonLat]] = None
    path_ll_simplified: Optional[List[LonLat]] = None
    path_ll_final: Optional[List[LonLat]] = None

    lengths_km: Optional[Dict[str, float]] = None

    # errors
    error: Optional[str] = None


def _path_len_km(path_ll: Optional[List[LonLat]]) -> float:
    if not path_ll or len(path_ll) < 2:
        return 0.0
    s = 0.0
    for a, b in zip(path_ll, path_ll[1:]):
        s += float(haversine_km(a, b))
    return float(s)


def run_p2p(
    out: Dict[str, Any],
    origin_ll: LonLat,
    dest_ll: LonLat,
    *,
    graph_cfg: Optional[GraphConfig] = None,
    snap_cfg: Optional[SnapConfig] = None,
    repair_cfg: Optional[RepairConfig] = None,
    simplify_cfg: Optional[SimplifyConfig] = None,
    run_cfg: Optional[RunConfig] = None,
    G_in: Optional[Any] = None,
    policy: Optional[RoutePolicy] = None,
) -> RouteResult:
    """End-to-end routing runner for the new Sea + E/T ring graph.

    Assumptions:
    - `out` already comes from build_aoi.
    - `routing_graph.build_base_graph` is the rings-compatible version.
    - snap.py already snaps to sea candidates and injects edges.
    """

    graph_cfg = graph_cfg or GraphConfig(bbox_ll=out.get("bbox_ll"))
    snap_cfg = snap_cfg or SnapConfig()
    simplify_cfg = simplify_cfg or SimplifyConfig()
    run_cfg = run_cfg or RunConfig()

    policy = policy or RoutePolicy()
    res = RouteResult(origin_ll=origin_ll, dest_ll=dest_ll)

    bbox_ll = graph_cfg.bbox_ll or out.get("bbox_ll")

    # projector + collision
    try:
        proj = get_projector(out, bbox_ll=bbox_ll)
    except Exception:
        proj = out.get("proj", None)

    collision_m, _is_prep = get_collision_metric(out, prefer_prepared=True)

    # build graph
    try:
        if G_in is not None:
            G, stats = G_in, None
        else:
            G, stats = build_base_graph(
                out,
                include_sea=graph_cfg.include_sea,
                include_cc=graph_cfg.include_cc,
                include_gateb_sea=graph_cfg.include_gateb_sea,
                include_c_gateb=graph_cfg.include_c_gateb,
                include_rings=graph_cfg.include_rings,
                include_et=graph_cfg.include_et,
                include_tgate_sea=graph_cfg.include_tgate_sea,
                max_sea_edges=graph_cfg.max_sea_edges,
                max_ring_edges=graph_cfg.max_ring_edges,
                weight_unit=graph_cfg.weight_unit,
                bbox_ll=bbox_ll,
                hard_lat_cap_deg=policy.hard_lat_cap_deg,
            )
        res.G = G
        res.graph_stats = stats
        if run_cfg.debug:
            try:
                print(f"[pipeline][graph] nodes={G.number_of_nodes()} edges={G.number_of_edges()} stats={stats}")
            except Exception:
                print("[pipeline][graph] built")
    except Exception as e:
        res.error = f"graph_build_error: {e}"
        return res
    
    extra = {}
    if getattr(snap_cfg, "force_start_policy", None) is not None:
        extra["start_policy"] = snap_cfg.force_start_policy
    if getattr(snap_cfg, "force_end_policy", None) is not None:
        extra["end_policy"] = snap_cfg.force_end_policy


    # snap pair
    try:
        pair = snap_pair_component_aware(
            out,
            origin_ll,
            dest_ll,
            k_near=snap_cfg.k_near,
            r_max_km=snap_cfg.r_max_km,
            k_inject=snap_cfg.k_inject,
            prefer_ok_set=snap_cfg.prefer_ok_set,
            allow_fallback_non_ok=snap_cfg.allow_fallback_non_ok,
            allow_radius_fallback=snap_cfg.allow_radius_fallback,
            do_nudge=snap_cfg.do_nudge,
            k_near_coast=snap_cfg.k_near_coast,
            r_max_km_coast=snap_cfg.r_max_km_coast,
            enable_local_entrance_aug=snap_cfg.enable_local_entrance_aug,
            aug_dist_trigger_km=snap_cfg.aug_dist_trigger_km,
            aug_delta_end_km=snap_cfg.aug_delta_end_km,
            aug_angle_trigger_deg=snap_cfg.aug_angle_trigger_deg,
            aug_seed_neighbors_cap=snap_cfg.aug_seed_neighbors_cap,
            aug_seed_count=snap_cfg.aug_seed_count,
            **extra,
        )

        # choose candidates (these are EXISTING graph nodes we will connect to)
        start_pick = pair.start_pick or (pair.start.candidates[: snap_cfg.k_inject] if pair.start else [])
        end_pick = pair.end_pick or (pair.end.candidates[: snap_cfg.k_inject] if pair.end else [])

        
        # base query node ids that will be inserted into the graph:
        # We inject two virtual query nodes into the graph so A* keys are always stable.
        start_id = "Q:START"
        end_id = "Q:END"

        start_ll_used = (float(pair.start.p_used_ll[0]), float(pair.start.p_used_ll[1]))
        end_ll_used = (float(pair.end.p_used_ll[0]), float(pair.end.p_used_ll[1]))

        # persist used lonlat for final polyline assembly
        res.start_ll_used = start_ll_used
        res.end_ll_used = end_ll_used

        # nearest picked graph node (for debug / viz)
        start_snap = (float(start_pick[0].node_ll[0]), float(start_pick[0].node_ll[1])) if start_pick else None
        end_snap = (float(end_pick[0].node_ll[0]), float(end_pick[0].node_ll[1])) if end_pick else None

        res.start_ll_snap = start_snap  # closest existing node (not the injected key)
        res.end_ll_snap = end_snap
        res.snap_debug = getattr(pair, "debug", None) if hasattr(pair, "debug") else None

        if run_cfg.debug:
            print(
                f"[pipeline][snap] start_in={origin_ll} used={start_ll_used} -> pick={start_snap} | "
                f"end_in={dest_ll} used={end_ll_used} -> pick={end_snap} | {res.snap_debug}"
            )

        if not start_pick or not end_pick:
            res.error = "snap_failed"
            return res

        # inject (IMPORTANT: inject using node_id keys for A*)
        inject_point_edges(G, start_id, start_ll_used, start_pick, k_inject=snap_cfg.k_inject, etype="inject")
        inject_point_edges(G, end_id, end_ll_used, end_pick, k_inject=snap_cfg.k_inject, etype="inject")


    except Exception as e:
        res.error = f"snap_inject_error: {e}"
        return res

    # A*
    try:
        def _node_ll(n):
            # Prefer node attrs (covers ring nodes and sea nodes)
            if n in G.nodes:
                nd = G.nodes[n]
                if isinstance(nd, dict) and "lon" in nd and "lat" in nd:
                    return (float(nd["lon"]), float(nd["lat"]))
            # Fallback: parse node_id encoding "...lon,lat"
            s = str(n)
            if ":" in s:
                s2 = s.split(":", 1)[1]
            else:
                s2 = s
            if "," in s2:
                a, b = s2.split(",", 1)
                try:
                    return (float(a), float(b))
                except Exception:
                    pass
            raise KeyError(f"cannot get lon/lat for node {n}")

        G_view = apply_policy_view(G, policy)

        path_nodes = nx.astar_path(
            G_view,
            start_id,
            end_id,
            heuristic=lambda a, b: haversine_km(_node_ll(a), _node_ll(b)),
            weight="weight",
        )
        if run_cfg.debug:
            print("[pipeline][astar] start_id in G?", start_id in G)
            print("[pipeline][astar] end_id in G?", end_id in G)
        res.path_nodes = list(path_nodes)
        res.path_ll_raw = [_node_ll(n) for n in path_nodes]
        if run_cfg.debug:
            print(f"[pipeline][A*] n_nodes={len(path_nodes)}")
    except Exception as e:
        res.error = f"astar_error: {e}"
        return res

# repair
    path_ll_work = res.path_ll_raw
    if run_cfg.do_repair and repair_cfg is not None and collision_m is not None:
        try:
            rep = PathRepairer(repair_cfg)
            out_rep = rep.repair_path(G, res.path_nodes, collision_m=collision_m, proj=proj)
            res.path_ll_repaired = out_rep.path_ll
            res.repair_stats = out_rep.stats
            res.repair_debug = getattr(out_rep, "debug", None)
            path_ll_work = res.path_ll_repaired
            if run_cfg.debug:
                failed = getattr(out_rep.stats, "failed_edges", None)
                if failed is None:
                    print(f"[pipeline][repair] repaired_edges={out_rep.stats.repaired_edges} colliding={out_rep.stats.colliding_edges}")
                else:
                    print(f"[pipeline][repair] repaired_edges={out_rep.stats.repaired_edges} colliding={out_rep.stats.colliding_edges} failed={failed}")
        except Exception as e:
            res.path_ll_repaired = path_ll_work
            if run_cfg.debug:
                print(f"[pipeline][repair][warn] {e}")

    # simplify
    if run_cfg.do_simplify and simplify_cfg.enabled and collision_m is not None:
        try:
            simp_ll, simp_stats = simplify_path_visibility(
                path_ll_work,
                collision_m=collision_m,
                proj=proj,
                window_size=simplify_cfg.window_size,
                max_tries=simplify_cfg.max_tries,
                use_prepared_collision=simplify_cfg.use_prepared_collision,
                dateline_unwrap=simplify_cfg.dateline_unwrap,
                wrap_output_lon=simplify_cfg.wrap_output_lon,
                strategy=simplify_cfg.strategy,
            )
            res.path_ll_simplified = simp_ll
            if run_cfg.debug:
                print(f"[pipeline][simplify] {simp_stats.n_in}->{simp_stats.n_out} checks={simp_stats.n_checks}")
        except Exception as e:
            res.path_ll_simplified = path_ll_work
            if run_cfg.debug:
                print(f"[pipeline][simplify][warn] {e}")

    # final (ALWAYS include input->used and used->input legs for visualization)
    core = res.path_ll_simplified or res.path_ll_repaired or res.path_ll_raw or []

    path_final: List[LonLat] = []

    origin_in = res.origin_ll  # original user input
    dest_in = res.dest_ll      # original user input
    start_used = getattr(res, 'start_ll_used', None)  # injected query node lonlat
    end_used = getattr(res, 'end_ll_used', None)      # injected query node lonlat

    def _push(pt: LonLat):
        pt2 = (float(pt[0]), float(pt[1]))
        if not path_final or path_final[-1] != pt2:
            path_final.append(pt2)

    # prepend: input -> used
    if origin_in is not None and start_used is not None:
        _push(origin_in)
        _push(start_used)

    # core path (used -> ... -> used)
    for p in core:
        _push(p)

    # append: used -> input
    if dest_in is not None and end_used is not None:
        _push(end_used)
        _push(dest_in)

    res.path_ll_final = path_final


    lengths = {
        "raw": _path_len_km(res.path_ll_raw),
        "repaired": _path_len_km(res.path_ll_repaired),
        "simplified": _path_len_km(res.path_ll_simplified),
        "final": _path_len_km(res.path_ll_final),
    }
    res.lengths_km = lengths

    if run_cfg.debug:
        print(f"[pipeline][done] lengths_km={lengths}")

    return res

def run_p2p_multiworld(
    out: Dict[str, Any],
    origin_ll: LonLat,
    dest_ll: LonLat,
    *,
    graph_cfg: Optional[GraphConfig] = None,
    snap_cfg: Optional[SnapConfig] = None,
    repair_cfg: Optional[RepairConfig] = None,
    simplify_cfg: Optional[SimplifyConfig] = None,
    run_cfg: Optional[RunConfig] = None,
    G_in: Optional[nx.Graph] = None,
    policy: Optional[RoutePolicy] = None,
) -> RouteResult:
    """
    Multi-worldview runner:
    - Each endpoint chooses policy in {R,S} after pruning:
        R_NEAR_COAST_KM=120km, S_MAX_SNAP_KM=200km (from snap_cfg)
    - Runs up to 4 combos: RR/RS/SR/SS
    - Each combo runs full pipeline: snap -> inject -> A* -> repair -> simplify
    - Select winner by final simplified length (res.lengths_km["final"])
    """
    graph_cfg = graph_cfg or GraphConfig(bbox_ll=out.get("bbox_ll"))
    snap_cfg = snap_cfg or SnapConfig()
    simplify_cfg = simplify_cfg or SimplifyConfig()
    run_cfg = run_cfg or RunConfig()

    policy = policy or RoutePolicy()
    bbox_ll = graph_cfg.bbox_ll or out.get("bbox_ll")

    # --- projector + collision (for "in_collision" pruning safety) ---
    try:
        proj = get_projector(out, bbox_ll=bbox_ll)
    except Exception:
        proj = out.get("proj", None)

    collision_m, _is_prep = get_collision_metric(out, prefer_prepared=True)

    # --- build base graph ONCE ---
    try:
        if G_in is not None:
            G_base, stats = G_in, None
        else:
            G_base, stats = build_base_graph(
                out,
                include_sea=graph_cfg.include_sea,
                include_cc=graph_cfg.include_cc,
                include_gateb_sea=graph_cfg.include_gateb_sea,
                include_c_gateb=graph_cfg.include_c_gateb,
                include_rings=graph_cfg.include_rings,
                include_et=graph_cfg.include_et,
                include_tgate_sea=graph_cfg.include_tgate_sea,
                max_sea_edges=graph_cfg.max_sea_edges,
                max_ring_edges=graph_cfg.max_ring_edges,
                weight_unit=graph_cfg.weight_unit,
                bbox_ll=bbox_ll,
                hard_lat_cap_deg=policy.hard_lat_cap_deg,
            )

        if run_cfg.debug:
            try:
                print(f"[pipeline][graph] nodes={G_base.number_of_nodes()} edges={G_base.number_of_edges()} stats={stats}")
            except Exception:
                print("[pipeline][graph] built (multiworld)")
    except Exception as e:
        return RouteResult(origin_ll=origin_ll, dest_ll=dest_ll, error=f"graph_build_error: {e}")

    # --- compute pruning policies (R/S) using snap utilities (KDTree-based) ---
    # Using compute_multiworld_policies_for_point() avoids the common situation where
    # df-based distance estimates become None (e.g., missing x_m/y_m), which forces
    # running all 4 combos every time.
    R_NEAR = float(getattr(snap_cfg, "R_NEAR_COAST_KM", 120.0))
    S_MAX  = float(getattr(snap_cfg, "S_MAX_SNAP_KM", 200.0))

    d_ring_o: Optional[float] = None
    d_sea_o:  Optional[float] = None
    d_ring_d: Optional[float] = None
    d_sea_d:  Optional[float] = None
    P_start: List[str] = ["R", "S"]
    P_end:   List[str] = ["R", "S"]

    try:
        pr_o = compute_multiworld_policies_for_point(out, origin_ll, R_NEAR_COAST_KM=R_NEAR, S_MAX_SNAP_KM=S_MAX)
        pr_d = compute_multiworld_policies_for_point(out, dest_ll,   R_NEAR_COAST_KM=R_NEAR, S_MAX_SNAP_KM=S_MAX)

        # distances (km)
        try:
            d_ring_o = float(pr_o.get("d_ring_km"))  # type: ignore[arg-type]
        except Exception:
            d_ring_o = None
        try:
            d_sea_o  = float(pr_o.get("d_sea_km"))   # type: ignore[arg-type]
        except Exception:
            d_sea_o = None
        try:
            d_ring_d = float(pr_d.get("d_ring_km"))  # type: ignore[arg-type]
        except Exception:
            d_ring_d = None
        try:
            d_sea_d  = float(pr_d.get("d_sea_km"))   # type: ignore[arg-type]
        except Exception:
            d_sea_d = None

        P_start = list(pr_o.get("policies", ["R", "S"])) or ["R", "S"]
        P_end   = list(pr_d.get("policies", ["R", "S"])) or ["R", "S"]

    except Exception:
        # Fallback to legacy df-based approximation (conservative; may run more combos)
        # NOTE: this should be rare; prefer fixing out["sea_kdt"] / ring KDTree builders.
        # --- helper: min distance (km) from point to node dataframe using x_m/y_m ---
        def _min_df_dist_km(p_ll: LonLat, df) -> Optional[float]:
            if df is None:
                return None
            cols = getattr(df, "columns", [])
            if "x_m" not in cols or "y_m" not in cols:
                return None
            if proj is None or not hasattr(proj, "ll2m"):
                return None
            x0, y0 = proj.ll2m(float(p_ll[0]), float(p_ll[1]))
            try:
                import numpy as np  # type: ignore
                xs = df["x_m"].to_numpy(dtype=float)
                ys = df["y_m"].to_numpy(dtype=float)
                if xs.size == 0:
                    return None
                d2 = (xs - x0) ** 2 + (ys - y0) ** 2
                return float(np.sqrt(d2.min()) / 1000.0)
            except Exception:
                best = None
                for r in df.itertuples(index=False):
                    try:
                        dx = float(getattr(r, "x_m")) - x0
                        dy = float(getattr(r, "y_m")) - y0
                        d = (dx * dx + dy * dy) ** 0.5 / 1000.0
                        best = d if best is None else min(best, d)
                    except Exception:
                        continue
                return best

        def _in_collision(p_ll: LonLat) -> bool:
            if collision_m is None or proj is None or not hasattr(proj, "ll2m"):
                return False
            try:
                from shapely.geometry import Point
                x, y = proj.ll2m(float(p_ll[0]), float(p_ll[1]))
                return bool(collision_m.contains(Point(x, y)))
            except Exception:
                return False

        sea_nodes = out.get("sea_nodes", out.get("S_nodes", None))
        e_nodes = out.get("e_nodes", out.get("E_nodes", None))
        t_nodes = out.get("t_nodes", out.get("T_nodes", None))

        d_sea_o = _min_df_dist_km(origin_ll, sea_nodes)
        d_sea_d = _min_df_dist_km(dest_ll, sea_nodes)

        d_e_o = _min_df_dist_km(origin_ll, e_nodes)
        d_t_o = _min_df_dist_km(origin_ll, t_nodes)
        d_ring_o = min([v for v in [d_e_o, d_t_o] if v is not None], default=None)

        d_e_d = _min_df_dist_km(dest_ll, e_nodes)
        d_t_d = _min_df_dist_km(dest_ll, t_nodes)
        d_ring_d = min([v for v in [d_e_d, d_t_d] if v is not None], default=None)

        # --- pruning rules (necessary pruning only) ---
        def _allowed_policies(p_ll: LonLat, d_ring: Optional[float], d_sea: Optional[float]) -> List[str]:
            allow_R = True
            allow_S = True
            if _in_collision(p_ll):
                allow_R = True
            else:
                if d_ring is not None and d_ring > R_NEAR:
                    allow_R = False
            if d_sea is not None and d_sea > S_MAX:
                allow_S = False
            out_p: List[str] = []
            if allow_R:
                out_p.append("R")
            if allow_S:
                out_p.append("S")
            if not out_p:
                out_p = ["R"]
            return out_p

        P_start = _allowed_policies(origin_ll, d_ring_o, d_sea_o)
        P_end   = _allowed_policies(dest_ll,   d_ring_d, d_sea_d)

    combos: List[Tuple[str, str]] = [(a, b) for a in P_start for b in P_end]
    if run_cfg.debug:
        print(
            f"[pipeline][multiworld][prune] "
            f"start d_ring={d_ring_o} d_sea={d_sea_o} -> {P_start} | "
            f"end d_ring={d_ring_d} d_sea={d_sea_d} -> {P_end} | "
            f"combos={[''.join(c) for c in combos]}"
        )

# --- run each combo and select best by final length ---
    results_ok: List[RouteResult] = []
    table: List[Dict[str, Any]] = []

    for sp, ep in combos:
        combo = f"{sp}{ep}"

        # copy base graph because inject mutates the graph
        try:
            G_run = G_base.copy()
        except Exception:
            # in case graph copy fails for any reason, rebuild (slower but safe)
            G_run, _ = build_base_graph(
                out,
                include_sea=graph_cfg.include_sea,
                include_cc=graph_cfg.include_cc,
                include_gateb_sea=graph_cfg.include_gateb_sea,
                include_c_gateb=graph_cfg.include_c_gateb,
                include_rings=graph_cfg.include_rings,
                include_et=graph_cfg.include_et,
                include_tgate_sea=graph_cfg.include_tgate_sea,
                max_sea_edges=graph_cfg.max_sea_edges,
                max_ring_edges=graph_cfg.max_ring_edges,
                weight_unit=graph_cfg.weight_unit,
                bbox_ll=bbox_ll,
                hard_lat_cap_deg=policy.hard_lat_cap_deg,
            )

        snap_cfg_run = replace(snap_cfg, force_start_policy=sp, force_end_policy=ep)
        if run_cfg.debug:
            print(f"[pipeline][multiworld][run] combo={combo} start_policy={sp} end_policy={ep}")

        res = run_p2p(
            out,
            origin_ll,
            dest_ll,
            graph_cfg=graph_cfg,
            snap_cfg=snap_cfg_run,
            repair_cfg=repair_cfg,
            simplify_cfg=simplify_cfg,
            run_cfg=run_cfg,
            G_in=G_run,  # reuse built graph (copied)
        )
        res.multiworld_combo = combo

        final_km: Optional[float] = None
        if res.error is None and res.lengths_km is not None:
            try:
                final_km = float(res.lengths_km.get("final", 0.0))
            except Exception:
                final_km = None

        table.append(
            {
                "combo": combo,
                "error": res.error,
                "final_km": final_km,
                "raw_km": (res.lengths_km or {}).get("raw") if res.lengths_km else None,
                "repaired_km": (res.lengths_km or {}).get("repaired") if res.lengths_km else None,
                "simplified_km": (res.lengths_km or {}).get("simplified") if res.lengths_km else None,
                # repair stats (may be None if repair disabled)
                "colliding_edges": getattr(getattr(res, "repair_stats", None), "colliding_edges", None),
                "repaired_edges": getattr(getattr(res, "repair_stats", None), "repaired_edges", None),
                "failed_edges": getattr(getattr(res, "repair_stats", None), "failed_edges", None),
            }
        )

        if res.error is None and final_km is not None:
            if run_cfg.debug:
                print(f"[pipeline][multiworld][result] combo={combo} final_km={final_km}")
            results_ok.append(res)
        else:
            if run_cfg.debug:
                print(f"[pipeline][multiworld][result] combo={combo} FAIL: {res.error}")

    if not results_ok:
        return RouteResult(origin_ll=origin_ll, dest_ll=dest_ll, error="multiworld_all_failed")

    # rank by final length (lower is better)
    results_ok.sort(key=lambda r: float((r.lengths_km or {}).get("final", 1e30)))

    # pick rank1 = shortest by distance
    best = results_ok[0]
    best.multiworld_rank = 1
    best.multiworld_table = table

    # pick rank2 = first route that is *meaningfully different* from rank1.
    # In practice some combos can snap to the same injected nodes and produce identical polylines.
    # We de-dup by a coarse polyline signature first, and fall back to final_km difference.
    eps_km = 1e-6
    best_final = float((best.lengths_km or {}).get("final", 1e30))

    def _sig(path: Optional[List[LonLat]]) -> Optional[Tuple[Tuple[float, float], ...]]:
        if not path:
            return None
        # round to ~1e-5 deg (~1m at equator) to be robust to float noise
        return tuple((round(float(x), 5), round(float(y), 5)) for x, y in path)

    best_sig = _sig(getattr(best, "path_ll_final", None) or getattr(best, "path_ll_simplified", None))

    runner_up: Optional[RouteResult] = None
    for cand in results_ok[1:]:
        cand_sig = _sig(getattr(cand, "path_ll_final", None) or getattr(cand, "path_ll_simplified", None))
        cand_final = float((cand.lengths_km or {}).get("final", 1e30))
        different_path = (best_sig is not None and cand_sig is not None and cand_sig != best_sig)
        different_len = (abs(cand_final - best_final) > eps_km)
        if different_path or different_len:
            runner_up = cand
            break

    # safety-aware swap: allow switching to runner_up if it is meaningfully safer
    pick_meta: Dict[str, Any] = {
        "picked": "rank1",
        "reason": "shortest_final_km",
    }
    if runner_up is not None:
        d1 = best_final
        d2 = float((runner_up.lengths_km or {}).get("final", 1e30))
        s1 = getattr(best, "repair_stats", None)
        s2 = getattr(runner_up, "repair_stats", None)
        f1 = getattr(s1, "failed_edges", None)
        f2 = getattr(s2, "failed_edges", None)

        # distance gate: only accept extra distance up to min(pct, K)
        pct = float(getattr(run_cfg, "select_pct", 0.10))
        K = float(getattr(run_cfg, "select_k_km", 150.0))
        dist_gate = min(d1 * (1.0 + pct), d1 + K)
        within_gate = (d2 <= dist_gate)

        # safety gate: require ΔF >= threshold (missing -> no swap)
        dF_thr = int(getattr(run_cfg, "select_delta_failed_edges", 2))
        delta_f = None
        if isinstance(f1, int) and isinstance(f2, int):
            delta_f = int(f1) - int(f2)

        pick_meta.update(
            {
                "rank1_combo": best.multiworld_combo,
                "rank2_combo": runner_up.multiworld_combo,
                "rank1_final_km": d1,
                "rank2_final_km": d2,
                "rank1_failed_edges": f1,
                "rank2_failed_edges": f2,
                "delta_failed_edges": delta_f,
                "delta_failed_thr": dF_thr,
                "pct": pct,
                "K_km": K,
                "dist_gate_km": dist_gate,
                "within_gate": within_gate,
            }
        )

        if within_gate and (delta_f is not None) and (delta_f >= dF_thr):
            # swap: choose safer (runner_up) as rank1
            best, runner_up = runner_up, best
            best.multiworld_rank = 1
            pick_meta["picked"] = "rank2"
            pick_meta["reason"] = "safer_delta_failed_edges_within_distance_gate"
        else:
            pick_meta["picked"] = "rank1"
            pick_meta["reason"] = "safety_or_distance_gate_not_met"

    # attach runner-up (rank 2) for debugging / visualization
    if runner_up is not None:
        runner_up.multiworld_rank = 2
        runner_up.multiworld_table = table
        best.multiworld_alternatives = [runner_up]
    else:
        best.multiworld_alternatives = []

    # attach a small hint in snap_debug
    if best.snap_debug is None:
        best.snap_debug = {}
    best.snap_debug["multiworld_selected"] = True
    best.snap_debug["R_NEAR_COAST_KM"] = R_NEAR
    best.snap_debug["S_MAX_SNAP_KM"] = S_MAX
    best.snap_debug["multiworld_combo_rank1"] = best.multiworld_combo
    if best.multiworld_alternatives:
        best.snap_debug["multiworld_combo_rank2"] = best.multiworld_alternatives[0].multiworld_combo
    best.snap_debug["multiworld_pick_meta"] = pick_meta

    if run_cfg.debug:
        print(f"[pipeline][multiworld][pick] {pick_meta}")

    return best



__all__ = [
    "GraphConfig",
    "SnapConfig",
    "SimplifyConfig",
    "RunConfig",
    "RouteResult",
    "run_p2p",
    "run_p2p_multiworld",
]
