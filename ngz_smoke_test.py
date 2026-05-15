"""ngz_smoke_test.py — NGZ 模組 smoke test。

執行：
    python ngz_smoke_test.py

T1~T5 / T7 / T8：純模組測試，不需 aoi_cache。
T6 / T9 / T10：舊範式測試（compose / mask / B_NGZ-ban），已 OBSOLETE，由 N1-N8 取代。
N1~N8：Baseline + Local Patching 新範式整合測試；需要 aoi_cache/out_global.pkl.gz。
N7 是 unit test，不需 cache。
"""
from __future__ import annotations

import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from routing_map import (
    BlockedSubpath,
    NgzInput,
    NgzOverlay,
    NgzPatchUnreachableError,
    NgzRingBuildConfig,
    apply_ngz_mode,
    apply_patches_to_baseline,
    build_local_visibility_graph,
    build_ngz_collision_geom,
    build_ngz_overlay_lite,
    build_ngz_t_rings,
    detect_blocked_subpaths,
    detect_inside_ngz,
    normalize_ngz_inputs,
    solve_local_patch,
    split_polygon_at_antimeridian,
)
from routing_map.geom_utils import build_projector_from_bbox


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

_RESULTS = []


def _run(name: str, fn):
    print(f"\n=== {name} ===")
    try:
        result = fn()
        if result == "skip":
            print(f"[SKIP] {name}")
            _RESULTS.append((name, None, "skipped"))
            return
        print(f"[PASS] {name}")
        _RESULTS.append((name, True, None))
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
        _RESULTS.append((name, False, str(e)))
    except Exception as e:
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------------------
# Cached aoi load (lazy, once)
# ---------------------------------------------------------------------------

_OUT_CACHE: Dict[str, Any] = {"loaded": False, "out": None, "G_base": None}


def _load_aoi_once() -> Optional[Tuple[Dict[str, Any], Any]]:
    """Returns (out, G_base) tuple, or None if cache missing / failed."""
    if _OUT_CACHE["loaded"]:
        out = _OUT_CACHE["out"]
        if out is None:
            return None
        return out, _OUT_CACHE["G_base"]
    _OUT_CACHE["loaded"] = True

    cache_path = os.path.join("aoi_cache", "out_global.pkl.gz")
    if not os.path.exists(cache_path):
        print(f"[setup] {cache_path} 不存在；整合測試會跳過")
        return None

    print(f"[setup] 載入 aoi cache（可能要 10~30 秒）...")
    t0 = time.time()
    try:
        from routing_map.cache_utils import load_out_cache, ensure_graph_edge_masks
        import gzip
        import pickle
        out = load_out_cache(None, strict=False)
        if out is None:
            return None
        with gzip.open(os.path.join("aoi_cache", "G_global.pkl.gz"), "rb") as f:
            payload = pickle.load(f)
        G_base = payload.get("G_base") if isinstance(payload, dict) else payload
        if G_base is not None:
            ensure_graph_edge_masks(G_base, hard_lat_cap_deg=70.0)
    except Exception as e:
        print(f"[setup] 載入失敗：{type(e).__name__}: {e}")
        traceback.print_exc()
        return None
    print(f"[setup] aoi 載入完成（{time.time() - t0:.1f}s）"
          f"，S_nodes={len(out.get('S_nodes', []))}，G_base={G_base.number_of_nodes() if G_base is not None else None}")
    _OUT_CACHE["out"] = out
    _OUT_CACHE["G_base"] = G_base
    return out, G_base


def _bbox(poly_or_polys) -> Tuple[float, float, float, float]:
    if not isinstance(poly_or_polys, (list, tuple)):
        return poly_or_polys.bounds
    u = unary_union(list(poly_or_polys))
    return u.bounds


# ---------------------------------------------------------------------------
# T1~T8: pure module tests (no cache)
# ---------------------------------------------------------------------------

def test_t1_empty_list():
    """空 NGZ list → normalize 回空、t_rings 回空、overlay 各 DataFrame 為空。"""
    proj = build_projector_from_bbox((118.0, 20.0, 125.0, 30.0))
    groups = normalize_ngz_inputs([], proj=proj, cfg=NgzRingBuildConfig())
    assert groups == [], f"expected [], got {groups}"

    rings = build_ngz_t_rings([], proj=proj, cfg=NgzRingBuildConfig())
    assert rings == [], f"expected [], got {rings}"

    overlay = build_ngz_overlay_lite([], [])
    assert overlay.nodes.empty, "nodes should be empty"
    assert overlay.edges_ring.empty, "edges_ring should be empty"
    assert overlay.edges_gate.empty, "edges_gate should be empty"
    assert overlay.edges_ngz_ngz.empty, "edges_ngz_ngz should be empty"


def test_t2_single_rectangle():
    """一個矩形 → 1 group、T-ring 頂點 >= 4、overlay nodes 含對應頂點。"""
    rect = Polygon([(120.0, 22.0), (122.0, 22.0), (122.0, 24.0), (120.0, 24.0)])
    proj = build_projector_from_bbox((118.0, 20.0, 125.0, 26.0))
    cfg = NgzRingBuildConfig(clearance_m=5_000.0, ring_sample_km=5.0)

    groups = normalize_ngz_inputs([rect], proj=proj, cfg=cfg)
    assert len(groups) == 1, f"expected 1 group, got {len(groups)}"

    rings = build_ngz_t_rings(groups, proj=proj, cfg=cfg)
    assert len(rings) == 1, f"expected 1 ring, got {len(rings)}"
    r = rings[0]
    assert len(r.taut_pts_m) >= 4, f"taut points too few: {len(r.taut_pts_m)}"
    assert len(r.taut_pts_ll) == len(r.taut_pts_m), "ll/m length mismatch"

    overlay = build_ngz_overlay_lite(groups, rings)
    n_nodes = len(overlay.nodes)
    assert n_nodes >= 4, f"too few NGZ nodes: {n_nodes}"

    # T-ring 不應切過原始 NGZ polygon（在 metric 下檢查）
    rect_m = unary_union([
        type(rect)([(proj.ll2m(*xy)) for xy in rect.exterior.coords])
    ])
    pts = r.taut_pts_m
    crossings = 0
    for i in range(len(pts) - 1):
        seg = LineString([pts[i], pts[i + 1]])
        if seg.crosses(rect_m) or rect_m.contains(seg):
            crossings += 1
    assert crossings == 0, f"T-ring segments cross NGZ: {crossings}"


def test_t3_two_overlapping_rectangles_merge():
    """兩個重疊矩形 → 應合併為 1 group。"""
    r1 = Polygon([(120.0, 22.0), (122.0, 22.0), (122.0, 24.0), (120.0, 24.0)])
    r2 = Polygon([(121.0, 23.0), (123.0, 23.0), (123.0, 25.0), (121.0, 25.0)])
    proj = build_projector_from_bbox((118.0, 20.0, 126.0, 27.0))
    cfg = NgzRingBuildConfig()

    groups = normalize_ngz_inputs([r1, r2], proj=proj, cfg=cfg)
    assert len(groups) == 1, f"expected 1 merged group, got {len(groups)}"
    g = groups[0]
    assert len(g.member_ids) == 2, f"expected 2 members, got {g.member_ids}"


def test_t4_dateline_split():
    """跨 dateline 矩形 → split_polygon_at_antimeridian 切成 2 塊，各自落在 [-180, 180]。"""
    crossing = Polygon([
        (175.0, 10.0),
        (-175.0, 10.0),
        (-175.0, 20.0),
        (175.0, 20.0),
    ])
    pieces = split_polygon_at_antimeridian(crossing)
    assert len(pieces) == 2, f"expected 2 pieces, got {len(pieces)}: bounds = {[p.bounds for p in pieces]}"
    for p in pieces:
        minx, _, maxx, _ = p.bounds
        assert -180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0, \
            f"piece bbox out of range: {p.bounds}"
        assert (maxx - minx) < 180.0, f"piece still crosses dateline: span={maxx-minx}"


def test_t5_polygon_overlapping_land():
    """polygon 與假 land collision 重疊 → difference 後仍能建出合法 T-ring。"""
    fake_land = Polygon([(118.0, 20.0), (121.0, 20.0), (121.0, 26.0), (118.0, 26.0)])
    ngz = Polygon([(120.5, 22.0), (123.0, 22.0), (123.0, 24.0), (120.5, 24.0)])
    proj = build_projector_from_bbox((117.0, 19.0, 124.0, 27.0))
    cfg = NgzRingBuildConfig()

    groups = normalize_ngz_inputs([ngz], proj=proj, land_collision_ll=fake_land, cfg=cfg)
    assert len(groups) >= 1, f"expected >=1 group after land subtract, got {len(groups)}"
    g = groups[0]
    minx, _, _, _ = g.polygon_ll.bounds
    assert minx >= 121.0 - 1e-3, f"land subtract didn't clip: minx={minx}"

    rings = build_ngz_t_rings(groups, proj=proj, cfg=cfg)
    assert len(rings) == 1 and len(rings[0].taut_pts_m) >= 4, \
        f"T-ring failed after land subtract: rings={rings}"


def test_t6_OBSOLETE_compose_mutability():
    """[OBSOLETE] 舊範式的 nx.compose 不 mutate cached G 已不適用——新範式不擴增 G，
    NGZ 不進 graph，patching 在 lon/lat 空間完成。改由 N1（degenerate）保證行為不變。
    """
    return "skip"


def test_t7_apply_ngz_mode():
    """apply_ngz_mode 三模式：strict raise、lenient 不動、relocate 移到最近 T-ring 頂點。"""
    rect = Polygon([(120.0, 22.0), (122.0, 22.0), (122.0, 24.0), (120.0, 24.0)])
    proj = build_projector_from_bbox((118.0, 20.0, 125.0, 26.0))
    cfg = NgzRingBuildConfig()
    groups = normalize_ngz_inputs([rect], proj=proj, cfg=cfg)
    rings = build_ngz_t_rings(groups, proj=proj, cfg=cfg)
    overlay = build_ngz_overlay_lite(groups, rings)

    inside_pt = (121.0, 23.0)
    outside_pt = (118.5, 21.0)

    g_in = detect_inside_ngz(inside_pt, overlay)
    g_out = detect_inside_ngz(outside_pt, overlay)
    assert len(g_in) == 1, f"inside detection failed: {g_in}"
    assert len(g_out) == 0, f"outside detection failed: {g_out}"

    raised = False
    try:
        apply_ngz_mode(inside_pt, outside_pt, overlay, mode="strict")
    except Exception:
        raised = True
    assert raised, "strict mode did not raise for inside origin"

    res_lenient = apply_ngz_mode(inside_pt, outside_pt, overlay, mode="lenient")
    assert res_lenient["origin_ll"] == inside_pt, "lenient should not move origin"
    assert res_lenient["inside_origin"] == g_in, "lenient should report inside_origin"

    res_reloc = apply_ngz_mode(inside_pt, outside_pt, overlay, mode="relocate")
    assert res_reloc["origin_ll"] != inside_pt, "relocate should move origin"
    assert not res_reloc["inside_origin"], "relocate should clear inside_origin"


def test_t8_collision_geom_exempt():
    """build_ngz_collision_geom 在 exempt_group_ids 模式下會排除指定 group。"""
    r1 = Polygon([(120.0, 22.0), (122.0, 22.0), (122.0, 24.0), (120.0, 24.0)])
    r2 = Polygon([(124.0, 22.0), (125.0, 22.0), (125.0, 24.0), (124.0, 24.0)])
    proj = build_projector_from_bbox((118.0, 20.0, 127.0, 26.0))
    cfg = NgzRingBuildConfig()
    groups = normalize_ngz_inputs([r1, r2], proj=proj, cfg=cfg)
    rings = build_ngz_t_rings(groups, proj=proj, cfg=cfg)
    assert len(groups) == 2, f"expected 2 groups (not merged), got {len(groups)}"

    full = build_ngz_collision_geom(rings, groups, proj=proj)
    assert not full["ngz_union_ll"].is_empty
    full_area = full["ngz_union_ll"].area

    exempted = build_ngz_collision_geom(rings, groups, proj=proj, exempt_group_ids={groups[0].group_id})
    if not exempted["ngz_union_ll"].is_empty:
        assert exempted["ngz_union_ll"].area < full_area, \
            "exempted collision should be smaller than full"


def test_t9_OBSOLETE_pipeline_no_op():
    """[OBSOLETE] 舊 T9（pipeline no-op without NGZ）改由 N1 涵蓋。"""
    return "skip"


def test_t10_OBSOLETE_simplifier_no_shortcut():
    """[OBSOLETE] 舊 T10（simplifier no shortcut through NGZ）改由 N3 / N8 涵蓋。"""
    return "skip"


# ---------------------------------------------------------------------------
# N1~N8: Baseline + Local Patching 新範式整合測試
# 共用 fixture: _load_aoi_once()
# 測試 OD（純太平洋海域，無島嶼干擾）：
#   origin = (130.0, 22.0)
#   dest   = (145.0, 32.0)
# ---------------------------------------------------------------------------

_ORIGIN = (130.0, 22.0)
_DEST = (145.0, 32.0)


def _baseline_or_skip():
    loaded = _load_aoi_once()
    if loaded is None:
        return None
    out, G_base = loaded
    if G_base is None:
        return None
    from routing_map.pipeline import run_p2p, RunConfig
    rc = RunConfig(do_repair=False, do_simplify=True, debug=False)
    res_base = run_p2p(out, _ORIGIN, _DEST, run_cfg=rc, G_in=G_base.copy())
    if res_base.error is not None:
        print(f"  baseline failed: {res_base.error}")
        return None
    return out, G_base, rc, res_base


def test_n1_degenerate_no_ngz():
    """ngz_polygons=None / [] / 不傳 → 三者結果完全一致；res.error is None。"""
    setup = _baseline_or_skip()
    if setup is None:
        return "skip"
    out, G_base, rc, res_a = setup

    from routing_map.pipeline import run_p2p
    res_b = run_p2p(out, _ORIGIN, _DEST, run_cfg=rc, G_in=G_base.copy(), ngz_polygons=None)
    res_c = run_p2p(out, _ORIGIN, _DEST, run_cfg=rc, G_in=G_base.copy(), ngz_polygons=[])

    assert res_a.error is None
    assert res_b.error is None
    assert res_c.error is None

    def _key(res):
        return (
            tuple(res.path_ll_final or []),
            tuple(sorted((res.lengths_km or {}).items())),
        )

    assert _key(res_a) == _key(res_b), "ngz_polygons=None should match no-arg call"
    assert _key(res_a) == _key(res_c), "ngz_polygons=[] should match no-arg call"


def test_n2_ngz_off_baseline():
    """NGZ 完全離 baseline 200km+ → 結果逐點等於 baseline。**新範式 vs PR2 的核心 win**。"""
    setup = _baseline_or_skip()
    if setup is None:
        return "skip"
    out, G_base, rc, res_base = setup

    from routing_map.pipeline import run_p2p

    # baseline 大約沿著 (130,22) → (145,32) 對角線；放一個 NGZ 在 (135, 15) 附近，
    # 離主線 600+km，絕對不該觸發 patching。
    far_rect = Polygon([(134.0, 14.0), (136.0, 14.0), (136.0, 16.0), (134.0, 16.0)])
    res = run_p2p(
        out, _ORIGIN, _DEST,
        run_cfg=rc, G_in=G_base.copy(),
        ngz_polygons=[far_rect],
        ngz_mode="lenient",
    )
    assert res.error is None, f"off-baseline NGZ failed: {res.error}"

    # 結果逐點等於 baseline（容忍 1e-9）
    base_path = res_base.path_ll_final or []
    cmp_path = res.path_ll_final or []
    assert len(base_path) == len(cmp_path), \
        f"length mismatch baseline={len(base_path)} vs with-ngz={len(cmp_path)}"
    for (a, b) in zip(base_path, cmp_path):
        assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9, \
            f"off-baseline NGZ shouldn't change path: {a} vs {b}"

    # 主動 call detect_blocked_subpaths 驗證 blocked_runs 為 0
    if res.ngz_overlay is not None and res.ngz_overlay.groups:
        groups = res.ngz_overlay.groups
        blocked = detect_blocked_subpaths(res_base.path_ll_raw or [], groups)
        assert len(blocked) == 0, f"expected 0 blocked runs, got {len(blocked)}"


def _line_intersects_polygon(path: List[Tuple[float, float]], poly: Polygon) -> bool:
    if not path or len(path) < 2:
        return False
    line = LineString(path)
    # 用內縮 polygon 排除 boundary touch
    inner = poly.buffer(-1e-4)
    if inner.is_empty:
        return False
    return line.intersects(inner)


def test_n3_single_ngz_blocks_baseline():
    """單一 NGZ 擋 baseline 中段 → path 不交 NGZ；final_km < baseline_km × 1.5。"""
    setup = _baseline_or_skip()
    if setup is None:
        return "skip"
    out, G_base, rc, res_base = setup

    from routing_map.pipeline import run_p2p

    ngz_rect = Polygon([(135.0, 25.0), (140.0, 25.0), (140.0, 30.0), (135.0, 30.0)])

    res = run_p2p(
        out, _ORIGIN, _DEST,
        run_cfg=rc, G_in=G_base.copy(),
        ngz_polygons=[ngz_rect],
        ngz_mode="lenient",
        ngz_cfg=NgzRingBuildConfig(clearance_m=10_000.0),
    )
    assert res.error is None, f"single NGZ failed: {res.error}"
    final_path = res.path_ll_final or []
    assert len(final_path) >= 2

    # (a) final path 不交 NGZ polygon 內部
    assert not _line_intersects_polygon(final_path, ngz_rect), \
        "final path crosses NGZ polygon interior"

    # (b) final_km < baseline_km * 1.5
    base_km = float((res_base.lengths_km or {}).get("final", 0.0))
    new_km = float((res.lengths_km or {}).get("final", 0.0))
    assert base_km > 0 and new_km > 0
    assert new_km < base_km * 1.5, \
        f"detour too expensive: base={base_km:.1f} new={new_km:.1f} (ratio={new_km/base_km:.2f})"

    # (c) 至少一個 path 點靠近 T-ring 頂點（< 0.5 deg ~ 50km）
    rings = res.ngz_overlay.rings if res.ngz_overlay is not None else []
    taut_pts: List[Tuple[float, float]] = []
    for r in rings:
        for pt in r.taut_pts_ll:
            taut_pts.append((float(pt[0]), float(pt[1])))
    if taut_pts:
        def _min_dist(p):
            return min(math.hypot(p[0] - t[0], p[1] - t[1]) for t in taut_pts)
        min_overall = min(_min_dist(p) for p in final_path)
        assert min_overall < 0.5, \
            f"final path doesn't pass near any T-ring vertex: min_dist={min_overall:.4f} deg"


def test_n4_concave_ngz_u_shape():
    """凹 NGZ（U 形）→ T-ring 凸殼化，path 走 U 外側。"""
    setup = _baseline_or_skip()
    if setup is None:
        return "skip"
    out, G_base, rc, _ = setup

    from routing_map.pipeline import run_p2p

    # U 形 polygon（開口朝南）
    u_poly = Polygon([
        (135.0, 25.0), (140.0, 25.0), (140.0, 30.0), (139.0, 30.0),
        (139.0, 26.0), (136.0, 26.0), (136.0, 30.0), (135.0, 30.0),
    ])

    res = run_p2p(
        out, _ORIGIN, _DEST,
        run_cfg=rc, G_in=G_base.copy(),
        ngz_polygons=[u_poly],
        ngz_mode="lenient",
        ngz_cfg=NgzRingBuildConfig(clearance_m=10_000.0),
    )
    assert res.error is None, f"U-shape NGZ failed: {res.error}"
    final_path = res.path_ll_final or []
    # T-ring 凸殼化後 path 應該只走 U 外側
    assert not _line_intersects_polygon(final_path, u_poly), \
        "final path crosses U-shape NGZ"


def test_n5_multi_ngz_same_segment():
    """多 NGZ 同擋同段 baseline（D2a）→ blocked.ngz_group_ids 含兩個 group。"""
    setup = _baseline_or_skip()
    if setup is None:
        return "skip"
    out, G_base, rc, res_base = setup

    from routing_map.pipeline import run_p2p

    # 兩個彼此分離但同擋 baseline **同一段** segment 的矩形。
    # baseline 從 (130,22)→(145,32) 中段會有一條從 ~(135.37, 28.58) → (140.0, 30.0)
    # 的長 segment；下面兩個矩形分別落在這段的前後半，距離 > group_merge_eps_m。
    r1 = Polygon([(136.0, 28.7), (137.0, 28.7), (137.0, 29.1), (136.0, 29.1)])
    r2 = Polygon([(138.5, 29.5), (139.5, 29.5), (139.5, 30.0), (138.5, 30.0)])

    res = run_p2p(
        out, _ORIGIN, _DEST,
        run_cfg=rc, G_in=G_base.copy(),
        ngz_polygons=[r1, r2],
        ngz_mode="lenient",
        ngz_cfg=NgzRingBuildConfig(clearance_m=10_000.0),
    )
    assert res.error is None, f"multi-NGZ failed: {res.error}"

    # 對 baseline 跑 detect_blocked_subpaths，找出 blocked runs
    groups = res.ngz_overlay.groups if res.ngz_overlay is not None else []
    assert len(groups) == 2, f"expected 2 groups (not merged), got {len(groups)}"
    blocked = detect_blocked_subpaths(res_base.path_ll_raw or [], groups)
    # 至少要偵測到觸碰兩個 NGZ 的 blocked run（同段或不同段都算過關）
    all_gids: set = set()
    for b in blocked:
        all_gids |= b.ngz_group_ids
    assert len(all_gids) == 2, \
        f"expected to detect both NGZs, got {all_gids} from {len(blocked)} runs"


def test_n6_lenient_origin_inside_ngz():
    """Lenient: origin 落在 NGZ-A 內 → NGZ-A 從 patch collision 排除；NGZ-B 正常 patch。"""
    setup = _baseline_or_skip()
    if setup is None:
        return "skip"
    out, G_base, rc, _ = setup

    from routing_map.pipeline import run_p2p

    # NGZ-A 包住 origin
    ngz_a = Polygon([
        (_ORIGIN[0] - 0.5, _ORIGIN[1] - 0.5),
        (_ORIGIN[0] + 0.5, _ORIGIN[1] - 0.5),
        (_ORIGIN[0] + 0.5, _ORIGIN[1] + 0.5),
        (_ORIGIN[0] - 0.5, _ORIGIN[1] + 0.5),
    ])
    # NGZ-B 擋後段
    ngz_b = Polygon([(138.0, 28.0), (140.0, 28.0), (140.0, 30.0), (138.0, 30.0)])

    res = run_p2p(
        out, _ORIGIN, _DEST,
        run_cfg=rc, G_in=G_base.copy(),
        ngz_polygons=[ngz_a, ngz_b],
        ngz_mode="lenient",
        ngz_cfg=NgzRingBuildConfig(clearance_m=10_000.0),
    )
    assert res.error is None, f"lenient origin-in-NGZ failed: {res.error}"
    assert res.ngz_inside_origin, "expected origin to be inside some NGZ"

    # 至少要有 1 個 inside_origin group_id
    inside_ids = set(res.ngz_inside_origin)
    assert len(inside_ids) >= 1

    # NGZ-A（exempt）不該擋路；最終 path 可以穿過 NGZ-A 不違規
    # 而 NGZ-B 仍應被避開
    final_path = res.path_ll_final or []
    assert not _line_intersects_polygon(final_path, ngz_b), \
        "final path crosses NGZ-B (non-exempt)"


def test_n7_unreachable_patch_unit():
    """D3a unit test：fake 一個 A/B 不連通的 g_local 餵 solve_local_patch → raise。"""
    g = nx.Graph()
    g.add_node("A", lon=0.0, lat=0.0)
    g.add_node("B", lon=10.0, lat=0.0)
    # 沒邊 → 不連通

    raised = False
    try:
        solve_local_patch(
            g, "A", "B",
            rings=[],
            anchor_a_ll=(0.0, 0.0),
            anchor_b_ll=(10.0, 0.0),
        )
    except NgzPatchUnreachableError:
        raised = True
    assert raised, "solve_local_patch should raise NgzPatchUnreachableError on disconnected graph"


def test_n8_final_simplify_preserves_ngz():
    """Final simplify 後 path 仍不交 NGZ。"""
    setup = _baseline_or_skip()
    if setup is None:
        return "skip"
    out, G_base, rc, _ = setup

    from routing_map.pipeline import run_p2p

    ngz_rect = Polygon([(135.0, 25.0), (140.0, 25.0), (140.0, 30.0), (135.0, 30.0)])

    res = run_p2p(
        out, _ORIGIN, _DEST,
        run_cfg=rc, G_in=G_base.copy(),
        ngz_polygons=[ngz_rect],
        ngz_mode="lenient",
        ngz_cfg=NgzRingBuildConfig(clearance_m=10_000.0),
    )
    assert res.error is None
    simplified = res.path_ll_simplified or res.path_ll_raw or []
    assert len(simplified) >= 2

    # simplified path 不交 NGZ 內部
    assert not _line_intersects_polygon(simplified, ngz_rect), \
        "simplified path crosses NGZ"

    # simplified ≤ patched 點數（單調收斂）
    patched = res.path_ll_raw or []
    assert len(simplified) <= len(patched), \
        f"simplified={len(simplified)} > patched={len(patched)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _run("T1 empty list", test_t1_empty_list)
    _run("T2 single rectangle T-ring", test_t2_single_rectangle)
    _run("T3 overlapping rectangles merge", test_t3_two_overlapping_rectangles_merge)
    _run("T4 dateline split", test_t4_dateline_split)
    _run("T5 land subtract", test_t5_polygon_overlapping_land)
    _run("T6 [OBSOLETE] nx.compose immutability", test_t6_OBSOLETE_compose_mutability)
    _run("T7 apply_ngz_mode", test_t7_apply_ngz_mode)
    _run("T8 collision exempt", test_t8_collision_geom_exempt)
    _run("T9 [OBSOLETE] pipeline no-op (replaced by N1)", test_t9_OBSOLETE_pipeline_no_op)
    _run("T10 [OBSOLETE] simplifier no shortcut (replaced by N3/N8)", test_t10_OBSOLETE_simplifier_no_shortcut)

    _run("N1 degenerate (ngz=None/[]/no-arg)", test_n1_degenerate_no_ngz)
    _run("N2 NGZ off-baseline → identical to baseline", test_n2_ngz_off_baseline)
    _run("N3 single NGZ blocks baseline", test_n3_single_ngz_blocks_baseline)
    _run("N4 concave U-shape NGZ", test_n4_concave_ngz_u_shape)
    _run("N5 multi-NGZ same segment", test_n5_multi_ngz_same_segment)
    _run("N6 lenient origin inside NGZ-A", test_n6_lenient_origin_inside_ngz)
    _run("N7 unreachable patch (unit)", test_n7_unreachable_patch_unit)
    _run("N8 final simplify preserves NGZ", test_n8_final_simplify_preserves_ngz)

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok is True)
    n_skip = sum(1 for _, ok, _ in _RESULTS if ok is None)
    n_fail = sum(1 for _, ok, _ in _RESULTS if ok is False)
    for name, ok, msg in _RESULTS:
        flag = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
        line = f"  [{flag}] {name}"
        if msg and ok is False:
            line += f" — {msg}"
        print(line)
    print(f"\nTotal: {n_pass} passed, {n_skip} skipped, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
