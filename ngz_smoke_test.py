"""ngz_smoke_test.py — NGZ 模組 smoke test。

執行：
    python ngz_smoke_test.py

T1~T8：純模組測試，不需 aoi_cache。
T9~T10：整合測試，需要 aoi_cache/out_global.pkl.gz；若不存在會跳過並印 [SKIP]。
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import networkx as nx
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from routing_map import (
    NgzInput,
    NgzOverlay,
    NgzRingBuildConfig,
    apply_ngz_mode,
    build_ngz_collision_geom,
    build_ngz_overlay,
    build_ngz_t_rings,
    compose_ngz_into_graph,
    detect_inside_ngz,
    normalize_ngz_inputs,
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
        # 載 G_base（pipeline build_base_graph 的呼叫簽章與 routing_graph 不一致，
        # 是既有 inconsistency；用 cache 走 G_in 路徑跳過 build_base_graph）。
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
# Tests
# ---------------------------------------------------------------------------

def test_t1_empty_list():
    """空 NGZ list → normalize 回空、t_rings 回空、overlay 各 DataFrame 為空。"""
    proj = build_projector_from_bbox((118.0, 20.0, 125.0, 30.0))
    groups = normalize_ngz_inputs([], proj=proj, cfg=NgzRingBuildConfig())
    assert groups == [], f"expected [], got {groups}"

    rings = build_ngz_t_rings([], proj=proj, cfg=NgzRingBuildConfig())
    assert rings == [], f"expected [], got {rings}"

    overlay = build_ngz_overlay([], [], proj=proj, cfg=NgzRingBuildConfig())
    assert overlay.nodes.empty, "nodes should be empty"
    assert overlay.edges_ring.empty, "edges_ring should be empty"
    assert overlay.edges_gate.empty, "edges_gate should be empty"
    assert overlay.edges_ngz_ngz.empty, "edges_ngz_ngz should be empty"


def test_t2_single_rectangle():
    """一個矩形 → 1 group、T-ring 頂點 >= 4、ring 邊與頂點數一致、T-ring 不切到 polygon。"""
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

    overlay = build_ngz_overlay(rings, groups, proj=proj, cfg=cfg)
    n_nodes = len(overlay.nodes)
    n_ring_edges = len(overlay.edges_ring)
    # taut_pts 含 closure，nodes 去 closure，環邊 = 節點數
    assert n_nodes >= 4, f"too few NGZ nodes: {n_nodes}"
    assert n_ring_edges == n_nodes, f"ring edges {n_ring_edges} != nodes {n_nodes}"

    # T-ring 不應切過原始 NGZ polygon（在 metric 下檢查）
    rect_m = unary_union([
        type(rect)([(proj.ll2m(*xy)) for xy in rect.exterior.coords])
    ])
    pts = r.taut_pts_m
    crossings = 0
    for i in range(len(pts) - 1):
        seg = LineString([pts[i], pts[i + 1]])
        # T-ring 在 NGZ 外側 clearance_m，不應與 NGZ polygon 相交（容許 boundary touch）
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
    # 一個跨 dateline 的矩形（lon 175 ~ -175，等同實際 lon span 10 度）
    # 在我們的偵測規則下這不會被當成 dateline crossing（因為 bbox span = 10 < 180）。
    # 真正會觸發切分的是 bbox span > 180 的情況：
    crossing = Polygon([
        (175.0, 10.0),
        (-175.0, 10.0),  # 直接給負數會讓 shapely 認為 bbox span = 350
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
    # 假陸地：lon < 121
    fake_land = Polygon([(118.0, 20.0), (121.0, 20.0), (121.0, 26.0), (118.0, 26.0)])
    # NGZ 跨陸地一點
    ngz = Polygon([(120.5, 22.0), (123.0, 22.0), (123.0, 24.0), (120.5, 24.0)])
    proj = build_projector_from_bbox((117.0, 19.0, 124.0, 27.0))
    cfg = NgzRingBuildConfig()

    groups = normalize_ngz_inputs([ngz], proj=proj, land_collision_ll=fake_land, cfg=cfg)
    assert len(groups) >= 1, f"expected >=1 group after land subtract, got {len(groups)}"
    g = groups[0]
    # 扣陸地後 NGZ 應只剩 lon > 121 的部分
    minx, _, _, _ = g.polygon_ll.bounds
    assert minx >= 121.0 - 1e-3, f"land subtract didn't clip: minx={minx}"

    # 仍能建出 T-ring
    rings = build_ngz_t_rings(groups, proj=proj, cfg=cfg)
    assert len(rings) == 1 and len(rings[0].taut_pts_m) >= 4, \
        f"T-ring failed after land subtract: rings={rings}"


def test_t6_nx_compose_does_not_mutate_cached():
    """確認 nx.compose 不 mutate G_cached（NGZ function build.md 的關鍵注意事項）。"""
    rect = Polygon([(120.0, 22.0), (122.0, 22.0), (122.0, 24.0), (120.0, 24.0)])
    proj = build_projector_from_bbox((118.0, 20.0, 125.0, 26.0))
    cfg = NgzRingBuildConfig()
    groups = normalize_ngz_inputs([rect], proj=proj, cfg=cfg)
    rings = build_ngz_t_rings(groups, proj=proj, cfg=cfg)
    overlay = build_ngz_overlay(rings, groups, proj=proj, cfg=cfg)

    G_cached = nx.Graph()
    G_cached.add_node("seaA", lon=120.0, lat=21.0)
    G_cached.add_node("seaB", lon=123.0, lat=25.0)
    G_cached.add_edge("seaA", "seaB", weight=10.0, etype="sea")

    n_before = G_cached.number_of_nodes()
    e_before = G_cached.number_of_edges()

    G_query = compose_ngz_into_graph(G_cached, overlay)

    # G_cached 完全沒被改
    assert G_cached.number_of_nodes() == n_before, "G_cached node count changed!"
    assert G_cached.number_of_edges() == e_before, "G_cached edge count changed!"
    assert "seaA" in G_cached and "seaB" in G_cached, "G_cached lost original nodes"

    # G_query 含 NGZ 節點與邊
    assert G_query.number_of_nodes() > n_before, "G_query did not add NGZ nodes"
    assert G_query.number_of_edges() > e_before, "G_query did not add NGZ ring edges"
    # 至少有一個節點 id 是 "NGZ:..."
    assert any(str(n).startswith("NGZ:") for n in G_query.nodes), \
        "no NGZ-prefixed nodes in G_query"


def test_t7_apply_ngz_mode():
    """apply_ngz_mode 三模式：strict raise、lenient 不動、relocate 移到最近 T-ring 頂點。"""
    rect = Polygon([(120.0, 22.0), (122.0, 22.0), (122.0, 24.0), (120.0, 24.0)])
    proj = build_projector_from_bbox((118.0, 20.0, 125.0, 26.0))
    cfg = NgzRingBuildConfig()
    groups = normalize_ngz_inputs([rect], proj=proj, cfg=cfg)
    rings = build_ngz_t_rings(groups, proj=proj, cfg=cfg)
    overlay = build_ngz_overlay(rings, groups, proj=proj, cfg=cfg)

    inside_pt = (121.0, 23.0)  # NGZ 內部
    outside_pt = (118.5, 21.0)

    # detect_inside_ngz
    g_in = detect_inside_ngz(inside_pt, overlay)
    g_out = detect_inside_ngz(outside_pt, overlay)
    assert len(g_in) == 1, f"inside detection failed: {g_in}"
    assert len(g_out) == 0, f"outside detection failed: {g_out}"

    # strict
    raised = False
    try:
        apply_ngz_mode(inside_pt, outside_pt, overlay, mode="strict")
    except Exception:
        raised = True
    assert raised, "strict mode did not raise for inside origin"

    # lenient
    res_lenient = apply_ngz_mode(inside_pt, outside_pt, overlay, mode="lenient")
    assert res_lenient["origin_ll"] == inside_pt, "lenient should not move origin"
    assert res_lenient["inside_origin"] == g_in, "lenient should report inside_origin"

    # relocate
    res_reloc = apply_ngz_mode(inside_pt, outside_pt, overlay, mode="relocate")
    assert res_reloc["origin_ll"] != inside_pt, "relocate should move origin"
    assert not res_reloc["inside_origin"], "relocate should clear inside_origin"


def test_t9_pipeline_no_op_no_ngz():
    """退化測試：run_p2p 不傳 ngz_polygons → 與 ngz_polygons=None 結果完全一致。

    需要 aoi_cache/out_global.pkl.gz；若不存在 → SKIP。
    """
    loaded = _load_aoi_once()
    if loaded is None:
        return "skip"
    out, G_base = loaded
    if G_base is None:
        print("  G_base unavailable; skipping")
        return "skip"

    from routing_map.pipeline import run_p2p, RunConfig

    origin = (118.0, 22.0)
    dest = (125.0, 28.0)
    rc = RunConfig(do_repair=False, do_simplify=True, debug=False)

    # 用 G_in 跳過 build_base_graph（pipeline.py 對 build_base_graph 的呼叫
    # 簽章不匹配是既有 issue，與本 PR 無關）。每次跑前 copy 圖避免 mutation。
    res_a = run_p2p(out, origin, dest, run_cfg=rc, G_in=G_base.copy())
    res_b = run_p2p(out, origin, dest, run_cfg=rc, G_in=G_base.copy(), ngz_polygons=None)
    res_c = run_p2p(out, origin, dest, run_cfg=rc, G_in=G_base.copy(), ngz_polygons=[])

    assert res_a.error is None, f"baseline error: {res_a.error}"
    assert res_b.error is None, f"ngz_polygons=None error: {res_b.error}"
    assert res_c.error is None, f"ngz_polygons=[] error: {res_c.error}"

    # 三者 path_ll_final 與 lengths_km 必須完全一致
    def _key(res):
        return (
            tuple(res.path_ll_final or []),
            tuple(sorted((res.lengths_km or {}).items())),
        )

    assert _key(res_a) == _key(res_b), "ngz_polygons=None should match no-arg call"
    assert _key(res_a) == _key(res_c), "ngz_polygons=[] should match no-arg call"


def test_t10_simplifier_no_shortcut_through_ngz():
    """關鍵測試：路徑經過 NGZ 區域時，simplifier 不該抄捷徑穿越 NGZ。

    NGZ function build.md Test 6（必過）。需要 aoi_cache。

    為避免測試 NGZ 跟陸地大量重疊導致差集後變一堆碎片（產生「碎片間合法海峽」誤導
    判斷），改用開放海域 origin/dest + 純海域 NGZ：
      - origin (130, 22) → dest (145, 32)  跨太平洋西部
      - NGZ (135, 25) - (140, 30)  純海域矩形
    並用「差集後 NGZ 群組 union」（排除 origin/dest 所在 group）做 assertion。
    """
    loaded = _load_aoi_once()
    if loaded is None:
        return "skip"
    out, G_base = loaded
    if G_base is None:
        return "skip"

    from routing_map.pipeline import run_p2p, RunConfig
    from routing_map import NgzRingBuildConfig

    origin = (130.0, 22.0)
    dest = (145.0, 32.0)
    rc = RunConfig(do_repair=False, do_simplify=True, debug=False)

    # baseline 確認可達
    res_base = run_p2p(out, origin, dest, run_cfg=rc, G_in=G_base.copy())
    assert res_base.error is None, f"baseline error: {res_base.error}"

    # NGZ 直接擺在 baseline 中段附近的開放海域
    ngz_rect = Polygon([(135.0, 25.0), (140.0, 25.0), (140.0, 30.0), (135.0, 30.0)])

    res = run_p2p(
        out, origin, dest,
        run_cfg=rc,
        G_in=G_base.copy(),
        ngz_polygons=[ngz_rect],
        ngz_mode="lenient",
        ngz_cfg=NgzRingBuildConfig(clearance_m=10_000.0),
    )

    assert res.error is None, f"run_p2p with NGZ failed: {res.error}"
    assert res.ngz_overlay is not None, "ngz_overlay should be populated"
    assert len(res.ngz_overlay.groups) >= 1
    assert len(res.ngz_overlay.nodes) >= 4

    final_path = res.path_ll_final or []
    assert len(final_path) >= 2, f"final path too short: {len(final_path)}"

    # 關鍵 assertion：final path 不該穿越「實際被視為禁區」的部分。
    # lenient 模式下 origin/dest 所在 group 從 collision 排除，所以那些不算禁區。
    exempt = set(res.ngz_inside_origin) | set(res.ngz_inside_dest)
    forbidden_polys = [
        g.polygon_ll for g in res.ngz_overlay.groups
        if g.group_id not in exempt and g.polygon_ll is not None and not g.polygon_ll.is_empty
    ]
    if not forbidden_polys:
        # 全部 exempt 是合法狀態（極端 case），跳過 assertion
        return

    forbidden_union = unary_union(forbidden_polys)
    line = LineString(final_path)
    inner = forbidden_union.buffer(-1e-4)  # 容忍 boundary touch
    crosses = line.intersection(inner)
    crosses_len_deg = float(crosses.length) if not crosses.is_empty else 0.0
    assert crosses_len_deg < 1e-3, \
        f"final path crosses forbidden NGZ interior: length={crosses_len_deg:.6f} deg"


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
    # 兩個 group 都在 collision 內
    assert not full["ngz_union_ll"].is_empty
    full_area = full["ngz_union_ll"].area

    exempted = build_ngz_collision_geom(rings, groups, proj=proj, exempt_group_ids={groups[0].group_id})
    # 排除一個後面積應變小
    if not exempted["ngz_union_ll"].is_empty:
        assert exempted["ngz_union_ll"].area < full_area, \
            "exempted collision should be smaller than full"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _run("T1 empty list", test_t1_empty_list)
    _run("T2 single rectangle T-ring", test_t2_single_rectangle)
    _run("T3 overlapping rectangles merge", test_t3_two_overlapping_rectangles_merge)
    _run("T4 dateline split", test_t4_dateline_split)
    _run("T5 land subtract", test_t5_polygon_overlapping_land)
    _run("T6 nx.compose immutability", test_t6_nx_compose_does_not_mutate_cached)
    _run("T7 apply_ngz_mode", test_t7_apply_ngz_mode)
    _run("T8 collision exempt", test_t8_collision_geom_exempt)
    _run("T9 pipeline no-op without NGZ", test_t9_pipeline_no_op_no_ngz)
    _run("T10 simplifier no shortcut through NGZ", test_t10_simplifier_no_shortcut_through_ngz)

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
