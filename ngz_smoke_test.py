"""ngz_smoke_test.py — 獨立 smoke test 給 PR1 的 routing_map.ngz 模組用。

執行：
    python ngz_smoke_test.py

不需要 aoi_cache，不依賴既有 build_aoi。每個測試 print PASS/FAIL，最後總結。
"""
from __future__ import annotations

import sys
import traceback
from typing import Tuple

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
        fn()
        print(f"[PASS] {name}")
        _RESULTS.append((name, True, None))
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
        _RESULTS.append((name, False, str(e)))
    except Exception as e:
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


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

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    for name, ok, msg in _RESULTS:
        flag = "PASS" if ok else "FAIL"
        line = f"  [{flag}] {name}"
        if msg:
            line += f" — {msg}"
        print(line)
    print(f"\nTotal: {n_pass}/{len(_RESULTS)} passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
