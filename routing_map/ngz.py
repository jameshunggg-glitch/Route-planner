"""routing_map/ngz.py — No-Go Zone (NGZ) overlay for query-time routing.

NGZ overlay 把使用者指定的禁區當成「另一座島」，在 query 時動態建 T-ring + visibility
連接邊，與既有 cached graph 用 nx.compose 合成臨時擴增圖。Overlay 不進 cache。

設計細節參見 NGZ function build.md。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple, Union

import math
import pandas as pd
import networkx as nx

from shapely.affinity import translate as shp_translate
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import unary_union
from shapely.prepared import prep

from .config import NgzRingBuildConfig
from .geom_utils import (
    AOIProjector,
    LonLat,
    XY,
    geom_to_ll,
    geom_to_m,
    wrap_lon,
)
from .ring_envelope import (
    build_envelope_polys_m,
    extract_exterior_lines,
    fix_ring_points_outside_collision,
    sample_ring_lines_m,
)
from .ring_taut import _segment_intersects_collision, taut_simplify_closed_ring
from .ring_types import RingBuildConfig


PolyLike = Union[Polygon, MultiPolygon]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NgzInput:
    """單一 NGZ 的輸入資料（lon/lat polygon）。"""
    polygon: PolyLike
    ngz_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NgzGroup:
    """連通群組合併後的單一處理單元。polygon 為 lon/lat 與投影後 metric 各一份。"""
    group_id: str
    member_ids: List[str]
    polygon_ll: PolyLike
    polygon_m: PolyLike


@dataclass
class NgzRingResult:
    """單一群組的 T-ring 結果。"""
    group_id: str
    envelope_pts_m: List[XY]
    taut_pts_m: List[XY]
    taut_pts_ll: List[LonLat]


@dataclass
class NgzOverlay:
    """整個 query 的 NGZ overlay 結果，供 routing_graph 與 viz 使用。"""
    groups: List[NgzGroup]
    rings: List[NgzRingResult]
    nodes: pd.DataFrame
    edges_ring: pd.DataFrame
    edges_gate: pd.DataFrame
    edges_ngz_ngz: pd.DataFrame
    masked_existing_nodes: Set[Any] = field(default_factory=set)
    masked_existing_edges: Set[Tuple[Any, Any]] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_polygons(geom: Any) -> List[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out: List[Polygon] = []
        for g in geom.geoms:
            out.extend(_iter_polygons(g))
        return out
    return []


def _ensure_valid(poly: PolyLike) -> PolyLike:
    if poly is None or poly.is_empty:
        return poly
    if not poly.is_valid:
        return poly.buffer(0)
    return poly


def _haversine_km(p1: LonLat, p2: LonLat) -> float:
    R = 6371.0088
    lon1, lat1 = math.radians(p1[0]), math.radians(p1[1])
    lon2, lat2 = math.radians(p2[0]), math.radians(p2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return float(2.0 * R * math.asin(min(1.0, math.sqrt(a))))


def _to_ngz_input(item: Union[NgzInput, Polygon, MultiPolygon], idx: int) -> NgzInput:
    if isinstance(item, NgzInput):
        if not item.ngz_id:
            return NgzInput(polygon=item.polygon, ngz_id=f"ngz_{idx}", metadata=dict(item.metadata))
        return item
    if isinstance(item, (Polygon, MultiPolygon)):
        return NgzInput(polygon=item, ngz_id=f"ngz_{idx}")
    raise TypeError(f"Unsupported NGZ input type: {type(item)!r}")


def clip_collision_to_ngz_bbox(
    collision_m: Optional[Any],
    ngz_polys_m: Sequence[Any],
    *,
    pad_m: float,
) -> Optional[Any]:
    """把 metric collision 用 box(NGZ.bbox + pad) 取交集，回傳「local 版」的小 collision。

    動機：global land collision 是含上千 polygon 的 multipolygon，對它做 unary_union /
    buffer / difference 都很慢。NGZ 的處理範圍只在自身 clearance + visibility 半徑內，
    遠處的陸地不影響結果。先 clip 成 local 後續所有 shapely 重型操作都跑得快。

    參數 `pad_m` 應大於 `clearance_m + visibility_max_dist_km*1000`，確保 T-ring 與
    visibility edge 的可達範圍內所有陸地都被保留。

    任何例外都退回原 `collision_m`（保險：寧可慢也不要錯）。
    """
    if collision_m is None or getattr(collision_m, "is_empty", True):
        return collision_m
    polys = [p for p in ngz_polys_m if p is not None and not getattr(p, "is_empty", True)]
    if not polys:
        return collision_m
    try:
        u = unary_union(polys)
        if u is None or u.is_empty:
            return collision_m
        minx, miny, maxx, maxy = u.bounds
        win = box(minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)
        clipped = collision_m.intersection(win)
        if clipped is None or clipped.is_empty:
            # 範圍內沒陸地也是合法狀態（NGZ 在開闊海域）；回 None 讓後續視為「無 land」
            return None
        return clipped
    except Exception:
        return collision_m


# ---------------------------------------------------------------------------
# Dateline splitting
# ---------------------------------------------------------------------------

def split_polygon_at_antimeridian(poly: PolyLike) -> List[Polygon]:
    """若 poly 跨 dateline，切成兩半（各自落於 [-180, 180]）；否則回 [poly]。

    判斷準則：bbox lon span > 180 視為跨 dateline。實作策略：把所有頂點 unwrap 到
    與其經度中位數一致的連續經度空間，再以 ±180 為 boundary 用 box intersection 切兩塊，
    切完平移回正規範圍。
    """
    if poly is None or poly.is_empty:
        return []

    polys_in: List[Polygon] = _iter_polygons(poly)
    out: List[Polygon] = []
    for p in polys_in:
        out.extend(_split_single_polygon_at_antimeridian(p))
    return out


def _split_single_polygon_at_antimeridian(poly: Polygon) -> List[Polygon]:
    if poly.is_empty:
        return []
    minx, miny, maxx, maxy = poly.bounds
    if (maxx - minx) < 180.0 and -180.0 <= minx and maxx <= 180.0:
        return [poly]

    ext_coords = list(poly.exterior.coords)
    xs_sorted = sorted(c[0] for c in ext_coords)
    ref = xs_sorted[len(xs_sorted) // 2]

    def _unwrap(x: float, r: float) -> float:
        d = (x - r + 180.0) % 360.0 - 180.0
        return r + d

    new_ext = [(_unwrap(x, ref), y) for (x, y) in ext_coords]
    new_holes = [
        [(_unwrap(x, ref), y) for (x, y) in interior.coords]
        for interior in poly.interiors
    ]
    try:
        unwrapped = Polygon(new_ext, new_holes)
        if not unwrapped.is_valid:
            unwrapped = unwrapped.buffer(0)
    except Exception:
        return [poly]

    ux_min, uy_min, ux_max, uy_max = unwrapped.bounds
    pad_y = 1.0
    pieces: List[Polygon] = []

    for boundary in (-180.0, 180.0):
        if ux_min < boundary < ux_max:
            left_box = box(ux_min - 1, uy_min - pad_y, boundary, uy_max + pad_y)
            right_box = box(boundary, uy_min - pad_y, ux_max + 1, uy_max + pad_y)
            left = unwrapped.intersection(left_box)
            right = unwrapped.intersection(right_box)
            for sub in _iter_polygons(left):
                lminx, _, lmaxx, _ = sub.bounds
                if lmaxx <= -180.0 + 1e-9:
                    sub = shp_translate(sub, xoff=360)
                elif lminx < -180.0:
                    pass  # straddles, leave as-is（不應發生，但保險）
                pieces.append(sub)
            for sub in _iter_polygons(right):
                rminx, _, rmaxx, _ = sub.bounds
                if rminx >= 180.0 - 1e-9:
                    sub = shp_translate(sub, xoff=-360)
                pieces.append(sub)
            if pieces:
                return pieces

    return [unwrapped]


# ---------------------------------------------------------------------------
# Normalize: dateline split + land subtract + group merge
# ---------------------------------------------------------------------------

def normalize_ngz_inputs(
    ngz_inputs: Sequence[Union[NgzInput, Polygon, MultiPolygon]],
    *,
    proj: AOIProjector,
    land_collision_ll: Optional[Any] = None,
    land_collision_m: Optional[Any] = None,
    cfg: Optional[NgzRingBuildConfig] = None,
) -> List[NgzGroup]:
    """規範化 NGZ：dateline 切分 → 扣陸地 → 投影 metric → 連通群組合併。

    `land_collision_ll` 與 `land_collision_m` 至少傳一個（用於差集扣陸地）。
    若都不傳則跳過扣陸地。`cfg.group_merge_eps_m` 控制群組合併的距離閾值。
    """
    cfg = cfg or NgzRingBuildConfig()
    if not ngz_inputs:
        return []

    # 1. 標準化 input
    items: List[NgzInput] = [_to_ngz_input(it, i) for i, it in enumerate(ngz_inputs)]

    # 2. 把 land_collision 投影成 lon/lat（若只給 metric）
    if land_collision_ll is None and land_collision_m is not None:
        try:
            land_collision_ll = geom_to_ll(land_collision_m, proj)
        except Exception:
            land_collision_ll = None

    # 3. 對每個 NGZ：dateline 切分 → 扣陸地 → 暫存 (member_id, sub_poly_ll)
    sub_records: List[Tuple[str, Polygon]] = []
    for item in items:
        cleaned = _ensure_valid(item.polygon)
        if cleaned is None or cleaned.is_empty:
            continue
        for sub in split_polygon_at_antimeridian(cleaned):
            sub = _ensure_valid(sub)
            if sub is None or sub.is_empty:
                continue
            if land_collision_ll is not None and not getattr(land_collision_ll, "is_empty", True):
                try:
                    diffed = sub.difference(land_collision_ll)
                except Exception:
                    diffed = sub
                for piece in _iter_polygons(_ensure_valid(diffed)):
                    if piece.is_empty:
                        continue
                    # 只取 exterior（忽略 holes），照 NGZ function build.md 決議
                    sub_records.append((item.ngz_id, Polygon(piece.exterior)))
            else:
                sub_records.append((item.ngz_id, Polygon(sub.exterior)))

    if not sub_records:
        return []

    # 4. 投影到 metric，再以 unary_union + buffer(group_merge_eps_m/2) 合併連通群組
    metrics: List[Polygon] = [_ensure_valid(geom_to_m(p, proj)) for (_mid, p) in sub_records]
    member_ids: List[str] = [mid for (mid, _p) in sub_records]

    eps = float(cfg.group_merge_eps_m)
    buffer_amount = max(0.0, eps / 2.0)
    expanded = [m.buffer(buffer_amount) if buffer_amount > 0 else m for m in metrics]
    union_m = unary_union(expanded) if expanded else None
    if union_m is None or union_m.is_empty:
        return []

    cluster_polys: List[Polygon] = _iter_polygons(union_m)

    # 5. 對每個 sub_record 找出它落在哪個 cluster（用點包含或交集）
    cluster_members: List[List[int]] = [[] for _ in cluster_polys]
    cluster_member_polys_m: List[List[Polygon]] = [[] for _ in cluster_polys]
    for i, (m_poly, _mid) in enumerate(zip(metrics, member_ids)):
        if m_poly.is_empty:
            continue
        # representative_point 落在 cluster 內 → 該 cluster
        rp = m_poly.representative_point()
        assigned = False
        for ci, cluster in enumerate(cluster_polys):
            if cluster.contains(rp) or cluster.intersects(m_poly):
                cluster_members[ci].append(i)
                cluster_member_polys_m[ci].append(m_poly)
                assigned = True
                break
        if not assigned:
            # fallback：最近 cluster
            best_ci, best_d = 0, float("inf")
            for ci, cluster in enumerate(cluster_polys):
                d = float(cluster.distance(m_poly))
                if d < best_d:
                    best_ci, best_d = ci, d
            cluster_members[best_ci].append(i)
            cluster_member_polys_m[best_ci].append(m_poly)

    # 6. 各 cluster 還原回 lon/lat 以便後續視覺化與 mask（差集後的版本）
    groups: List[NgzGroup] = []
    for ci, idxs in enumerate(cluster_members):
        if not idxs:
            continue
        merged_m = unary_union([metrics[i] for i in idxs])
        merged_m = _ensure_valid(merged_m)
        if merged_m is None or merged_m.is_empty:
            continue
        try:
            merged_ll = _ensure_valid(geom_to_ll(merged_m, proj))
        except Exception:
            merged_ll = unary_union([sub_records[i][1] for i in idxs])
            merged_ll = _ensure_valid(merged_ll)
        groups.append(NgzGroup(
            group_id=f"ngz_group_{ci}",
            member_ids=sorted({member_ids[i] for i in idxs}),
            polygon_ll=merged_ll,
            polygon_m=merged_m,
        ))

    return groups


# ---------------------------------------------------------------------------
# T-ring building
# ---------------------------------------------------------------------------

def _ngz_cfg_to_ring_cfg(cfg: NgzRingBuildConfig) -> RingBuildConfig:
    """把 NgzRingBuildConfig 對應到既有 ring_envelope/ring_taut 用的 RingBuildConfig。"""
    return RingBuildConfig(
        smooth_m=0.0,
        clearance_m=cfg.clearance_m,
        ring_sample_km=cfg.ring_sample_km,
        point_fix_step_m=cfg.point_fix_step_m,
        point_fix_max_iter=cfg.point_fix_max_iter,
        taut_window_size=cfg.taut_window_size,
        taut_max_tries=cfg.taut_max_tries,
        cut_strategy="best_gap",
        taut_use_clearance_buffer=True,
        taut_collision_buffer_m=None,
        min_island_area_km2=cfg.min_island_area_km2,
        min_ring_length_km=cfg.min_ring_length_km,
    )


def build_ngz_t_rings(
    groups: List[NgzGroup],
    *,
    proj: AOIProjector,
    land_collision_m: Optional[Any] = None,
    cfg: Optional[NgzRingBuildConfig] = None,
) -> List[NgzRingResult]:
    """對每個 NGZ group 建 T-ring。其他 group 的 polygon 視為 collision 一部分。"""
    cfg = cfg or NgzRingBuildConfig()
    ring_cfg = _ngz_cfg_to_ring_cfg(cfg)

    results: List[NgzRingResult] = []
    n = len(groups)
    for i, g in enumerate(groups):
        # 其他 group 的 polygon_m 合併
        other_polys = [groups[j].polygon_m for j in range(n) if j != i]
        other_union_m = unary_union(other_polys) if other_polys else None

        # collision MUST 包含當前 group 自己——T-ring 雖然在外側 clearance，但 taut
        # 簡化器靠 visibility 剪短環線，若 group_i 不在 collision 中，shortcut 會
        # 抄捷徑穿越 NGZ，導致 T-ring 縮成穿過 NGZ 的線段。
        collision_pieces = [g.polygon_m]
        if land_collision_m is not None and not getattr(land_collision_m, "is_empty", True):
            collision_pieces.append(land_collision_m)
        if other_union_m is not None and not getattr(other_union_m, "is_empty", True):
            collision_pieces.append(other_union_m)
        collision_for_ring = unary_union(collision_pieces)

        # 1. envelope 包圍 NGZ polygon
        env_polys = build_envelope_polys_m(g.polygon_m, clearance_m=ring_cfg.clearance_m)
        if not env_polys:
            continue

        # 2. 取最大那個 envelope（理論上 NGZ 一個 group 應該一個外輪廓）
        env_poly_main = max(env_polys, key=lambda p: float(p.area))
        lines = extract_exterior_lines([env_poly_main])
        sampled = sample_ring_lines_m(lines, step_m=float(ring_cfg.ring_sample_km) * 1000.0)
        if not sampled:
            continue

        env_pts = sampled[0]
        # 3. 修正掉進其他群組或陸地的 envelope 取樣點
        env_pts_fixed, _n_fix = fix_ring_points_outside_collision(
            env_pts,
            collision_geom=collision_for_ring,
            cfg=ring_cfg,
        )

        # 4. taut 簡化
        taut_pts, _stats = taut_simplify_closed_ring(
            env_pts_fixed,
            collision_taut_m=collision_for_ring,
            collision_hard_m=collision_for_ring,
            cfg=ring_cfg,
        )

        if not taut_pts:
            continue

        # 5. metric → lon/lat
        taut_ll: List[LonLat] = []
        for (xm, ym) in taut_pts:
            lon, lat = proj.m2ll(xm, ym)
            taut_ll.append((wrap_lon(lon), float(lat)))

        results.append(NgzRingResult(
            group_id=g.group_id,
            envelope_pts_m=list(env_pts_fixed),
            taut_pts_m=list(taut_pts),
            taut_pts_ll=taut_ll,
        ))

    return results


# ---------------------------------------------------------------------------
# Internal: node id helper
# ---------------------------------------------------------------------------

def _node_id(group_id: str, i: int) -> str:
    return f"NGZ:{group_id}:{i}"


def build_ngz_collision_geom(
    ngz_results: List[NgzRingResult],
    groups: List[NgzGroup],
    *,
    proj: AOIProjector,
    exempt_group_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """產出 repair / simplify 階段需要的 NGZ collision union（lon/lat 與 metric 各一份）。

    `exempt_group_ids` 中的 NGZ 從 collision 中排除（lenient 模式用）。
    """
    exempt = exempt_group_ids or set()
    keep_groups = [g for g in groups if g.group_id not in exempt]
    if not keep_groups:
        empty = Polygon()
        return {
            "ngz_union_ll": empty,
            "ngz_union_m": empty,
            "ngz_union_ll_prepared": None,
            "ngz_union_m_prepared": None,
        }

    union_ll = unary_union([g.polygon_ll for g in keep_groups if g.polygon_ll is not None])
    union_m = unary_union([g.polygon_m for g in keep_groups if g.polygon_m is not None])

    return {
        "ngz_union_ll": union_ll,
        "ngz_union_m": union_m,
        "ngz_union_ll_prepared": prep(union_ll) if union_ll is not None and not union_ll.is_empty else None,
        "ngz_union_m_prepared": prep(union_m) if union_m is not None and not union_m.is_empty else None,
    }


# ---------------------------------------------------------------------------
# Baseline + Local Patching (新範式)
# ---------------------------------------------------------------------------

class NgzPatchUnreachableError(RuntimeError):
    """A 跟 B 在 local visibility graph 中不連通。Stage 3 raise。"""


@dataclass
class BlockedSubpath:
    """baseline 中跟 NGZ 相交的連續子段。"""
    start_idx: int            # baseline 中 anchor A 的 index（A 本身 not-blocked）
    end_idx: int              # baseline 中 anchor B 的 index（B 本身 not-blocked）
    anchor_a_ll: LonLat
    anchor_b_ll: LonLat
    ngz_group_ids: Set[str] = field(default_factory=set)


def _is_prepared_geom(g: Any) -> bool:
    return g is not None and hasattr(g, "context") and not hasattr(g, "buffer")


def _to_prep(g: Any) -> Any:
    if g is None:
        return None
    if _is_prepared_geom(g):
        return g
    try:
        if getattr(g, "is_empty", False):
            return None
        return prep(g)
    except Exception:
        return None


def detect_blocked_subpaths(
    path_ll: List[LonLat],
    groups: List[NgzGroup],
    *,
    exempt_group_ids: Optional[Set[str]] = None,
) -> List[BlockedSubpath]:
    """走訪 baseline 每段 (p_i, p_{i+1})；對每個 NGZ group 的 polygon_ll 做 intersects 檢查。
    把連續 blocked segment 收集成同一個 BlockedSubpath。
    A = blocked run 之前最後一個 not-blocked 點；B = blocked run 之後第一個 not-blocked 點。
    exempt 掉的 group 不參與 detection（lenient 模式 origin/dest 所在 NGZ）。
    """
    if not path_ll or len(path_ll) < 2 or not groups:
        return []

    exempt = exempt_group_ids or set()
    active_groups = [
        g for g in groups
        if g.group_id not in exempt
        and g.polygon_ll is not None
        and not g.polygon_ll.is_empty
    ]
    if not active_groups:
        return []

    try:
        union_ll = unary_union([g.polygon_ll for g in active_groups])
        union_prep = prep(union_ll) if union_ll is not None and not union_ll.is_empty else None
    except Exception:
        union_prep = None

    group_preps: List[Tuple[str, Any]] = []
    for g in active_groups:
        try:
            group_preps.append((g.group_id, prep(g.polygon_ll)))
        except Exception:
            continue

    n = len(path_ll)
    blocked_flags: List[bool] = [False] * (n - 1)
    seg_groups: List[Set[str]] = [set() for _ in range(n - 1)]
    for i in range(n - 1):
        a = path_ll[i]
        b = path_ll[i + 1]
        # cross-dateline 段視為不可比較（TODO: 未來 PR 處理 dateline patching）
        if abs(float(a[0]) - float(b[0])) > 180.0:
            continue
        seg = LineString([a, b])
        if union_prep is not None:
            try:
                if not union_prep.intersects(seg):
                    continue
            except Exception:
                pass
        for gid, gp in group_preps:
            try:
                if gp.intersects(seg):
                    blocked_flags[i] = True
                    seg_groups[i].add(gid)
            except Exception:
                continue

    results: List[BlockedSubpath] = []
    i = 0
    while i < n - 1:
        if not blocked_flags[i]:
            i += 1
            continue
        j = i
        gids: Set[str] = set()
        while j < n - 1 and blocked_flags[j]:
            gids.update(seg_groups[j])
            j += 1
        # anchor A = baseline[i]（segment i 起點，前一段沒 blocked → 不在 NGZ）
        # anchor B = baseline[j]（segment j 起點，亦即 blocked run 之後第一個 not-blocked 點）
        a_idx = i
        b_idx = j
        a_ll = path_ll[a_idx]
        b_ll = path_ll[b_idx]
        results.append(BlockedSubpath(
            start_idx=int(a_idx),
            end_idx=int(b_idx),
            anchor_a_ll=(float(a_ll[0]), float(a_ll[1])),
            anchor_b_ll=(float(b_ll[0]), float(b_ll[1])),
            ngz_group_ids=gids,
        ))
        i = j

    return results


def build_local_visibility_graph(
    blocked: BlockedSubpath,
    groups: List[NgzGroup],
    rings: List[NgzRingResult],
    *,
    collision_ll: Any,
    pairwise_visibility: bool = False,
) -> Tuple[nx.Graph, Any, Any]:
    """Local visibility graph for one BlockedSubpath。

    節點：'A', 'B' + 每 group T-ring 頂點 (id 用既有 _node_id)。
    邊：
      - A↔v / B↔v：視線檢查通過 → 加邊（haversine km）
      - A↔B：直接視線檢查（通常不通；fallback 用）
      - 每 group 內部 ring 連續邊
      - pairwise_visibility=True 時加非相鄰 V 兩兩視線邊（預設 False，靠 Stage 5 收尾）

    `collision_ll` 接受 raw 或 prepared geom；anchor 落在 collision 內 → raise
    NgzPatchUnreachableError("anchor inside collision")。
    """
    collision_prep = _to_prep(collision_ll)

    g = nx.Graph()
    anchor_a_id = "A"
    anchor_b_id = "B"
    a_ll = blocked.anchor_a_ll
    b_ll = blocked.anchor_b_ll
    g.add_node(anchor_a_id, lon=float(a_ll[0]), lat=float(a_ll[1]))
    g.add_node(anchor_b_id, lon=float(b_ll[0]), lat=float(b_ll[1]))

    if collision_prep is not None:
        try:
            if collision_prep.contains(Point(a_ll[0], a_ll[1])) or \
               collision_prep.contains(Point(b_ll[0], b_ll[1])):
                raise NgzPatchUnreachableError(
                    f"anchor inside collision: A={a_ll} B={b_ll}"
                )
        except NgzPatchUnreachableError:
            raise
        except Exception:
            pass

    ring_by_gid = {r.group_id: r for r in rings}
    vertex_records: List[Tuple[str, int, LonLat, str]] = []  # (gid, seq, ll, node_id)
    for grp in groups:
        ring = ring_by_gid.get(grp.group_id)
        if ring is None or not ring.taut_pts_ll:
            continue
        pts_ll = list(ring.taut_pts_ll)
        if pts_ll and pts_ll[0] == pts_ll[-1]:
            pts_ll = pts_ll[:-1]
        for seq, ll in enumerate(pts_ll):
            nid = _node_id(grp.group_id, seq)
            g.add_node(nid, lon=float(ll[0]), lat=float(ll[1]),
                       group_id=grp.group_id, seq=seq)
            vertex_records.append((grp.group_id, seq, (float(ll[0]), float(ll[1])), nid))

    nodes_by_gid: Dict[str, List[Tuple[int, str, LonLat]]] = {}
    for (gid, seq, ll, nid) in vertex_records:
        nodes_by_gid.setdefault(gid, []).append((seq, nid, ll))
    for gid, rows in nodes_by_gid.items():
        rows_sorted = sorted(rows, key=lambda r: r[0])
        n_g = len(rows_sorted)
        if n_g < 2:
            continue
        for k in range(n_g):
            _, u_id, u_ll = rows_sorted[k]
            _, v_id, v_ll = rows_sorted[(k + 1) % n_g]
            d_km = _haversine_km(u_ll, v_ll)
            g.add_edge(u_id, v_id, weight=float(d_km), etype="ngz_ring", group_id=gid)

    def _seg_blocked(p1: LonLat, p2: LonLat) -> bool:
        if collision_prep is None:
            return False
        return _segment_intersects_collision(p1, p2, collision_prep, None)

    for endpoint_id, endpoint_ll in ((anchor_a_id, a_ll), (anchor_b_id, b_ll)):
        for (gid, seq, ll, nid) in vertex_records:
            if _seg_blocked(endpoint_ll, ll):
                continue
            d_km = _haversine_km(endpoint_ll, ll)
            g.add_edge(endpoint_id, nid, weight=float(d_km), etype="ngz_visibility")

    if not _seg_blocked(a_ll, b_ll):
        d_km = _haversine_km(a_ll, b_ll)
        g.add_edge(anchor_a_id, anchor_b_id, weight=float(d_km), etype="ngz_visibility")

    if pairwise_visibility and len(vertex_records) >= 2:
        for i in range(len(vertex_records)):
            gid_i, seq_i, ll_i, nid_i = vertex_records[i]
            for j in range(i + 1, len(vertex_records)):
                gid_j, seq_j, ll_j, nid_j = vertex_records[j]
                if gid_i == gid_j:
                    rows = nodes_by_gid.get(gid_i, [])
                    n_g = len(rows)
                    diff = abs(seq_i - seq_j)
                    if diff == 1 or diff == n_g - 1:
                        continue
                if _seg_blocked(ll_i, ll_j):
                    continue
                if g.has_edge(nid_i, nid_j):
                    continue
                d_km = _haversine_km(ll_i, ll_j)
                g.add_edge(nid_i, nid_j, weight=float(d_km), etype="ngz_visibility")

    return g, anchor_a_id, anchor_b_id


def solve_local_patch(
    g_local: nx.Graph,
    anchor_a_id: Any,
    anchor_b_id: Any,
    *,
    rings: List[NgzRingResult],
    anchor_a_ll: LonLat,
    anchor_b_ll: LonLat,
) -> List[LonLat]:
    """跑 nx.shortest_path(weight='weight') 並把 node id 序列翻成 lon/lat 序列。

    NetworkXNoPath → raise NgzPatchUnreachableError（含 anchors）。
    """
    try:
        node_ids = nx.shortest_path(g_local, anchor_a_id, anchor_b_id, weight="weight")
    except nx.NetworkXNoPath as e:
        raise NgzPatchUnreachableError(
            f"A={anchor_a_ll} B={anchor_b_ll} disconnected: {e}"
        )

    ring_by_gid = {r.group_id: r for r in rings}

    def _resolve(nid: Any) -> LonLat:
        if nid == anchor_a_id:
            return anchor_a_ll
        if nid == anchor_b_id:
            return anchor_b_ll
        nd = g_local.nodes.get(nid, {})
        if isinstance(nd, dict) and "lon" in nd and "lat" in nd:
            return (float(nd["lon"]), float(nd["lat"]))
        s = str(nid)
        if s.startswith("NGZ:"):
            try:
                _, gid, seq_str = s.split(":", 2)
                seq = int(seq_str)
                ring = ring_by_gid.get(gid)
                if ring is not None:
                    pts = list(ring.taut_pts_ll)
                    if pts and pts[0] == pts[-1]:
                        pts = pts[:-1]
                    if 0 <= seq < len(pts):
                        return (float(pts[seq][0]), float(pts[seq][1]))
            except Exception:
                pass
        raise KeyError(f"cannot resolve lon/lat for node {nid!r}")

    return [_resolve(nid) for nid in node_ids]


def apply_patches_to_baseline(
    baseline_ll: List[LonLat],
    patches: List[Tuple[int, int, List[LonLat]]],
) -> List[LonLat]:
    """把每個 patch 替換進 baseline 對應區段（splice）。

    patches 元素：(start_idx, end_idx, patch_ll)；patch_ll[0] 應 == baseline[start_idx]、
    patch_ll[-1] 應 == baseline[end_idx]（anchor 一致）。內部依 start_idx 降冪排序避免
    index shift。最後 dedupe 連續重複點。
    """
    if not patches:
        return [(float(p[0]), float(p[1])) for p in baseline_ll]
    work: List[LonLat] = [(float(p[0]), float(p[1])) for p in baseline_ll]
    for start_idx, end_idx, patch_ll in sorted(patches, key=lambda p: p[0], reverse=True):
        if start_idx < 0 or end_idx >= len(work) or start_idx > end_idx:
            continue
        patch_norm = [(float(p[0]), float(p[1])) for p in patch_ll]
        work[start_idx:end_idx + 1] = patch_norm

    deduped: List[LonLat] = []
    for pt in work:
        if not deduped or deduped[-1] != pt:
            deduped.append(pt)
    return deduped


def build_ngz_overlay_lite(
    groups: List[NgzGroup],
    rings: List[NgzRingResult],
) -> NgzOverlay:
    """精簡版 overlay：只填 groups + rings + nodes（taut 頂點 DataFrame）；
    edges_* / masked_* 給空。維持 NgzOverlay dataclass 與 viz / RouteResult 兼容。
    """
    node_rows: List[Dict[str, Any]] = []
    for ring in rings:
        pts_m = list(ring.taut_pts_m)
        pts_ll = list(ring.taut_pts_ll)
        if pts_m and pts_m[0] == pts_m[-1]:
            pts_m = pts_m[:-1]
            pts_ll = pts_ll[:-1]
        for i, ((xm, ym), (lon, lat)) in enumerate(zip(pts_m, pts_ll)):
            node_rows.append({
                "node_id": _node_id(ring.group_id, i),
                "lon": float(lon),
                "lat": float(lat),
                "x_m": float(xm),
                "y_m": float(ym),
                "group_id": ring.group_id,
                "seq": int(i),
            })
    nodes_df = pd.DataFrame(
        node_rows,
        columns=["node_id", "lon", "lat", "x_m", "y_m", "group_id", "seq"],
    )
    empty_edges = pd.DataFrame(
        columns=["u", "v", "weight", "length_km", "etype", "group_id"]
    )
    empty_ngz_ngz = pd.DataFrame(
        columns=["u", "v", "weight", "length_km", "etype", "group_id_a", "group_id_b"]
    )
    return NgzOverlay(
        groups=list(groups),
        rings=list(rings),
        nodes=nodes_df,
        edges_ring=empty_edges.copy(),
        edges_gate=empty_edges.copy(),
        edges_ngz_ngz=empty_ngz_ngz,
    )


# ---------------------------------------------------------------------------
# Mode handling (strict / lenient / relocate)
# ---------------------------------------------------------------------------

class NgzInsideError(ValueError):
    """origin / dest 在 NGZ 內且 mode='strict' 時 raise。"""


def detect_inside_ngz(point_ll: LonLat, overlay: NgzOverlay) -> Set[str]:
    """回傳這個點所在的 NGZ group_ids（可能多個，重疊區）。"""
    if overlay is None or not overlay.groups:
        return set()
    p = Point(float(point_ll[0]), float(point_ll[1]))
    inside: Set[str] = set()
    for g in overlay.groups:
        if g.polygon_ll is None or g.polygon_ll.is_empty:
            continue
        try:
            if g.polygon_ll.contains(p):
                inside.add(g.group_id)
        except Exception:
            continue
    return inside


def _nearest_taut_vertex_ll(
    point_ll: LonLat,
    overlay: NgzOverlay,
    *,
    group_ids: Optional[Set[str]] = None,
) -> LonLat:
    """在指定 group 的 T-ring 頂點中找最近一點（lon/lat 距離）。"""
    if overlay is None or overlay.nodes is None or overlay.nodes.empty:
        return point_ll
    df = overlay.nodes
    if group_ids:
        df = df[df["group_id"].isin(group_ids)]
    if df.empty:
        return point_ll
    lon = float(point_ll[0])
    lat = float(point_ll[1])
    # 用 Haversine 近似（小範圍可用平面）
    best_d = float("inf")
    best_ll = point_ll
    for _, r in df.iterrows():
        d = _haversine_km((lon, lat), (float(r["lon"]), float(r["lat"])))
        if d < best_d:
            best_d = d
            best_ll = (float(r["lon"]), float(r["lat"]))
    return best_ll


def apply_ngz_mode(
    origin_ll: LonLat,
    dest_ll: LonLat,
    overlay: NgzOverlay,
    mode: Literal["strict", "lenient", "relocate"] = "lenient",
) -> Dict[str, Any]:
    """依 mode 處理 origin/dest 的 inside-NGZ 狀況。

    回傳 {origin_ll, dest_ll, inside_origin, inside_dest}。
    """
    inside_origin = detect_inside_ngz(origin_ll, overlay)
    inside_dest = detect_inside_ngz(dest_ll, overlay)

    if mode == "strict":
        if inside_origin or inside_dest:
            raise NgzInsideError(
                f"origin inside={inside_origin}, dest inside={inside_dest} (mode=strict)"
            )
    elif mode == "relocate":
        if inside_origin:
            origin_ll = _nearest_taut_vertex_ll(origin_ll, overlay, group_ids=inside_origin)
            inside_origin = set()
        if inside_dest:
            dest_ll = _nearest_taut_vertex_ll(dest_ll, overlay, group_ids=inside_dest)
            inside_dest = set()
    elif mode == "lenient":
        pass
    else:
        raise ValueError(f"Unknown ngz_mode: {mode!r}")

    return {
        "origin_ll": origin_ll,
        "dest_ll": dest_ll,
        "inside_origin": inside_origin,
        "inside_dest": inside_dest,
    }


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "NgzInput",
    "NgzGroup",
    "NgzRingResult",
    "NgzOverlay",
    "NgzInsideError",
    "NgzPatchUnreachableError",
    "BlockedSubpath",
    "split_polygon_at_antimeridian",
    "normalize_ngz_inputs",
    "build_ngz_t_rings",
    "build_ngz_overlay_lite",
    "build_ngz_collision_geom",
    "detect_inside_ngz",
    "detect_blocked_subpaths",
    "build_local_visibility_graph",
    "solve_local_patch",
    "apply_patches_to_baseline",
    "apply_ngz_mode",
    "clip_collision_to_ngz_bbox",
]
