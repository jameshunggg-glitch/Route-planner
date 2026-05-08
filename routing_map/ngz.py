"""routing_map/ngz.py — No-Go Zone (NGZ) overlay for query-time routing.

NGZ overlay 把使用者指定的禁區當成「另一座島」，在 query 時動態建 T-ring + visibility
連接邊，與既有 cached graph 用 nx.compose 合成臨時擴增圖。Overlay 不進 cache。

設計細節參見 NGZ function build.md。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple, Union

import math
import numpy as np
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
# Overlay (nodes + edges + masks)
# ---------------------------------------------------------------------------

def _node_id(group_id: str, i: int) -> str:
    return f"NGZ:{group_id}:{i}"


def build_ngz_overlay(
    ngz_results: List[NgzRingResult],
    groups: List[NgzGroup],
    *,
    proj: AOIProjector,
    out: Optional[Dict[str, Any]] = None,
    cfg: Optional[NgzRingBuildConfig] = None,
    land_collision_m: Optional[Any] = None,
) -> NgzOverlay:
    """產生 NGZ overlay：節點、群組內 ring 邊、視線邊（gate / ngz↔ngz）、既有節點/邊 mask。

    若 `out` 為 None 或缺少 sea/ring data，gate 邊與 mask 留空（PR1 純模組測試用）。
    """
    cfg = cfg or NgzRingBuildConfig()

    # ---- 1. 節點 DataFrame ----
    node_rows: List[Dict[str, Any]] = []
    for ring in ngz_results:
        # taut 環是 closed（first == last），輸出為節點時跳過最後重複點
        pts_m = ring.taut_pts_m
        pts_ll = ring.taut_pts_ll
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
    nodes_df = pd.DataFrame(node_rows, columns=["node_id", "lon", "lat", "x_m", "y_m", "group_id", "seq"])

    # ---- 2. 群組內 ring 連續邊（L_NGZ_RING）----
    ring_edge_rows: List[Dict[str, Any]] = []
    nodes_by_group: Dict[str, List[Dict[str, Any]]] = {}
    for row in node_rows:
        nodes_by_group.setdefault(row["group_id"], []).append(row)

    for gid, rows in nodes_by_group.items():
        rows_sorted = sorted(rows, key=lambda r: r["seq"])
        n = len(rows_sorted)
        if n < 2:
            continue
        for k in range(n):
            a = rows_sorted[k]
            b = rows_sorted[(k + 1) % n]
            d_km = _haversine_km((a["lon"], a["lat"]), (b["lon"], b["lat"]))
            ring_edge_rows.append({
                "u": a["node_id"],
                "v": b["node_id"],
                "weight": float(d_km),
                "length_km": float(d_km),
                "etype": "ngz_ring",
                "group_id": gid,
            })
    edges_ring_df = pd.DataFrame(ring_edge_rows, columns=["u", "v", "weight", "length_km", "etype", "group_id"])

    # ---- 3. visibility collision: land ∪ all NGZ groups ----
    coll_pieces: List[Any] = []
    if land_collision_m is not None and not getattr(land_collision_m, "is_empty", True):
        coll_pieces.append(land_collision_m)
    elif out is not None:
        layers = out.get("layers") if isinstance(out, dict) else None
        if isinstance(layers, dict) and layers.get("COLLISION_M") is not None:
            coll_pieces.append(layers["COLLISION_M"])
    for g in groups:
        if g.polygon_m is not None and not g.polygon_m.is_empty:
            coll_pieces.append(g.polygon_m)
    visibility_collision_m = unary_union(coll_pieces) if coll_pieces else None
    visibility_prep = prep(visibility_collision_m) if visibility_collision_m is not None else None

    def _seg_visible(p1_m: XY, p2_m: XY) -> bool:
        if visibility_prep is None:
            return True
        seg = LineString([p1_m, p2_m])
        try:
            return not bool(visibility_prep.intersects(seg))
        except Exception:
            return False

    # ---- 4. NGZ 頂點 → sea node 視線邊（L_NGZ_GATE）----
    gate_edge_rows: List[Dict[str, Any]] = []
    if out is not None and isinstance(out, dict):
        S_nodes = out.get("S_nodes")
        sea_kdt = out.get("sea_kdt")
        if S_nodes is not None and sea_kdt is not None and not nodes_df.empty:
            try:
                k_sea = max(1, int(cfg.visibility_k_sea))
                max_d_m = float(cfg.visibility_max_dist_km) * 1000.0
                S_xy = S_nodes[["x_m", "y_m"]].to_numpy()
                S_ids = S_nodes["node_id"].tolist()
                S_lon = S_nodes["lon"].to_numpy()
                S_lat = S_nodes["lat"].to_numpy()
                k_query = min(k_sea, len(S_ids))
                # KDTree 統一介面（支援 sklearn 或 scipy）
                for _, nrow in nodes_df.iterrows():
                    qx, qy = float(nrow["x_m"]), float(nrow["y_m"])
                    idxs = _kdt_query(sea_kdt, qx, qy, k_query)
                    for j in idxs:
                        sx, sy = float(S_xy[j, 0]), float(S_xy[j, 1])
                        d_m = math.hypot(sx - qx, sy - qy)
                        if d_m > max_d_m:
                            continue
                        if not _seg_visible((qx, qy), (sx, sy)):
                            continue
                        s_lon, s_lat = float(S_lon[j]), float(S_lat[j])
                        d_km = _haversine_km((nrow["lon"], nrow["lat"]), (s_lon, s_lat))
                        gate_edge_rows.append({
                            "u": nrow["node_id"],
                            "v": S_ids[j],
                            "weight": float(d_km),
                            "length_km": float(d_km),
                            "etype": "ngz_gate_sea",
                            "group_id": nrow["group_id"],
                        })
            except Exception:
                # gate 邊失敗不阻塞整體 overlay
                pass

        # NGZ 頂點 → 陸地 T-ring 頂點視線邊
        try:
            ring_graph = out.get("ring_graph")
            if ring_graph is not None and isinstance(ring_graph, dict):
                T_nodes = ring_graph.get("T_nodes")
                if T_nodes is not None and not T_nodes.empty and not nodes_df.empty:
                    k_land = max(1, int(cfg.visibility_k_land_t))
                    max_d_m = float(cfg.visibility_max_dist_km) * 1000.0
                    T_xy = T_nodes[["x_m", "y_m"]].to_numpy()
                    T_ids = T_nodes.get("node_key", T_nodes.get("node_id")).tolist()
                    T_lon = T_nodes["lon"].to_numpy()
                    T_lat = T_nodes["lat"].to_numpy()
                    k_query = min(k_land, len(T_ids))
                    if k_query > 0:
                        try:
                            from scipy.spatial import cKDTree as _ScipyKDT
                            t_tree = _ScipyKDT(T_xy)
                            use_scipy = True
                        except Exception:
                            t_tree = None
                            use_scipy = False
                        for _, nrow in nodes_df.iterrows():
                            qx, qy = float(nrow["x_m"]), float(nrow["y_m"])
                            if use_scipy and t_tree is not None:
                                _d, idxs = t_tree.query([qx, qy], k=k_query)
                                idxs = np.atleast_1d(idxs).tolist()
                            else:
                                # fallback: brute force
                                d2 = (T_xy[:, 0] - qx) ** 2 + (T_xy[:, 1] - qy) ** 2
                                idxs = np.argsort(d2)[:k_query].tolist()
                            for j in idxs:
                                tx, ty = float(T_xy[j, 0]), float(T_xy[j, 1])
                                d_m = math.hypot(tx - qx, ty - qy)
                                if d_m > max_d_m:
                                    continue
                                if not _seg_visible((qx, qy), (tx, ty)):
                                    continue
                                t_lon, t_lat = float(T_lon[j]), float(T_lat[j])
                                d_km = _haversine_km((nrow["lon"], nrow["lat"]), (t_lon, t_lat))
                                gate_edge_rows.append({
                                    "u": nrow["node_id"],
                                    "v": T_ids[j],
                                    "weight": float(d_km),
                                    "length_km": float(d_km),
                                    "etype": "ngz_gate_land_t",
                                    "group_id": nrow["group_id"],
                                })
        except Exception:
            pass
    edges_gate_df = pd.DataFrame(
        gate_edge_rows,
        columns=["u", "v", "weight", "length_km", "etype", "group_id"],
    )

    # ---- 5. NGZ ↔ NGZ 視線邊（L_NGZ_NGZ_GATE）----
    ngz_ngz_rows: List[Dict[str, Any]] = []
    if not nodes_df.empty and len(groups) >= 2:
        gids = sorted(nodes_df["group_id"].unique())
        nodes_by_gid = {gid: nodes_df[nodes_df["group_id"] == gid] for gid in gids}
        max_d_m = float(cfg.visibility_max_dist_km) * 1000.0
        for ai in range(len(gids)):
            for bi in range(ai + 1, len(gids)):
                a_rows = nodes_by_gid[gids[ai]]
                b_rows = nodes_by_gid[gids[bi]]
                a_xy = a_rows[["x_m", "y_m"]].to_numpy()
                b_xy = b_rows[["x_m", "y_m"]].to_numpy()
                a_ids = a_rows["node_id"].tolist()
                b_ids = b_rows["node_id"].tolist()
                a_lon = a_rows["lon"].to_numpy()
                a_lat = a_rows["lat"].to_numpy()
                b_lon = b_rows["lon"].to_numpy()
                b_lat = b_rows["lat"].to_numpy()
                # 簡單 O(n*m)：NGZ 數通常小
                for i in range(len(a_ids)):
                    for j in range(len(b_ids)):
                        d_m = math.hypot(a_xy[i, 0] - b_xy[j, 0], a_xy[i, 1] - b_xy[j, 1])
                        if d_m > max_d_m:
                            continue
                        if not _seg_visible(
                            (float(a_xy[i, 0]), float(a_xy[i, 1])),
                            (float(b_xy[j, 0]), float(b_xy[j, 1])),
                        ):
                            continue
                        d_km = _haversine_km(
                            (float(a_lon[i]), float(a_lat[i])),
                            (float(b_lon[j]), float(b_lat[j])),
                        )
                        ngz_ngz_rows.append({
                            "u": a_ids[i],
                            "v": b_ids[j],
                            "weight": float(d_km),
                            "length_km": float(d_km),
                            "etype": "ngz_ngz_gate",
                            "group_id_a": gids[ai],
                            "group_id_b": gids[bi],
                        })
    edges_ngz_ngz_df = pd.DataFrame(
        ngz_ngz_rows,
        columns=["u", "v", "weight", "length_km", "etype", "group_id_a", "group_id_b"],
    )

    # ---- 6. mask 既有節點/邊（用 lon/lat union 與 STRtree 加速可選）----
    masked_nodes: Set[Any] = set()
    masked_edges: Set[Tuple[Any, Any]] = set()

    if out is not None and groups:
        try:
            ngz_union_ll = unary_union([g.polygon_ll for g in groups if g.polygon_ll is not None])
            if ngz_union_ll is not None and not ngz_union_ll.is_empty:
                ngz_union_ll_prep = prep(ngz_union_ll)
                S_nodes = out.get("S_nodes")
                if S_nodes is not None and not S_nodes.empty:
                    for _, n_row in S_nodes.iterrows():
                        if ngz_union_ll_prep.contains(Point(float(n_row["lon"]), float(n_row["lat"]))):
                            masked_nodes.add(n_row["node_id"])
                # 既有 sea_graph 邊 mask
                sea_graph = out.get("sea_graph")
                if sea_graph is not None:
                    id2ll = out.get("id2ll") or {}
                    for u, v in sea_graph.edges:
                        ull = id2ll.get(u)
                        vll = id2ll.get(v)
                        if ull is None or vll is None:
                            continue
                        seg = LineString([ull, vll])
                        try:
                            if ngz_union_ll_prep.intersects(seg):
                                masked_edges.add((u, v))
                        except Exception:
                            continue
        except Exception:
            pass

    return NgzOverlay(
        groups=list(groups),
        rings=list(ngz_results),
        nodes=nodes_df,
        edges_ring=edges_ring_df,
        edges_gate=edges_gate_df,
        edges_ngz_ngz=edges_ngz_ngz_df,
        masked_existing_nodes=masked_nodes,
        masked_existing_edges=masked_edges,
    )


def _kdt_query(kdt: Any, x: float, y: float, k: int) -> List[int]:
    """KDTree 查詢統一介面（支援 sklearn KDTree 與 scipy cKDTree）。"""
    try:
        # sklearn 介面
        if hasattr(kdt, "query") and hasattr(kdt, "valid_metrics"):
            d, idx = kdt.query(np.array([[x, y]]), k=k)
            return [int(i) for i in np.atleast_1d(idx[0]).tolist()]
    except Exception:
        pass
    try:
        # scipy 介面
        d, idx = kdt.query([x, y], k=k)
        return [int(i) for i in np.atleast_1d(idx).tolist()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Compose into graph + collision geom + mode handling
# ---------------------------------------------------------------------------

def compose_ngz_into_graph(
    G_cached: nx.Graph,
    overlay: NgzOverlay,
    *,
    inside_ngz_for_origin: Optional[Set[str]] = None,
    inside_ngz_for_dest: Optional[Set[str]] = None,
    layer_ring: int = 1 << 9,
    layer_gate: int = 1 << 10,
    layer_ngz_ngz: int = 1 << 11,
    ban_for_masked: int = 1 << 1,
) -> nx.Graph:
    """用 nx.compose 把 overlay 合進 cached graph，回傳臨時 G_query。**不 mutate G_cached**。

    Layer / ban mask 預設值對應 NGZ function build.md 的 L_NGZ_RING / L_NGZ_GATE / L_NGZ_NGZ_GATE
    與待新增的 B_NGZ。PR2 整合 routing_graph.py 後可改傳實際常數。
    """
    if overlay is None:
        return G_cached

    # 1. 建一個只含 overlay 的 subgraph
    G_overlay = nx.Graph()

    # 節點
    for _, row in overlay.nodes.iterrows():
        G_overlay.add_node(
            row["node_id"],
            lon=float(row["lon"]),
            lat=float(row["lat"]),
            x_m=float(row["x_m"]),
            y_m=float(row["y_m"]),
            group_id=str(row["group_id"]),
            kind="NGZ_T",
        )

    # ring 邊
    for _, e in overlay.edges_ring.iterrows():
        G_overlay.add_edge(
            e["u"], e["v"],
            weight=float(e["weight"]),
            length_km=float(e["length_km"]),
            etype=str(e["etype"]),
            layer_mask=int(layer_ring),
            ban_mask=0,
            lat_max_abs=90.0,
        )

    # gate 邊
    for _, e in overlay.edges_gate.iterrows():
        G_overlay.add_edge(
            e["u"], e["v"],
            weight=float(e["weight"]),
            length_km=float(e["length_km"]),
            etype=str(e["etype"]),
            layer_mask=int(layer_gate),
            ban_mask=0,
            lat_max_abs=90.0,
        )

    # ngz↔ngz 邊
    for _, e in overlay.edges_ngz_ngz.iterrows():
        G_overlay.add_edge(
            e["u"], e["v"],
            weight=float(e["weight"]),
            length_km=float(e["length_km"]),
            etype=str(e["etype"]),
            layer_mask=int(layer_ngz_ngz),
            ban_mask=0,
            lat_max_abs=90.0,
        )

    # 2. nx.compose 合成（回傳新 Graph 但「邊 attr dict」與 G_cached 共用 reference）
    G_query = nx.compose(G_cached, G_overlay)

    # 3. 對 masked 既有節點/邊：把連到該節點的邊加 ban_mask
    #    關鍵：直接 attr["ban_mask"]= 會 mutate 到 G_cached 共用的 dict，違反規格的
    #    「不 mutate cached G」精神。改成「先淺拷 attr dict 再用 add_edge 取代」——
    #    add_edge 在 G_query 上把這條邊重新指向新 dict，G_cached 的原 dict 完全不動。
    inside_origin = inside_ngz_for_origin or set()
    inside_dest = inside_ngz_for_dest or set()

    def _ban_edge(u: Any, v: Any) -> None:
        if not G_query.has_edge(u, v):
            return
        old_attr = G_query[u][v]
        new_attr = dict(old_attr)
        new_attr["ban_mask"] = int(new_attr.get("ban_mask", 0) | ban_for_masked)
        G_query.add_edge(u, v, **new_attr)

    if overlay.masked_existing_nodes:
        for n in overlay.masked_existing_nodes:
            if not G_query.has_node(n):
                continue
            for nbr in list(G_query.neighbors(n)):
                _ban_edge(n, nbr)

    if overlay.masked_existing_edges:
        for (u, v) in overlay.masked_existing_edges:
            _ban_edge(u, v)

    # 4. lenient: origin/dest 所在 NGZ 群組的 mask 解除（未實作於 graph，需配合 collision geom 排除）
    #    這裡的 graph mask 只阻擋既有 sea/ring 邊；inside_origin/dest 主要影響 collision geom
    #    （見 build_ngz_collision_geom），故此處 graph 不做特殊處理。
    _ = (inside_origin, inside_dest)

    return G_query


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
    "split_polygon_at_antimeridian",
    "normalize_ngz_inputs",
    "build_ngz_t_rings",
    "build_ngz_overlay",
    "compose_ngz_into_graph",
    "build_ngz_collision_geom",
    "detect_inside_ngz",
    "apply_ngz_mode",
    "clip_collision_to_ngz_bbox",
]
