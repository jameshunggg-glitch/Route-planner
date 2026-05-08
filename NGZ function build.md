# NGZ (No-Go Zone) Overlay 功能實作計畫

## Context（背景與動機）

本專案 `Route Planner` 是一套海上路徑規劃系統，目前架構為「scgraph 海洋骨架 + ring graph 陸地繞行 + repair / simplify 後處理」，使用 A* 在預建圖上找最短路徑。已上產品線、運作良好。

**現況痛點**：當需要避開使用者指定的禁區（**No-Go Zone**, 簡稱 **NGZ**，例如氣象浪高區、海盜區、ECA 排放管制區、客戶自訂禁航區）時，目前的做法只能對受影響的 scgraph 邊加超大權重——但因為**靜態圖的拓撲在建圖時就決定了**，剩餘圖的密度不足以提供合理替代邊，A* 會選出「**繞超遠 / 怪繞行 / 理論上不該走的航段**」。這是業界共通困境（pre-built graph + A* 範式的根本限制）。

**本次目標**：新增「動態 NGZ overlay」能力，讓使用者輸入任意形狀的 NGZ polygon 後，系統能產出**真正貼著 NGZ 邊緣繞過去**的合理替代航線，且回應時間仍在毫秒～秒級。

**核心洞察**：把 NGZ 當作「另一座島」走完整套既有的 ring + gate + visibility 流水線，使用 **T-ring**（taut visibility-simplified ring）作為繞行邊界。每對連續 T-ring 頂點之間的線段都是「拉到不能再拉」的 collision-free 視線段，A* 沿著它走出來的就是繞 NGZ 的最短路徑近似。NGZ overlay **只在 query 時動態建立、不進 cache**，使用 `nx.compose` 與既有 cached graph 合成臨時擴增圖。

---

## 設計總覽

```
使用者輸入：origin_ll、dest_ll、ngz_polygons (List[Polygon])、ngz_mode
       ↓
[既有] 載入 cached AOI（land collision、ring graph、sea_nodes、KDTree、prepared geom）
       ↓
[新增] NGZ Overlay 階段（query-time，純運算）：
   1. 規範化 NGZ：dateline 切分、polygon 差集（扣陸地）
   2. 連通群組合併（重疊或鄰近的 NGZ 合併成一塊）
   3. 每組 NGZ 建 T-ring（複用 ring_taut 邏輯）
   4. 建 NGZ T-ring 連續邊（L_NGZ_RING）
   5. 建 NGZ 頂點 → sea_nodes 視線邊（L_NGZ_GATE）
   6. 建 NGZ 頂點 → 陸地 T-ring 視線邊（L_NGZ_GATE）
   7. 建 NGZ 頂點 → 其他 NGZ 頂點視線邊（L_NGZ_NGZ_GATE）
   8. 對既有 G 中「位於 NGZ 內 / 穿越 NGZ」的節點/邊上 mask
   9. 把上述 overlay 用 nx.compose 合進臨時 G_query
       ↓
[既有] snap_pair_component_aware 起終點注入（snap 候選擴及 NGZ T-ring 頂點）
       ↓
[既有] A* 求路（heuristic 不變）
       ↓
[既有] Repair → Simplify（**collision geometry 須包含 NGZ**）
       ↓
[既有] 路徑組裝、輸出
```

---

## 已敲定的設計決議（不要重新討論）

| 項目 | 決議 |
|---|---|
| NGZ 輸入格式 | `shapely.geometry.Polygon` 或 `MultiPolygon`（List 形式傳入） |
| 模組位置 | 新建 `routing_map/ngz.py` 統包所有 NGZ 邏輯 |
| Inside-NGZ 預設模式 | `lenient`（允許從 NGZ 出發但不能進入其他 NGZ） |
| 多 NGZ 互動 | 連通群組（intersects 或距離 < eps）合併後各建一個 T-ring |
| NGZ-陸地相交 | `polygon.difference(land_collision)`，多塊各自處理，忽略 holes（只用 `polygon.exterior`） |
| Dateline 跨越 | 偵測 `bbox.lon_span > 180` → 切兩半，複用 `geom_utils` |
| 既有節點/邊在 NGZ 內 | 不刪節點、加 mask（沿用 layer_mask / ban_mask 模式） |
| Cache 策略 | NGZ overlay **不進 cache**；複用既有 cache 的 KDTree / prepared geom |
| Graph 合成 | `nx.compose(G_cached, G_ngz_overlay)` 出臨時擴增圖（不 mutate cached G） |
| Repair / Simplifier | 必須把 NGZ union 進其 collision geometry |
| Layer 常數 | 新增 `L_NGZ_RING`、`L_NGZ_GATE`、`L_NGZ_NGZ_GATE` |
| Multiworld 互動 | 與 NGZ 正交，R/S 分類不變 |
| A\* heuristic | 不變（Haversine） |
| 找不到路時 | 自然 raise，包成具體錯誤回呼叫端 |
| 效能 budget | 無上限，但仍用 STRtree 加速空間查詢避免浪費 |

---

## 新增模組：`routing_map/ngz.py`

這個模組統包所有 NGZ overlay 邏輯。**所有對外公開的 NGZ API 都從這裡 export**。

### 主要資料類型

```python
@dataclass
class NgzInput:
    """單一 NGZ 的輸入資料。"""
    polygon: Polygon | MultiPolygon  # 經緯度 (lon, lat)
    ngz_id: str                       # 使用者給的識別字串，用於 viz / debug
    metadata: Dict[str, Any] = field(default_factory=dict)  # 可選，例如 {"type": "weather", "color": "red"}

@dataclass
class NgzGroup:
    """連通群組合併後的單一處理單元。"""
    group_id: str                     # 例如 "ngz_group_0"
    member_ids: List[str]             # 哪些 NgzInput 合併進來
    polygon_ll: Polygon | MultiPolygon
    polygon_m: Polygon | MultiPolygon  # 投影到 metric CRS

@dataclass
class NgzRingResult:
    """單一群組的 T-ring 結果。"""
    group_id: str
    envelope_pts_m: List[Tuple[float, float]]  # E-ring（debug 用）
    taut_pts_m: List[Tuple[float, float]]      # T-ring 頂點（核心輸出）
    taut_pts_ll: List[Tuple[float, float]]     # 同上經緯度版

@dataclass
class NgzOverlay:
    """整個 query 的 NGZ overlay 結果，供 routing_graph 與 viz 使用。"""
    groups: List[NgzGroup]
    rings: List[NgzRingResult]
    nodes: pd.DataFrame  # 新增的 NGZ 節點：{node_id: "NGZ:gid:i", lon, lat, x_m, y_m, group_id}
    edges_ring: pd.DataFrame   # L_NGZ_RING：群組內 T-ring 連續邊
    edges_gate: pd.DataFrame   # L_NGZ_GATE：NGZ 對 sea/land 視線邊
    edges_ngz_ngz: pd.DataFrame  # L_NGZ_NGZ_GATE：NGZ 對 NGZ 視線邊
    masked_existing_nodes: Set[Any]  # 既有節點被 NGZ 蓋住的 id
    masked_existing_edges: Set[Tuple[Any, Any]]  # 既有邊被 NGZ 切到的 (u, v)
```

### 主要函式

```python
def normalize_ngz_inputs(
    ngz_inputs: List[NgzInput],
    *,
    land_collision_ll: Any,           # 陸地 collision，經緯度
    proj: AOIProjector,
    dateline_span_threshold: float = 180.0,
) -> List[NgzGroup]:
    """
    1. 對每個 NGZ：偵測跨 dateline → split_polygon_at_antimeridian
    2. 對每個（拆分後的）NGZ：difference(land_collision_ll) 扣陸地
    3. 處理 polygon with holes：只取 .exterior（每個 sub-polygon）
    4. 投影到 metric CRS 用於 T-ring 計算
    5. 連通群組合併：unary_union 後拆 multipolygon，按連通分組
    回傳合併後的 NgzGroup list。
    """

def build_ngz_t_rings(
    groups: List[NgzGroup],
    *,
    land_collision_m: Any,            # 陸地 collision（metric）
    cfg: NgzRingBuildConfig,
) -> List[NgzRingResult]:
    """
    對每個群組建 T-ring。
    對群組 i 建 T-ring 時：
        collision_hard_m = land_collision_m ∪ (其他所有群組的 polygon_m)
        collision_taut_m = 同上
    複用既有：
      - ring_envelope.build_envelope_polys_m（外擴 envelope）
      - ring_envelope.sample_ring_lines_m（取樣）
      - ring_envelope.fix_ring_points_outside_collision（修正掉進 collision 的點）
      - ring_taut.taut_simplify_closed_ring（核心 taut 簡化）
    """

def build_ngz_overlay(
    ngz_results: List[NgzRingResult],
    groups: List[NgzGroup],
    *,
    out: Dict[str, Any],          # cached AOI（含 sea_nodes、ring_graph、KDTree、prepared geom）
    cfg: NgzRingBuildConfig,
) -> NgzOverlay:
    """
    1. 為每個 ring 頂點生 NgzOverlay.nodes（含 metric & lon/lat 座標）
    2. 對每群組建 ring 連續邊（L_NGZ_RING）
    3. 對每個 NGZ 頂點：
         - 用 sea_kdt 查 K 個最近 sea_nodes 候選
         - 用 STRtree（陸地 T-ring）查附近陸地 T 頂點候選
         - 對每個候選做視線檢查（segment vs land_collision ∪ all_ngz_groups）
         - 通過的視線邊存進 edges_gate
    4. 對每對群組 (i, j), i < j：
         - 對候選頂點配對做視線檢查
         - 通過的存進 edges_ngz_ngz
    5. Mask 既有節點/邊：
         - 用 STRtree（既有 sea_edges / ring_edges）查每個 NGZ polygon 的 bbox 內候選
         - 精確檢查：node within ngz_union_ll → 加進 masked_existing_nodes
         - edge.linestring.intersects(ngz_union_ll) → 加進 masked_existing_edges
    視線檢查實作：用 prepared geom（一次性 prep），對 LineString 做 intersects。
    """

def compose_ngz_into_graph(
    G_cached: nx.Graph,
    overlay: NgzOverlay,
    *,
    inside_ngz_for_origin: Set[str] = None,    # lenient mode 用
    inside_ngz_for_dest: Set[str] = None,      # lenient mode 用
) -> nx.Graph:
    """
    1. G_query = nx.compose(G_cached, build_overlay_subgraph(overlay))
       注意：compose 不修改 G_cached
    2. 對 overlay.masked_existing_nodes：把所有連到該節點的邊改 weight = +inf 或 ban_mask=True
    3. 對 overlay.masked_existing_edges：同上
    4. 若 inside_ngz_for_origin/dest 非空：
         在後續視線檢查（snap, repair, simplify）的 NGZ collision 中，
         排除這些 group_ids（透過共享 state 或上下文物件）
    回傳 G_query；A* 對它跑。
    """

def build_ngz_collision_geom(
    ngz_results: List[NgzRingResult],
    *,
    proj: AOIProjector,
    exempt_group_ids: Set[str] = None,  # lenient mode 給予豁免
) -> Dict[str, Any]:
    """
    產出後處理階段（repair、simplifier）需要的 NGZ collision union。
    回傳 dict：
      {
        "ngz_union_ll": ...,
        "ngz_union_m": ...,
        "ngz_union_ll_prepared": prep(...),
        "ngz_union_m_prepared": prep(...),
      }
    這個會在 pipeline 裡被 union 到 land collision 上。
    """

def detect_inside_ngz(
    point_ll: Tuple[float, float],
    overlay: NgzOverlay,
) -> Set[str]:
    """回傳這個點所在的 NGZ group_ids（可能多個，重疊區）。空集表示不在任何 NGZ 內。"""

def apply_ngz_mode(
    origin_ll: Tuple[float, float],
    dest_ll: Tuple[float, float],
    overlay: NgzOverlay,
    mode: Literal["strict", "lenient", "relocate"],
) -> Dict[str, Any]:
    """
    依 mode 決定 origin/dest 處理方式：
    - strict: 若任一在 NGZ 內 → raise NgzInsideError
    - lenient: 若 origin 在 NGZ 內 → 在 origin_ll 標記 inside_groups；A* 視線檢查時這些 NGZ 被豁免
    - relocate: 自動把點挪到最近 NGZ T-ring 頂點（更新 origin_ll/dest_ll）
    回傳 {origin_ll, dest_ll, inside_origin: Set[str], inside_dest: Set[str]}
    """
```

### Helper: 跨 dateline 切分

```python
def split_polygon_at_antimeridian(poly: Polygon) -> List[Polygon]:
    """
    若 poly 跨 dateline，切成兩半（西半 +180 / 東半 -180）。
    若不跨，回 [poly]。
    複用 geom_utils.wrap_lon / unwrap_lon 邏輯。
    """
```

---

## 對既有檔案的修改

### `routing_map/config.py`

新增：

```python
@dataclass
class NgzRingBuildConfig:
    """NGZ 專用環建構參數。預設值複製自 RingBuildConfig，但 NGZ 通常需要不同 clearance。"""
    clearance_m: float = 5_000.0      # NGZ 通常比陸地需要更小或更大的 buffer，依場景調整
    ring_sample_km: float = 5.0
    taut_window_size: int = 16
    taut_max_tries: int = 8
    point_fix_step_m: float = 1_000.0
    point_fix_max_iter: int = 20
    min_island_area_km2: float = 0.0  # NGZ 不過濾小面積
    min_ring_length_km: float = 0.0
    visibility_k_sea: int = 8         # NGZ 頂點對 sea_nodes 取幾個視線候選
    visibility_k_land_t: int = 4      # 對陸地 T 頂點取幾個
    visibility_max_dist_km: float = 200.0  # 視線連線最大距離
    group_merge_eps_m: float = 1_000.0     # 連通群組合併距離閾值
```

並在 `LandConfig` / `RoutePolicy` 等容器中加上對 NGZ 的指引（如有需要）。

### `routing_map/__init__.py`

對外 export：

```python
from .ngz import (
    NgzInput, NgzGroup, NgzRingResult, NgzOverlay,
    build_ngz_overlay, compose_ngz_into_graph,
)
from .config import NgzRingBuildConfig
```

### `routing_map/routing_graph.py`（或 `pipeline.py` 看 layer 常數定義在哪）

新增 layer 常數：

```python
L_NGZ_RING       = 1 << N      # NGZ T-ring 連續邊
L_NGZ_GATE       = 1 << (N+1)  # NGZ 頂點對 sea / land T 視線邊
L_NGZ_NGZ_GATE   = 1 << (N+2)  # 兩個 NGZ 之間的視線邊
```

實作者要找出既有 layer mask 的最高位元，往後接續。

`build_base_graph` 不需動（NGZ 在 query 階段才合成）。

### `routing_map/pipeline.py`

`run_p2p` 簽章擴充：

```python
def run_p2p(
    out: Dict[str, Any],
    origin_ll: Tuple[float, float],
    dest_ll: Tuple[float, float],
    *,
    graph_cfg: GraphConfig,
    snap_cfg: SnapConfig,
    simplify_cfg: SimplifyConfig,
    run_cfg: RunConfig,
    # === 以下為新增 ===
    ngz_polygons: Optional[List[Polygon | MultiPolygon | NgzInput]] = None,
    ngz_mode: Literal["strict", "lenient", "relocate"] = "lenient",
    ngz_cfg: Optional[NgzRingBuildConfig] = None,
) -> RouteResult:
```

在 `run_p2p` 流程中插入 NGZ overlay 步驟（順序見「設計總覽」）。

`run_p2p_multiworld` 透傳這些參數給內部呼叫的 `run_p2p`。

### `routing_map/repairer.py` 與 `routing_map/path_simplifier.py`

**關鍵：collision geometry 須在 query 時 union NGZ。**

實作方式：
- 兩者目前接收一個 `collision_*` geom 物件
- 新增 helper（或在 pipeline 呼叫前處理）：`build_query_collision_geom(out, ngz_overlay)` → 把 cached land collision 與 NGZ collision union 起來，prep 一份送進 repairer / simplifier
- 對 `lenient` mode：origin/dest 所在的 NGZ groups 從 query collision 中**排除**（透過 `exempt_group_ids` 參數），這樣 repair/simplify 不會在這些區域反過來懲罰路徑

**警告**：如果忘了這步，A* 找出的繞行路徑會被 `path_simplifier` 抄捷徑「優化」回穿越 NGZ 的直線。這是必踩的坑。

### `routing_map/snap.py`

`snap_pair_component_aware`（或其呼叫上層）的 K-nearest 候選清單**要包含 NGZ T-ring 頂點**。

實作上：
- 視 NGZ 頂點為新類型節點（kind="NGZ_T"）
- snap 時的 KDTree 要包含 overlay 的節點，或另外查一次 NGZ 節點 KDTree 並合併候選

對於 `lenient` mode：origin 落在 NGZ 內時，從 origin 注入的視線邊**允許穿越自己所在的 NGZ**（其他 NGZ 仍禁止）。實作為視線檢查多接 `exempt_group_ids` 參數。

### `routing_map/viz_folium.py` 或 `viz_layers.py`

新增 NGZ 視覺化 helper：

```python
def add_ngz_layers(
    m: folium.Map,
    overlay: NgzOverlay,
    *,
    show_envelope: bool = False,
    show_failed_visibility: bool = True,  # debug 用
    failed_visibility_candidates: Optional[List[...]] = None,
) -> None:
    """
    為每個 NGZ group 各加一個 FeatureGroup，內含：
      - 原始 polygon（半透明紅）
      - E-envelope（淺色虛線，可選）
      - T-ring 多邊形（實線 + 頂點 marker）
      - T-ring 頂點 → sea / land T 視線邊（綠色細線）
      - T-ring 頂點 → 其他 NGZ 視線邊（黃色）
      - （debug）失敗的視線候選邊（紅色虛線）
      - （若有）路徑經過 NGZ 區段（粗藍色突顯）
    各 FeatureGroup 名稱：f"NGZ:{group_id}:{layer_kind}"
    與既有 _SelectAllNoneControl 相容。
    """
```

實作上要把 `failed_visibility_candidates` 從 `build_ngz_overlay` 流程蒐集起來（debug 模式才開）。

---

## 演算法細節

### T-ring 構建（NGZ 群組 i）

```python
# Inputs:
#   group_i.polygon_m            # 此群組 polygon（metric）
#   land_collision_m             # 陸地 collision（metric）
#   other_ngz_unions_m           # 其他群組合併（metric）
#   cfg: NgzRingBuildConfig

# Step 1: build envelope around NGZ
env_polys = build_envelope_polys_m(
    land_union_m=group_i.polygon_m,         # 注意這裡傳的是 NGZ，不是陸地
    clearance_m=cfg.clearance_m,
)

# Step 2: extract & sample exterior rings
lines = extract_exterior_lines(env_polys)
sampled = sample_ring_lines_m(lines, step_m=cfg.ring_sample_km * 1000)

# Step 3: build collision = land + other NGZs
# (注意：不包含 group_i 自己，因為 T-ring 在 group_i 外面，不該避免自己)
collision_for_taut = unary_union([land_collision_m, other_ngz_unions_m])

# Step 4: taut simplify
for pts_closed in sampled:
    pts_fixed, _ = fix_ring_points_outside_collision(
        pts_closed, collision_geom=collision_for_taut, cfg=cfg
    )
    taut_pts, stats = taut_simplify_closed_ring(
        pts_fixed,
        collision_taut_m=collision_for_taut,
        collision_hard_m=collision_for_taut,
        cfg=cfg,
    )
    # store taut_pts as NgzRingResult
```

### 視線連線（NGZ 頂點 V → sea node S）

```python
# collision = land + ALL NGZ groups
# 注意：包含 V 自己所在的群組（因為 V 在群組外側 clearance_m，不會切到自己）
collision_visibility_m = unary_union([land_collision_m, all_ngz_unions_m])
collision_prep = prep(collision_visibility_m)

def is_visible(p1_m, p2_m) -> bool:
    seg = LineString([p1_m, p2_m])
    return not collision_prep.intersects(seg)

# 對每個 V：
#   1. 用 sea_kdt 查 K 個最近 sea_nodes
#   2. 對每個候選 S 做 is_visible
#   3. 通過的 (V, S) 加進 edges_gate（weight = haversine 距離）
```

### Mask 既有節點/邊

```python
ngz_union_ll = unary_union([g.polygon_ll for g in groups])
ngz_union_ll_prep = prep(ngz_union_ll)

# 用 STRtree 加速空間查詢
sea_node_tree = STRtree([Point(n.lon, n.lat) for n in sea_nodes])
sea_edge_tree = STRtree([LineString(...) for e in sea_edges])

# 過濾既有節點
for ngz_poly in groups:
    candidates = sea_node_tree.query(ngz_poly.polygon_ll)
    for node in candidates:
        if ngz_union_ll_prep.contains(Point(node.lon, node.lat)):
            masked_existing_nodes.add(node.id)

# 過濾既有邊
for ngz_poly in groups:
    candidates = sea_edge_tree.query(ngz_poly.polygon_ll)
    for edge in candidates:
        if ngz_union_ll_prep.intersects(edge.linestring):
            masked_existing_edges.add((edge.u, edge.v))
```

### 模式處理（mode）

```python
inside_origin = detect_inside_ngz(origin_ll, overlay)
inside_dest = detect_inside_ngz(dest_ll, overlay)

if mode == "strict":
    if inside_origin or inside_dest:
        raise NgzInsideError(...)

elif mode == "lenient":
    # 不修改 origin/dest，但下游視線檢查時對這些 group 豁免
    pass

elif mode == "relocate":
    if inside_origin:
        origin_ll = nearest_taut_vertex_ll(origin_ll, overlay, group_ids=inside_origin)
        inside_origin = set()  # relocate 後不再 inside
    if inside_dest:
        dest_ll = nearest_taut_vertex_ll(dest_ll, overlay, group_ids=inside_dest)
        inside_dest = set()
```

`exempt_group_ids = inside_origin | inside_dest` 在 lenient 模式下用於：
- snap 階段：origin/dest 對 NGZ T-ring 的視線檢查
- repair / simplify 階段：collision geom 排除這些 group

### Dateline 處理

```python
def split_polygon_at_antimeridian(poly: Polygon) -> List[Polygon]:
    bbox = poly.bounds  # (minx, miny, maxx, maxy)
    if bbox[2] - bbox[0] < 180:
        return [poly]   # 不跨
    
    # 用 unwrap 把 polygon 在連續經度空間中表示
    # 例如：原始 lon ∈ {-179, 178} → unwrap 到 {-179, -182}
    # 然後在 lon = -180 切開
    # 切開後一半 unwrap 回 [-180, 180]
    # 細節：用 shapely 的 split + LineString(lon=-180) 切
    ...
```

最終輸出的 path 通過既有 `geom_utils.split_antimeridian_polyline` 處理，不需特殊改動。

---

## 可重用的既有函式（不要重新發明）

| 用途 | 既有函式 | 路徑 |
|---|---|---|
| 建 envelope polygon | `build_envelope_polys_m` | `routing_map/ring_envelope.py:29` |
| 取樣環上的點 | `sample_ring_lines_m` | `routing_map/ring_envelope.py:77` |
| 修正掉進 collision 的點 | `fix_ring_points_outside_collision` | `routing_map/ring_envelope.py:93` |
| Taut 簡化（核心） | `taut_simplify_closed_ring` | `routing_map/ring_taut.py:119` |
| 視線檢查 | `_segment_intersects_collision` | `routing_map/ring_taut.py:9` |
| Greedy 視線簡化 | `greedy_visibility_simplify_open` | `routing_map/ring_taut.py:85` |
| Dateline 經度 wrap | `wrap_lon` / `unwrap_lon` | `routing_map/geom_utils.py:20`、`28` |
| 路徑切 dateline | `split_antimeridian_polyline` | `routing_map/geom_utils.py:49` |
| 投影 lon/lat ↔ metric | `AOIProjector` / `geom_to_m` / `geom_to_ll` | `routing_map/geom_utils.py` |
| 取既有 projector | `get_projector` | `routing_map/geom_utils.py:242` |
| 取既有 collision geom | `get_collision_metric` | `routing_map/geom_utils.py:267` |
| Sea / ring KDTree | `out["sea_kdt"]` 等 | `routing_map/cache_utils.py`（重建在 load 時） |
| T-gate 視線連線（template） | `build_tgate_sea_connectors` | `routing_map/t_gate_connectors.py` |
| 起終點注入 | `snap_pair_component_aware` | `routing_map/snap.py` |
| Path repair | `PathRepairer.repair_path` | `routing_map/repairer.py` |
| Path simplify | `simplify_path_visibility` | `routing_map/path_simplifier.py` |

---

## 公開 API 契約

```python
from routing_map import (
    build_aoi, run_p2p, run_p2p_multiworld,
    NgzInput, NgzRingBuildConfig,
)
from shapely.geometry import Polygon

# 1. 既有：建 AOI（不變）
out = build_aoi(cfg)

# 2. 使用者準備 NGZ
ngz_polys = [
    Polygon([(120, 20), (122, 20), (122, 22), (120, 22)]),  # 一個矩形 NGZ
    NgzInput(polygon=Polygon(...), ngz_id="weather_zone_A", metadata={...}),
]

# 3. 跑路徑（單一 endpoint pair）
result = run_p2p(
    out, origin_ll=(118, 22), dest_ll=(125, 30),
    graph_cfg=..., snap_cfg=..., simplify_cfg=..., run_cfg=...,
    ngz_polygons=ngz_polys,
    ngz_mode="lenient",
    ngz_cfg=NgzRingBuildConfig(clearance_m=10_000),
)

# 4. multiworld 也支援
result = run_p2p_multiworld(
    out, origin_ll=..., dest_ll=...,
    ngz_polygons=ngz_polys,
    ngz_mode="lenient",
)

# 5. 結果中可拿到 NGZ overlay 給 viz 用
result.ngz_overlay  # NgzOverlay 物件，可丟給 add_ngz_layers
```

`RouteResult` 結構新增：
```python
@dataclass
class RouteResult:
    # ... 既有欄位 ...
    ngz_overlay: Optional[NgzOverlay] = None  # 新增
    ngz_inside_origin: Set[str] = field(default_factory=set)
    ngz_inside_dest: Set[str] = field(default_factory=set)
```

---

## 驗證計畫（end-to-end 測試）

實作完成後須跑以下測試確認正確性：

### Test 1: 退化測試（沒有 NGZ）
- 不傳 `ngz_polygons`（或傳空 list）
- 確認結果與既有 pipeline **完全相同**（path_ll_final、lengths_km 全部一致）
- 目的：確認 NGZ 邏輯零侵入既有功能

### Test 2: 簡單矩形 NGZ
- 在 origin → dest 直線途中放一個矩形 NGZ
- 預期：路徑明顯繞過 NGZ，且**沿著 NGZ 邊緣轉折**（不會繞超遠）
- 視覺驗證：用 `add_ngz_layers` 在 folium 上呈現，肉眼確認

### Test 3: U 字形非凸 NGZ
- 一個 U 字形 NGZ，原 scgraph 直線會穿過 U 開口
- 預期：路徑從 U 開口進入 / 出去，**不繞 U 的外圍**
- 目的：驗證「直接用 polygon 頂點不用 convex hull」的決策正確

### Test 4: 多 NGZ
- 三個 NGZ：一個跟陸地相交、一個跨 dateline、兩個彼此重疊
- 預期：
  - 陸地相交那個被 difference 扣掉陸地部分
  - dateline 那個被切兩半各自處理
  - 重疊那兩個合併成一個群組共用一個 T-ring
- 視覺驗證：每個 NGZ 的 T-ring 都正確顯示，且互不衝突

### Test 5: Inside-NGZ start/end (lenient 模式)
- origin 故意放在某個 NGZ 內部
- 預期：路徑成功生成，從 origin 直接出 NGZ 後正常繞行其他 NGZ
- 同樣測 strict（應 raise）與 relocate（應自動挪 origin）

### Test 6: Repair / Simplifier 不抄捷徑
- 設計一個 NGZ 在原本 A* 應走直線的地方，使得繞行明顯比直線長
- 跑完後檢查 `path_ll_final` 沒有任何邊穿過 NGZ
- 額外用 `path_polygon.intersects(ngz_polygon)` assertion 嚴格驗證
- **這是最容易出錯的一關**，必過

### Test 7: 視線連線 debug
- 開啟 `show_failed_visibility=True` 模式
- 確認失敗候選邊有正確記錄並渲染（紅色虛線）
- 用一個刁鑽形狀 NGZ 觀察哪些連線被擋掉

### Test 8: 既有節點 mask
- 設計 NGZ 完整覆蓋一個 sea_node
- 確認該 sea_node 連出的所有邊都被 mask（A* 不會用它）
- 驗證方式：log 出 `overlay.masked_existing_nodes`，肉眼比對

### Test 9: A* 找不到路（退化）
- 設計 NGZ 完全包圍 origin（origin 在某海灣，NGZ 把出口都堵住）
- 預期：raise 清楚錯誤訊息（不是 networkx 原始錯誤）
- mode 為 lenient 也應如此（origin 出得來但無路可達 dest）

### Test 10: 效能 sanity check
- 即便 budget 無上限，仍 log query-time NGZ overlay 各階段耗時
- 一般情況下總額應 < 1 秒（5 個 NGZ）；長航程可接受到 5 秒
- 用 STRtree 加速且如果某階段超 10 秒，視為 bug 要排查

---

## 實作順序建議

1. 建 `routing_map/ngz.py` 骨架：dataclasses + 函式 stub
2. 實作 `normalize_ngz_inputs`（dateline 切分、polygon 差集、群組合併）
3. 實作 `build_ngz_t_rings`（複用 ring_envelope / ring_taut）
4. 實作 `build_ngz_overlay`（節點/邊產生、視線連線、mask 既有）
5. 修改 `config.py` 加 `NgzRingBuildConfig`
6. 修改 `pipeline.py:run_p2p` 簽章與流程
7. 修改 `repairer.py` / `path_simplifier.py` 接受 NGZ collision
8. 修改 `snap.py` 把 NGZ T-ring 頂點納入 snap 候選
9. 實作 `apply_ngz_mode` 並接到 pipeline
10. 加 layer 常數到 `routing_graph.py`
11. 實作 viz helper `add_ngz_layers`
12. 寫測試（Test 1 ~ 10）
13. 手動跑 `Route_Planner.ipynb` 確認 end-to-end 正常

每一步完成後跑 Test 1 確認沒破壞既有功能。

---

## 風險與注意事項

1. **Repair / Simplifier 的 NGZ collision update 是必踩坑**——忘了這步系統會看似正常但實際抄捷徑。Test 6 必過。
2. **Lenient 模式的 exempt_group_ids 需要貫穿整個 query**——從 snap、A* 視線檢查到 repair/simplify。任一環節漏掉就會出錯。建議用 dataclass 把 query state 封裝起來傳遞，不要用 global 或 thread-local。
3. **`nx.compose` 的副作用**：請確認 networkx 版本（`requirements.txt: networkx==3.4.2`）的 compose 行為——它是否複製節點屬性？是否合併 edge attributes？實作者請寫一個小測試確認，避免後續 debug。
4. **空間索引快取**：`out["sea_kdt"]`、land T-ring STRtree 等都是既有的，**不要重複建**。如果發現某個索引沒在 `out` 裡，先檢查 `cache_utils` 是否在 load 時重建。
5. **不要動 idx**：呼應使用者的明確要求，本實作**只加東西、不改既有 sea_node / sea_edge 的 id 或順序**。所有 NGZ 節點 id 都用 `"NGZ:..."` 前綴與既有命名空間隔離。
6. **Dateline + 空間索引**：STRtree 用 lon/lat 在 dateline 附近會出錯（因為矩形跨不過 ±180）。建議：dateline 切分**在進入索引之前**完成，索引內的所有 polygon 都不跨 dateline。
