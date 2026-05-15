# NGZ Pipeline 重設計：Baseline + Local Patching

## Context

目前 `routing_map/ngz.py` + `routing_map/pipeline.py` 的 NGZ 整合（PR1 + PR2）採用
「**把 NGZ 加進 collision、把 NGZ 內既有 sea edge 全部 ban、在擴增圖上跑一次 A\***」
的範式。

問題：當 NGZ 擋住既有 scgraph 既存海路時，因為靜態圖的拓撲在建圖時就決定了，被 ban 後
剩下的圖密度不足以提供「形狀貼近原路線」的替代邊，A\* 會選出跟 baseline 形狀差很多
的繞行（例如沿海岸線繞一圈），不符合「最短繞 NGZ」的直覺。

新範式（使用者構想）：**Baseline + 局部 Patching**
1. 先跑「無 NGZ」baseline `P0`
2. 偵測 `P0` 中跟 NGZ polygon 相交的子段（可能 0、1、多段）
3. 對每段：取進入錨點 A、離開錨點 B，在 `{A, B, NGZ T-ring 頂點}` 上跑 visibility
   shortest path，替換掉被擋的子段
4. 對整條 patched path 做一次 visibility simplification（collision = land + NGZ）

關鍵性質：
- **NGZ 沒擋 baseline 時，結果跟 baseline 完全一致**（PR2 的 mask 機制不保證這點，
  因為 prep.intersects 會誤砍邊緣相切的 sea edge）
- 形狀承諾：跟 baseline 相似 + 只在被擋處鼓出去
- 直接利用 visibility graph 性質：凸障礙最短路徑必經頂點，T-ring 是 taut envelope
  本身已是「凸殼化」過的

---

## Settled decisions（不再討論）

- **D1a**：移除 PR2 的 compose / mask / gate-edge / `B_NGZ`-ban 機制；`pipeline.py` NGZ
  區塊重寫為新範式。保留 `normalize_ngz_inputs`、`build_ngz_t_rings`、
  `build_ngz_collision_geom`、`apply_ngz_mode`、`detect_inside_ngz`、
  `clip_collision_to_ngz_bbox`、`split_polygon_at_antimeridian`。
- **D2a**：多 NGZ 同時擋同一段 baseline → 所有相關 group 的 T-ring 頂點丟進**同一個**
  visibility graph 跑一次 shortest path。
- **D3a**：local visibility graph 中 A → B 不連通 → raise `NgzPatchUnreachableError`
  （無 fallback）。
- **D4**：snap 沿用現行設計（不認 T-ring 頂點；origin/dest 都 snap 到最近 sea_node；
  lenient 模式靠 collision exempt 允許穿越自己 NGZ）。
- **D5a**：baseline = pure A\* 結果（不 repair / 不 simplify）；patching 後做一次
  final visibility simplify（collision = land + NGZ_excluding_exempt）收尾。

---

## 1. Reuse map（既有 helper 直接接手）

| 用途 | 函式 / 物件 | 位置 | 備註 |
|---|---|---|---|
| 線段 vs collision 視線檢查 | `_segment_intersects_collision(a, b, taut_prep, hard_prep=None)` | `routing_map/ring_taut.py:9-37` | **prepared geometry 介面**。傳 `prep(collision_m)` 給 `taut_prep`、`hard_prep` 留 None。回傳 True 表示「擋住」。 |
| Path 視線簡化（Stage 5） | `simplify_path_visibility(path_ll, *, collision_m, proj, ...) -> (List[LonLat], SimplifyStats)` | `routing_map/path_simplifier.py:267-407` | `collision_m` 接受 raw 或 prepared 皆可（line 328-336 自動 prep）。Stage 5 不需改 collision 形狀。 |
| Haversine | `_haversine_km(p1, p2)` | `routing_map/ngz.py:122-129` | 私有但 module-internal；新 helper 同檔案，直接 call。 |
| Lon/Lat ↔ metric 投影 | `AOIProjector` / `geom_to_m` / `geom_to_ll` | `routing_map/geom_utils.py:148-197` | T-ring 已存 `taut_pts_m` + `taut_pts_ll`，patching 直接用 lon/lat（visibility + haversine 都在 ll 做）。 |
| NGZ collision union | `build_ngz_collision_geom(...)` | `routing_map/ngz.py:875-905` | 已支援 `exempt_group_ids`，回傳 `{ngz_union_ll, ngz_union_m, *_prepared}`。 |
| 取 cached land collision | `get_collision_metric(out, prefer_prepared=False)` | `routing_map/geom_utils.py:267-285` | `pipeline.py:283` 已用；`has_ngz=True` 走 `prefer_prepared=False`。 |
| A\* 入口 | `nx.astar_path(G_view, start, end, heuristic, weight)` | `routing_map/pipeline.py:513-519` | 結果 node id list；`pipeline.py:524` 的 `path_ll_raw` 即新範式的 P0，**不需跑兩次 A\***。 |
| Local shortest path（Stage 3） | `nx.shortest_path(G_local, A, B, weight="weight")` | networkx 內建 | 圖 ≤ 幾十節點，Dijkstra 即可。`NetworkXNoPath` → catch 後 raise `NgzPatchUnreachableError`。 |
| T-ring 頂點 | `NgzRingResult.taut_pts_ll` / `taut_pts_m` | `ngz.py:74-80, 396-468` | closed list（first==last），patching 用要 dedupe 末點。 |
| inside-NGZ 點偵測 | `detect_inside_ngz(point_ll, overlay)` | `ngz.py:916-930` | 沿用：產出 `inside_origin/dest` 作 exempt set。 |

**載入點關鍵發現**：
- `simplify_path_visibility` 接 raw 或 prep collision 都行 → Stage 5 wiring **完全不動**。
- `_segment_intersects_collision` **只**接 prepared → 新 helper 內部 `prep(...)` 一次 cache。
- `path_ll_raw` 是已含 inject 後 graph 節點 ll 的序列，結構穩定，可直接餵
  `detect_blocked_subpaths`。

---

## 2. 新增 helper functions（全部加在 `routing_map/ngz.py`）

```python
class NgzPatchUnreachableError(RuntimeError):
    """A 跟 B 在 local visibility graph 中不連通。Stage 3 raise。"""
```

```python
@dataclass
class BlockedSubpath:
    start_idx: int            # baseline 中 anchor A 的 index（A 本身 not-blocked）
    end_idx: int              # baseline 中 anchor B 的 index（B 本身 not-blocked）
    anchor_a_ll: LonLat
    anchor_b_ll: LonLat
    ngz_group_ids: Set[str]   # 該 blocked run 觸碰到的 NGZ group_ids
```

```python
def detect_blocked_subpaths(
    path_ll: List[LonLat],
    groups: List[NgzGroup],
    *,
    exempt_group_ids: Optional[Set[str]] = None,
) -> List[BlockedSubpath]:
    """走訪 baseline 每段 (p_i, p_{i+1})；對每個 NGZ group 的 polygon_ll 做 intersects 檢查。
    把連續 blocked segment 收集成同一個 BlockedSubpath。
    A = blocked run 之前最後一個 not-blocked 點；B = blocked run 之後第一個 not-blocked 點。
    exempt 掉的 group 不參與 detection（lenient 模式 origin/dest 所在 NGZ）。"""
```

設計細節：
- 主篩用 `prep(unary_union([g.polygon_ll for g in groups if g.group_id not in exempt]))`；
  命中後對每個 group 個別檢查 → 填 `ngz_group_ids`。
- Cross-dateline 段保護：`abs(p_i.lon - p_{i+1}.lon) > 180` 視為「不可比較」直接跳過，
  記 TODO 給未來 PR。
- baseline 第 0 段就 blocked / 最後一段 blocked → A/B 用端點。

```python
def build_local_visibility_graph(
    blocked: BlockedSubpath,
    groups: List[NgzGroup],
    rings: List[NgzRingResult],
    *,
    collision_ll: Any,           # land + NGZ_excluding_exempt（lon/lat 空間）
    pairwise_visibility: bool = False,
) -> Tuple[nx.Graph, Hashable, Hashable]:
    """節點：'A', 'B' + 每 group T-ring 頂點 (id 用既有 ngz._node_id)。
    邊：
      - A↔v / B↔v：對所有 v ∈ V，跑 _segment_intersects_collision；通過 → 加邊（haversine km）
      - A↔B：直接視線檢查（通常不通，fallback）
      - 每 group 內部 ring 連續邊（haversine km）
      - pairwise_visibility=True 時加非相鄰 V 兩兩視線邊（預設 False，靠 Stage 5 收尾）
    回傳 (G_local, 'A', 'B')。"""
```

```python
def solve_local_patch(
    g_local: nx.Graph,
    anchor_a_id: Hashable,
    anchor_b_id: Hashable,
    *,
    rings: List[NgzRingResult],
    anchor_a_ll: LonLat,
    anchor_b_ll: LonLat,
) -> List[LonLat]:
    """nx.shortest_path(weight='weight')。把 node id 序列翻成 lon/lat：
       A → 中間 'NGZ:gid:i' 查 rings 對應 taut_pts_ll[i] → B。
    NetworkXNoPath → raise NgzPatchUnreachableError(含 anchors + group_ids)。"""
```

```python
def apply_patches_to_baseline(
    baseline_ll: List[LonLat],
    patches: List[Tuple[int, int, List[LonLat]]],
) -> List[LonLat]:
    """每個 patch 替換 baseline[start_idx..end_idx]（inclusive；patch 第一/末點 == anchor A/B
    應一致，splicing 時 dedupe）。內部 sort by start_idx desc 後依序 splice，避免 index shift。"""
```

```python
def build_ngz_overlay_lite(
    groups: List[NgzGroup], rings: List[NgzRingResult],
) -> NgzOverlay:
    """精簡版：只填 groups + rings + nodes（taut 頂點 DataFrame）；edges_*/masked_* 給空。
    取代既有 build_ngz_overlay；維持 NgzOverlay dataclass 與 viz / RouteResult 兼容。"""
```

模組擺放：全部在 `ngz.py`。`__all__` 加上 `BlockedSubpath`、`NgzPatchUnreachableError`、
五個新函式；移除 `build_ngz_overlay`、`compose_ngz_into_graph`。

---

## 3. `pipeline.py` 修改點

### 3.1 Imports（line 10-32）

- **保留**：`L_BASE_SEA, L_RING_E, L_RING_T, L_ET_TRANSFER, L_TGATE_SEA, L_GATEWAY,
  L_NE_CORRIDOR, L_NW_CORRIDOR, L_INJECT, B_HIGH_LAT`
- **移除**：`L_NGZ_RING, L_NGZ_GATE, L_NGZ_NGZ_GATE, B_NGZ`（同時改 `RoutePolicy` 與
  `enabled_layers_mask()`）
- ngz import：移除 `build_ngz_overlay`、`compose_ngz_into_graph`；新增
  `detect_blocked_subpaths`、`build_local_visibility_graph`、`solve_local_patch`、
  `apply_patches_to_baseline`、`build_ngz_overlay_lite`、`NgzPatchUnreachableError`。

### 3.2 `RoutePolicy`（line 40-69）

- line 58：`active_ban_mask: int = B_HIGH_LAT | B_NGZ` → `... = B_HIGH_LAT`
- line 61-62：刪 `| L_NGZ_RING | L_NGZ_GATE | L_NGZ_NGZ_GATE`

### 3.3 NGZ block 重寫（line 316-412）

#### Block A — NGZ pre-compute（取代 line 316-412）

保留：
- `land_clip_m` 計算（line 322-346 不變）
- `normalize_ngz_inputs`（line 348-353 不變）
- `build_ngz_t_rings`（line 354-359 不變）
- `apply_ngz_mode`（line 366-372 不變）
- `build_ngz_collision_geom` + `collision_m = land_global.union(ngz_union_m)`（line 389-398 不變）

替換：
- line 360-364 `build_ngz_overlay` → `build_ngz_overlay_lite(ngz_groups, ngz_rings)`
- line 374-384 `compose_ngz_into_graph` 整段刪除（G 不擴增）
- line 401-409 debug print 改寫（去掉 edges_gate / edges_ngz_ngz / masked_*，新增
  groups / rings 計數）

新增變數保留到 outer scope（A\* 後 patching 用）：
- `ngz_groups: List[NgzGroup]`
- `ngz_rings: List[NgzRingResult]`
- `exempt_group_ids: Set[str] = inside_origin ∪ inside_dest`
- `collision_ll_for_patch`：`prep(geom_to_ll(land_clip_m, proj).union(ngz_union_ll_excluding_exempt))`
  （cache 一次，patching 重複用）

#### Block B — Stage 1 baseline A\*（line 489-529 不變）

A\* 跑在沒擴增的 G 上 → `path_ll_raw` 即 P0。**不增加第二次 A\***。

#### Block C — Stage 3-4 patching（**新增**，插在 line 529 後、line 531 前）

```python
if has_ngz and ngz_groups:
    try:
        blocked_runs = detect_blocked_subpaths(
            res.path_ll_raw, ngz_groups,
            exempt_group_ids=exempt_group_ids,
        )
        patches = []
        for blocked in blocked_runs:
            relevant_groups = [g for g in ngz_groups if g.group_id in blocked.ngz_group_ids]
            relevant_rings  = [r for r in ngz_rings  if r.group_id in blocked.ngz_group_ids]
            g_local, a_id, b_id = build_local_visibility_graph(
                blocked, relevant_groups, relevant_rings,
                collision_ll=collision_ll_for_patch,
                pairwise_visibility=False,
            )
            patch_ll = solve_local_patch(
                g_local, a_id, b_id,
                rings=relevant_rings,
                anchor_a_ll=blocked.anchor_a_ll,
                anchor_b_ll=blocked.anchor_b_ll,
            )
            patches.append((blocked.start_idx, blocked.end_idx, patch_ll))
        path_patched = apply_patches_to_baseline(res.path_ll_raw, patches)
        res.path_ll_raw = path_patched
    except NgzPatchUnreachableError as e:
        res.error = f"ngz_patch_unreachable: {e}"
        return res
    except Exception as e:
        res.error = f"ngz_patch_error: {e}"
        return res
```

### 3.4 Repair / Simplify 兼容（line 531-572）

問題：`PathRepairer.repair_path(G, path_nodes, ...)` 吃 `path_nodes` (line 536)，
patching 後 path 中的 T-ring lon/lat 沒對應 graph node id。

**選 (a)**：NGZ 模式下跳過 repair。line 533 條件改：

```python
if run_cfg.do_repair and repair_cfg is not None and collision_m is not None and not has_ngz:
```

理由：D5a 已聲明 baseline = pure A\*；patch 後也不應再 repair；最後 simplify 即可。
**Simplify 不變**（line 552-572）：吃 `res.path_ll_raw`（已是 patched），collision_m
已含 NGZ。

### 3.5 `RouteResult.ngz_overlay`（line 219）

保留欄位；新範式下 lite NgzOverlay 只填 `groups`/`rings`/`nodes`。型別兼容、debug HTML
仍能畫 T-ring。

---

## 4. 移除清單

### 4.1 `routing_map/ngz.py`

| 動作 | 對象 | 行號 |
|---|---|---|
| **刪除** | `def build_ngz_overlay(...)` | 479-744 |
| **刪除** | `def _kdt_query(...)` | 747-761（只被 build_ngz_overlay 用）|
| **刪除** | `def compose_ngz_into_graph(...)` | 768-872 |
| **新增** | `BlockedSubpath` dataclass / `NgzPatchUnreachableError` / 五個新函式 | 原 §「Compose into graph」位置 |
| `__all__`（1002-1017） | 移除 `build_ngz_overlay`、`compose_ngz_into_graph`；新增 5+ 個新名字 | |

`NgzOverlay` dataclass（line 83-92）**保留**（lite 版仍用，欄位不變但 edges/masked 給空）。
Grep 確認 `viz_layers.py` / `viz_folium.py` 沒 reference NGZ 欄位 → 移除安全。

### 4.2 `routing_map/routing_graph.py`

Grep 確認 `L_NGZ_RING / L_NGZ_GATE / L_NGZ_NGZ_GATE / B_NGZ` 在 `.py` 與 `.ipynb` 僅
被 `pipeline.py` / `ngz.py` / `routing_graph.py` 自身引用（其他 0 hit）。

| 動作 | 對象 | 行號 |
|---|---|---|
| **刪除** | `L_NGZ_RING = 1 << 9` | 35 |
| **刪除** | `L_NGZ_GATE = 1 << 10` | 36 |
| **刪除** | `L_NGZ_NGZ_GATE = 1 << 11` | 37 |
| **刪除** | `B_NGZ = 1 << 1` | 40 |

bit 9-11 + ban bit 1 → 變保留位。

### 4.3 `routing_map/__init__.py`

| 動作 | 行號 |
|---|---|
| import 區塊刪 `build_ngz_overlay`、`compose_ngz_into_graph` | 15、18 |
| `__all__` 刪這兩個 | 34、37 |
| 新增 export：`BlockedSubpath`、`NgzPatchUnreachableError`、四個新函式、`build_ngz_overlay_lite` | |

### 4.4 `ngz_smoke_test.py`

該檔（PR1+PR2 測試）import `build_ngz_overlay`、`compose_ngz_into_graph` 並有實際呼叫。
處置：**保留檔案**，把 obsolete 的 T9/T10 測試標 `[OBSOLETE]` skip；新增 N1-N8 替代。
不直接刪檔。

---

## 5. Edge cases & error handling

| 情境 | 處理 |
|---|---|
| origin 在 NGZ-A 內（lenient） | `apply_ngz_mode` → `inside_origin += {NGZ-A}`；`exempt_group_ids` 排除它 → detection 不把跟 NGZ-A 相交段標記 blocked；`collision_ll_for_patch` 與 final simplify collision 也排除 → 不反向阻擋 origin 端。其他 NGZ 仍正常。 |
| dest 在 NGZ 內 | 對稱。 |
| baseline 整段都在 exempt NGZ 內 | detection 回空 → 沒 patch；P0 = final（Stage 5 simplify collision = land + 空 NGZ → 等同只考慮 land）。 |
| NGZ 完全偏離 baseline | 0 個 blocked run；P0 直接過、最後 simplify。**新範式 vs PR2 的核心 win**。 |
| 多 NGZ 各擋 baseline 不同段 | 多個 BlockedSubpath，獨立 patch；`apply_patches_to_baseline` 內部 reverse-order splice。 |
| 多 NGZ 同擋同段（D2a） | 同一 BlockedSubpath 的 `ngz_group_ids` 含多個 → local graph 包含所有 group 頂點。 |
| Anchor A/B 落在 NGZ 內 | 不應發生；防禦性：`build_local_visibility_graph` 加邊前先 `prep.contains(Point(A))` → True → raise `NgzPatchUnreachableError("anchor inside collision")`。 |
| A→B 不連通（D3a） | `solve_local_patch` 內 `nx.NetworkXNoPath` → raise → pipeline 設 `res.error = "ngz_patch_unreachable: ..."` return。 |
| Patch 端點 vs anchor 浮點偏差 | `apply_patches_to_baseline` splice dedupe（連續重複點丟掉）— 同 pipeline.py:584-587 `_push` 風格。 |
| Cross-dateline NGZ | `normalize_ngz_inputs` 已切 dateline；T-ring 頂點落 [-180, 180]。`detect_blocked_subpaths` 對 baseline 跨 dateline 段加 `abs(lon_i - lon_{i+1}) > 180` 早期 return（fall through "not blocked"）。記 TODO。 |
| 邊緣相切 | shapely `intersects` 對 touches 也回 True → 視為 blocked（保守，避免 0 寬度貼邊）。 |

---

## 6. 驗證計畫（測試清單 N1-N8）

放在 `ngz_smoke_test.py` 新 section（取代舊 T9/T10 NGZ 整合測試）。共用 fixture：lazy
load `out_global.pkl.gz`、build base G、跑 `run_p2p` 兩次（無/有 NGZ）對比。

| ID | 場景 | Assert |
|---|---|---|
| **N1** Degenerate | `ngz_polygons=None` | `res_no_ngz.path_ll_final == res_with_ngz_None.path_ll_final` 逐點相等;`res.error is None`。 |
| **N2** NGZ off-baseline | NGZ 矩形離 baseline 200km+ | `res_with_ngz.path_ll_final == res_baseline.path_ll_final`（容忍 1e-9）；`lengths_km["final"]` 一致；`len(blocked_runs) == 0`。**核心 win assert**。 |
| **N3** 單一 NGZ 擋 baseline | 矩形跨 baseline 中段；OD 不在 NGZ 內 | (a) `LineString(path_final).intersects(NGZ_polygon_ll) == False`；(b) `final_km < baseline_km × 1.5`；(c) path 含至少一個 NGZ T-ring 頂點對應的 ll（`min_dist_to_taut_pts < 1e-3 deg`）。 |
| **N4** 凹 NGZ（U 形） | T-ring 已凸殼化 | path 只走 U 外側；`path.intersects(U) == False`；`final_km < detour_via_long_side`。 |
| **N5** 多 NGZ 同段（D2a） | 兩個分離但同擋的矩形（距離 > group_merge_eps） | 兩 group_id ∈ blocked.ngz_group_ids；patch 用兩組頂點（檢查至少一個屬 group_0、一個屬 group_1，或單邊繞行 — 看哪個短）。 |
| **N6** Lenient origin 在 NGZ 內 | NGZ-A 含 origin；NGZ-B 擋後段 | `res.ngz_inside_origin == {"NGZ-A"}`；NGZ-A 不在 patch collision；NGZ-B 正常 patch。 |
| **N7** 不可達 patch（D3a） | 構造 NGZ 圍困 baseline | `res.error.startswith("ngz_patch_unreachable")`；`res.path_ll_final is None`。若難以 reproduce，至少寫 unit test fake disconnected `g_local` 餵 `solve_local_patch` 確認 raise。 |
| **N8** Final simplify 不破 NGZ | N3/N5 設定 | `path_ll_simplified` LineString 不與 `ngz_union_ll` 相交；`len(simplified) <= len(patched)`；`simplified_km <= patched_km`。 |

呼叫 site：`ngz_smoke_test.py` `def test_n1()`...`def test_n8()`，`_load_aoi_once()` 共用。
Notebook driver `Route_Planner.ipynb` 不動（CLAUDE.md §2.2 改模組不改 notebook）。

---

## 7. 實作順序（implementation checklist）

1. **新增 helpers + dataclass + error**（`ngz.py`）：`BlockedSubpath`、
   `NgzPatchUnreachableError`、四個函式 + `build_ngz_overlay_lite`。
2. **改 `routing_graph.py`**：刪 4 個常數。
3. **改 `pipeline.py`** §3.1-3.6：imports / RoutePolicy / NGZ block 重寫 / repair skip
   條件 / debug print。**Stop**：跑 N1（degenerate）必須過 → 確保「不傳 NGZ 時行為與既有完全一致」。**Gate**。
4. **改 `__init__.py`** §4.3：補 export、移舊。
5. **刪 `ngz.py` 舊段** §4.1：`build_ngz_overlay` / `_kdt_query` / `compose_ngz_into_graph`。再跑 N1 確保沒 import 殘留。
6. **加 N2 測試**（NGZ off-baseline 一致性）。**Gate**：過了再進 7。
7. **加 N3-N6 測試**：實際 patching 行為。微調 `pairwise_visibility` 預設值（若 N3/N4
   形狀不夠合理 → 改 True 重跑）。
8. **加 N7-N8 + cleanup `ngz_smoke_test.py` 舊 T9/T10**（標 OBSOLETE）。

---

## Notes / 探索發現

- `simplify_path_visibility` 的 `collision_m` 接 raw 或 prep 都行（auto-prep）— Stage 5 wiring **不需改**。
- `_segment_intersects_collision` **只**接 prepared — 新 helper 內 prep 一次 cache。
- `pipeline.py:524` 的 `path_ll_raw` 已是 lon/lat baseline → **不需跑兩次 A\***。
- `_haversine_km` 在 `ngz.py:122` 私有 — 新 helper 同檔案直接 call，不需 expose。
- `viz_layers.py` / `viz_folium.py` 未 ref 任何 NGZ 欄位（Grep verified） — overlay 結構移除安全。
- `apply_ngz_mode` 只用 `overlay.groups` — lite NgzOverlay 完全兼容，簽章不動。
- Repair stage（line 531-550）吃 `path_nodes` 而非 `path_ll`，patched path 不對應
  graph node id → has_ngz 時 skip repair（同 D5a 精神）。
- `build_aoi` 結構（`out["sea_kdt"]` / `S_nodes` / `ring_graph`）：新範式 patching 完全不碰 — snap 仍走原 sea_kdt（D4），patching 只在 lon/lat + T-ring 自身頂點。
- ipynb 與其他 .py 沒 ref 4 個 NGZ 常數（Grep verified）— 可放心刪。
