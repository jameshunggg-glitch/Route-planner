# 單一 NGZ 進入 `run_p2p` 的執行流程（敘述版，非程式碼）

## Context

使用者跑了 demo，發現實際輸出跟預期有落差。為了精準找出落差在哪步、哪個分支，先把現行程式碼「給單一 NGZ polygon、`ngz_mode="lenient"` 時」的決策流程逐步敘述。
本文不貼程式碼，只描述「輸入 → 計算 → 輸出 / 分支」。

**假設情境**（後面所有步驟都基於這個）：

- 呼叫：`run_p2p(out, origin_ll, dest_ll, ngz_polygons=[ngz_rect], ngz_mode="lenient", ngz_cfg=...)`
- 一個 shapely Polygon 當 NGZ
- `ngz_cfg.clearance_m`（NGZ 外側緩衝距離，預設 5km，demo 設 10km）
- `ngz_cfg.visibility_max_dist_km`（NGZ 頂點 ↔ 鄰居視線邊最遠距離，預設 200km）
- `ngz_cfg.visibility_k_sea = 8`、`visibility_k_land_t = 4`、`group_merge_eps_m = 1km`

---

## 流程

### 步驟 0：pipeline 進入點

**輸入**：`out`、`origin_ll`、`dest_ll`、`ngz_polygons=[poly]`
**動作**：
- `has_ngz = bool(ngz_polygons)` → True
- 取出全球 land collision（metric CRS）：`collision_m = out["layers"]["COLLISION_M"]`，這是「整個地球海岸線 buffer」的 multipolygon
- `build_base_graph(...)` 或 `G_in` 取出 cached G（247K 節點 / 293K 邊規模）

**輸出**：手上有 `collision_m`（global，raw）、`G`（cached）。進入 NGZ 區塊。

---

### 步驟 1：把 global 陸地 clip 成 NGZ 周圍小範圍（效能關鍵）

**為什麼有這步**：global 海岸線含上千個 polygon，後面 shapely 重型操作（buffer / unary_union / difference）對它跑會耗 60-160 秒。NGZ 的處理範圍只在自己 bbox + clearance + visibility 半徑內。

**動作**：
- 把 NGZ polygon 投影到 metric → 算 bbox
- pad = max(`clearance_m × 2`, `visibility_max_dist_km × 1500`) + 5km（demo 設定下約 305km）
- `box(NGZ.bbox + pad).intersection(global_land)` → `land_clip_m`（小 multipolygon，只剩 NGZ 周圍幾十個 polygon）

**分支**：
- 若 NGZ 在開闊海域、bbox + pad 範圍內無陸地 → `land_clip_m = None`（後續所有「跟陸地有關」的差集 / collision 自動跳過）
- 若包含部分陸地 → `land_clip_m = 小 multipolygon`

**輸出**：`land_clip_m`（後續所有 NGZ 內部運算都用這個，不再碰 global）

---

### 步驟 2：`normalize_ngz_inputs` — 把使用者 NGZ 規範化

**動作分三個子步驟**：

#### 2a. 跨 dateline 切分
對輸入 polygon 看 bbox lon span：
- `lon_span < 180` 且 `lon ∈ [-180, 180]` → **不切，原樣**
- `lon_span ≥ 180` 或有頂點在 ±180 外 → 用「unwrap 到中位數參考經度 → box intersection 切兩半 → 平移回 [-180, 180]」處理，回傳 2 個子 polygon

**單一矩形 NGZ 在開放海域的情境**：通常 lon_span 小，不切，原樣往下。

#### 2b. 扣陸地
對每個（dateline 切分後的）子 polygon：
- 若 `land_clip_m` 為 None（步驟 1 回 None）→ **不扣，原樣**
- 否則先把 `land_clip_m` 投回 lon/lat → `sub.difference(land_collision_ll)` → 可能拆成多個小碎片
- 對每個碎片只取 `polygon.exterior`（**忽略 holes**，照規格決議），存成 `(member_id, sub_poly_ll)`

**分支**：
- NGZ 完全在海上 → 1 個子 polygon
- NGZ 邊緣切到一塊陸地 → 1~2 個碎片
- NGZ 大範圍跨陸地（例如台灣海峽矩形）→ 可能切出幾十個碎片

#### 2c. 合併連通群組
- 把所有碎片投影到 metric
- `unary_union([碎片 buffered by group_merge_eps_m/2])` → 用 `eps=1km` 的 buffer 把「幾乎相連」的碎片合在一起
- 拆 multipolygon 出多個 cluster
- 對每個碎片，看它的 representative_point 落在哪個 cluster → 歸入該 cluster
- 每個 cluster 內的碎片再做一次 `unary_union` 變成最終的 `polygon_m` 與 `polygon_ll`
- 包成 `NgzGroup(group_id="ngz_group_0", member_ids=[...], polygon_ll, polygon_m)` list

**單一海上矩形情境**：1 group。
**單一矩形跨陸地情境**：可能 1~N 個 group，依碎片是否相連而定。

**輸出**：`ngz_groups: List[NgzGroup]`

---

### 步驟 3：`build_ngz_t_rings` — 為每個 group 建 T-ring

對每個 group 獨立執行：

#### 3a. 算 collision（給 taut visibility 用）
collision = `當前 group 的 polygon_m` ∪ `其他 group 的 polygon_m`（單 group 時為空）∪ `land_clip_m`

關鍵：**當前 group 自己也納入 collision**——若不放，taut 簡化器會把環線剪到 NGZ 內部抄捷徑（PR1 T2 已踩過）。

#### 3b. 建 envelope（外擴 clearance）
- `group.polygon_m.buffer(clearance_m)` → 一個外擴 polygon（NGZ 外側 clearance_m 的形狀）
- 取最大那塊 envelope 的 exterior

#### 3c. 取樣 envelope 上的點
- 沿著 envelope 外輪廓每 `ring_sample_km`（預設 5km）取一個點，包成 closed list
- 一個矩形 NGZ + 10km clearance 大致取出 30-60 個點

#### 3d. 修正掉進 collision 的取樣點
- 對每個取樣點檢查是否在 collision 內
- 若是，用 `nearest_points(collision.boundary, p)` 找方向把它推出去；最多 `point_fix_max_iter=20` 次
- 一般 NGZ 在開放海域 + 沒有其他 group → 沒有點被推

#### 3e. taut 簡化（visibility-based）
- 從 closed ring 找最大 gap 切開變 open polyline
- 用「greedy farthest-visible jump」往前跳：從 i 找最遠的 j，使 (i,j) 直線不切到 collision
- 跳到 j 繼續，把所有 i ~ j 之間的點丟掉
- 完成後再封回 closed ring
- 失敗 → fallback 到原 envelope 取樣點

**輸出**：`ngz_rings: List[NgzRingResult]`，每個含 `taut_pts_m`、`taut_pts_ll`（投回 lon/lat）

**單一矩形 NGZ 預期**：T-ring 是繞著矩形外側 10km 的「圓角矩形」，頂點數通常 8~16 個。

---

### 步驟 4：`build_ngz_overlay` — 產節點 + 邊 + mask

#### 4a. 建 NGZ 節點 DataFrame
- 每個 group 的 taut 頂點變成一個節點
- node_id 格式：`"NGZ:ngz_group_0:0"`、`"NGZ:ngz_group_0:1"` ...
- 屬性：`lon, lat, x_m, y_m, group_id, seq`

#### 4b. 建 ring 邊（同 group 內連續頂點）
- node[i] ↔ node[(i+1) % N]，weight = haversine 距離
- etype = `"ngz_ring"`，layer = `L_NGZ_RING`
- N 個頂點 → N 條 ring 邊

#### 4c. 建 visibility collision
- collision = `land_clip_m` ∪ `所有 group 的 polygon_m`
- prep 一份備用

#### 4d. 建 gate 邊（NGZ 頂點 ↔ sea node）
對每個 NGZ 頂點：
- 用 `out["sea_kdt"]` 查 K=8 個最近的 sea_node
- 對每個候選：
  - 距離 > `visibility_max_dist_km`（200km）→ **跳過**
  - 視線檢查 `LineString(NGZ_pt, sea_pt).intersects(collision_prep)` → 不通過 → **跳過**
  - 通過 → 加一條 gate 邊（layer = `L_NGZ_GATE`）

**單一 NGZ 預期**：每個 NGZ 頂點通常產生 1~5 條 gate 邊（剩下的因被陸地擋或太遠而被砍）。

#### 4e. 建 gate 邊（NGZ 頂點 ↔ 陸地 T-ring 頂點）
類似 4d，但對的是 `out["ring_graph"]["T_nodes"]`。
- 用 cKDTree 查 K=4 個最近陸地 T 頂點
- 視線檢查通過 → 加 gate 邊
- 開放海域 NGZ 通常很少（陸地 T 頂點都在岸邊太遠）

#### 4f. 建 NGZ ↔ NGZ 視線邊
**單一 NGZ 跳過**（需要至少 2 個 group 才會跑這段）。

#### 4g. mask 既有節點 / 邊（落在 NGZ 內的）
- ngz_union_ll = `unary_union(所有 group 的 polygon_ll)`
- 對每個 sea_node 點：`prep(ngz_union_ll).contains(Point(lon, lat))` → 是 → 收進 `masked_existing_nodes`
- 對每條 sea_graph 邊：建 LineString → `prep.intersects(line)` → 是 → 收進 `masked_existing_edges`

**注意**：這步只 mark，**不刪除原本的節點 / 邊**。實際 ban 是在步驟 6 的 compose 時生效。

**輸出**：`NgzOverlay(groups, rings, nodes_df, edges_ring_df, edges_gate_df, edges_ngz_ngz_df, masked_existing_nodes, masked_existing_edges)`

---

### 步驟 5：`apply_ngz_mode` — 處理 origin/dest 落在 NGZ 內

- `inside_origin = detect_inside_ngz(origin_ll, overlay)` → 一個 set，列出 origin 所在 group_id（可能多個 group 重疊）
- 同理 `inside_dest`

**分支（依 mode）**：
- `strict`：若 inside_origin 或 inside_dest 任一非空 → **raise NgzInsideError**，整個 query 中止
- `lenient`（demo 用的）：origin/dest **不動**，把 inside_origin / inside_dest 兩個 set 記下，後續 collision 排除這些 group
- `relocate`：若 inside_origin 非空，把 origin_ll **改成最近 T-ring 頂點的 lon/lat**（這樣它就在 NGZ 外了），把 inside_origin 清空；dest 同理

**單一 NGZ + origin/dest 在 NGZ 外的情境**：inside_origin = inside_dest = ∅，三個 mode 行為一致。

**輸出**：可能修改後的 `origin_ll, dest_ll` + `inside_origin, inside_dest` 兩個 set。

---

### 步驟 6：`compose_ngz_into_graph` — 合成 G_query

**動作**：
1. 從 `ngz_overlay` 建一個只含 NGZ 節點與 NGZ 邊（ring / gate / ngz_ngz）的小 G_overlay
2. `G_query = nx.compose(G_cached, G_overlay)` → 新 Graph，含原有 sea/ring/inject 邊 + 新增 NGZ 邊
3. 對 `masked_existing_nodes` 中每個節點：找它連到的所有邊，把邊的 `ban_mask |= B_NGZ`（用 `add_edge` 重塞淺拷的 attr，不動原 G_cached 共用 dict）
4. 對 `masked_existing_edges` 中每條邊：同樣加 ban_mask

**結果**：G_query 中
- 落在 NGZ 內的既有 sea_node 還是「在圖上」，但所有連到它的邊都帶 `B_NGZ` ban。RoutePolicy.active_ban_mask 預設含 B_NGZ → A* 會跳過這些邊
- NGZ T-ring 節點 + ring 邊 + gate 邊都加進來，A* 可走

---

### 步驟 7：`build_ngz_collision_geom` — 給 repair / simplify 用的 collision

- exempt = `inside_origin ∪ inside_dest`（lenient 模式下這些 group 從 collision 排除；strict / 沒 inside 時為空集合）
- keep_groups = 不在 exempt 的所有 group
- `ngz_union_ll = unary_union(keep_groups 的 polygon_ll)`
- `ngz_union_m = unary_union(keep_groups 的 polygon_m)`

回到 pipeline 主流程：
- `collision_m = land_global.union(ngz_union_m)` ← **二元 union**（取代之前的 `unary_union`，避免重算 global multipolygon）
- 這個 `collision_m` 後續傳給 repair / simplify

---

### 步驟 8：snap origin/dest 到圖上

**這步沒被 NGZ 改過**，跟原 pipeline 一樣：
- `snap_pair_component_aware(out, origin_ll, dest_ll, ...)`
  - 用 `out["sea_kdt"]` 查 origin/dest 附近 K 個最近 sea_node 候選
  - **NGZ T-ring 頂點不在 sea_kdt 裡**，所以 snap 不會直接連到 NGZ T-ring 頂點（這是 PR2 限制，PR3 才補）
- 注入虛擬節點 `Q:START` / `Q:END` + inject 邊到候選 sea node

**對應使用者可能的預期落差**：如果 origin/dest 很靠近 NGZ T-ring（例如 lenient 模式 + origin 在 NGZ 內），實際 snap 仍然是去找最近 sea_node 而不是 T-ring 頂點。從 origin 出來的第一步是去最近 sea_node，再走到 T-ring 頂點。

---

### 步驟 9：A* on G_query

- `apply_policy_view(G_query, policy)`：
  - 過濾邊：只保留 `(layer & enabled_layers) != 0` 且 `(ban & active_bans) == 0` 的邊
  - 預設 enabled layers 含 BASE_SEA / RING_E / RING_T / ET / TGATE_SEA / INJECT / NGZ_RING / NGZ_GATE / NGZ_NGZ_GATE
  - 預設 active bans = `B_HIGH_LAT | B_NGZ` → 任何邊只要帶 `B_NGZ` 就被砍
- 從 `Q:START` 找最短路徑到 `Q:END`，heuristic = haversine

**A* 怎麼決定繞 NGZ**（這是核心決策）：
- 系統**不會明確問**「NGZ 有沒有擋住 baseline」
- 系統把 NGZ overlay 加進圖、把 NGZ 內既有邊都 ban 掉、把 T-ring 與 gate 邊加進來
- A* 看到的圖：原本「直線穿過 NGZ」的那些 sea_edge 被 ban → 走不通 → 自動繞道
- 繞道路徑通常是：sea_node → gate → NGZ T-ring 頂點 → ring → 另一邊 NGZ T-ring 頂點 → gate → sea_node
- 若 NGZ 沒擋到 baseline（例如 NGZ 偏離 baseline 路徑） → A* 用原 sea 邊的最短路徑跟 baseline 一樣

**輸出**：`path_nodes`（節點 id 序列）、`path_ll_raw`（lon/lat 序列）

---

### 步驟 10：repair（可選，預設 do_repair=True）

- `PathRepairer.repair_path(G, path_nodes, collision_m=land_global.union(ngz_union_m), ...)`
- 對 A* 出來的每條邊：檢查是否與 collision 相交
- 若相交：嘗試找中點 `p` 使 u→p 與 p→v 都不相交（fast patch），最多兩個中點

**對 NGZ 的影響**：A* 已經走 T-ring 不會穿越 NGZ，所以 NGZ 區段通常不會 trigger repair。repair 主要對付 inject 邊（origin/dest → snap 候選）穿越陸地的情形。

---

### 步驟 11：simplify（預設 do_simplify=True）

- `simplify_path_visibility(path, collision_m=land_global.union(ngz_union_m), ...)`
- greedy visibility 簡化：從 i 找最遠 j 使 (i,j) 直線不與 collision 相交，跳掉中間
- **這步是「會抄捷徑」的危險點**——若 collision_m 沒包含 NGZ，simplifier 會把繞行壓回穿越 NGZ 的直線

**目前實作**：`collision_m` 確實有 NGZ（步驟 7 union 過），所以 simplifier 不會抄捷徑穿越 NGZ。T10 smoke test 驗證過。

---

### 步驟 12：組裝 `path_ll_final`

- core = simplified（或 repaired / raw 的 fallback）
- 前面接 `[origin_in, start_used]`（原始輸入點 + snap 用的虛擬點 lon/lat）
- 後面接 `[end_used, dest_in]`
- 連續重複點被 dedup

**注意**：如果 origin == start_used（lenient 沒移點），prepend 變成 `[origin, origin]`，dedup 後只剩一個 origin。

---

## 重要：系統「沒做」的決策

這些可能跟使用者預期不一樣：

1. **沒判斷「NGZ 有沒有擋住 baseline」**——一律建 T-ring + gate 邊，A* 自動選擇。
2. **沒回傳「baseline 跟最終路徑差多少」**——使用者要自己跑兩次（不傳 NGZ + 傳 NGZ）對比。
3. **不處理「NGZ 完全包圍 origin/dest 沒出口」**——A* 找不到路會 raise networkx 的 `NetworkXNoPath`，不會包成更友善的錯誤訊息。
4. **snap 不認得 NGZ T-ring 頂點**——origin/dest 永遠 snap 到既有 sea_node，不會直接 snap 到 T-ring。實務影響：origin 在 NGZ 內（lenient mode）時，第一步是「origin → 最近 sea_node」，這條 inject 邊**可能穿越 NGZ**——但因為 origin 所在 group 在 exempt list，collision 排除了它，所以 simplify/repair 不會擋這條邊。
5. **mask 既有節點 / 邊用 `prep.contains` / `prep.intersects`**——只判斷「節點是否在 NGZ 內」「邊是否跟 NGZ 相交」。如果某條 sea edge 兩個端點都在 NGZ 外但中段穿越 NGZ → 被 mask（會 ban）；如果 NGZ 跟 sea edge 完全沒交集 → 不 mask。
6. **多個 NGZ 群組之間的 visibility 邊不接陸地 T-ring**——只接 sea_node 跟「另一個 NGZ 的 T-ring 頂點」，不會跨陸地 T 中繼。
7. **T-ring 的 clearance 是固定值**——不會根據 NGZ 大小自適應（小 NGZ 跟大 NGZ 都用同一個 clearance_m）。

---

## 使用者下一步建議

讀完上述流程後，請對照「實際輸出」跟「預期輸出」差在哪。可能落差點：

| 預期 vs 實際 | 對應步驟 |
|---|---|
| 預期 NGZ 有遮才繞、沒遮直走；實際看似有時繞了沒必要 | 步驟 4d/4e gate 邊建立 + 步驟 9 A* cost 比較 |
| 預期 NGZ 整塊禁區；實際被切成幾個碎片 | 步驟 2b 扣陸地 |
| 預期路徑緊貼 NGZ 邊緣；實際有 clearance 距離 | 步驟 3b clearance_m 設定 |
| 預期 origin → 直接到 T-ring；實際 origin → sea_node → T-ring | 步驟 8 snap 限制 |
| 預期 simplifier 把繞行壓直；實際保留繞行 | 步驟 11（這是預期行為，T10 必過） |
| 預期 strict mode 下 NGZ 內的 origin 自動移；實際 raise 錯誤 | 步驟 5 mode 不同行為 |
| 預期路徑長度跟 baseline 完全一樣（NGZ 沒擋）；實際稍微不同 | 步驟 4g mask 可能砍掉了某些 edge 導致 A* 換路 |

把你實際看到的、跟預期不同的具體現象告訴我，我就能對應到上面哪一步、決定要怎麼修。
