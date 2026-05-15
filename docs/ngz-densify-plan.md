# NGZ Polygon Densification Plan

> 目標讀者：下一個處理這個 repo 的 Claude agent。看完這份你應該能直接開始改 code。

## TL;DR

- **問題**：NGZ T-ring envelope 跟使用者畫的 NGZ 矩形對不齊（在長邊鼓出 50+ km，跟 5km clearance 嚴重不符）。截圖見 `docs/ngz_debug_file/Taut ring issue 1.png` / `2.png`。
- **根因**：`normalize_ngz_inputs` 把 4-vertex 矩形直接餵給 `geom_to_m`（AEQD per-vertex transform，沒 densify），4 段長 metric 直線投回 lon/lat 變成大圓弧。
- **修法**：在 `geom_to_m` 之前對每個 NGZ sub-polygon 做 densify（每條邊插中間點，segment ≤ 0.5°）。
- **預期**：弧形消失；`polygon_m` 跟 `polygon_ll` 在地球上對應同一塊區域；mask 跟 T-ring boundary 對齊。
- **改動範圍**：`routing_map/ngz.py` + `routing_map/config.py`，~30 行。
- **Risk**：低。smoke test 15/3/0 應全綠。

---

## Context

### 使用情境

Route planner 為了海上避開惡劣天氣（浪高 / 風速超標）而設計。Production 預期：
- 定期打天氣 API 抓最新資料
- 把超標 grid cell 圈起來變成 NGZ polygon
- 對 query 路徑套用 NGZ overlay，建議船隻繞行

NGZ 從 grid 出來通常已經是 dense polygon（幾十到幾百 vertex），但開發 / 測試 / fallback / 手動加 NGZ 場景會用 4 角矩形——這時 bug 浮現。

### 觀察到的現象

使用者用 4 角矩形 NGZ（在日本附近開放海域）測試時：
- NGZ rectangle 是垂直長條
- 紫色 taut ring 在**長邊（左右）**大幅向外鼓，弧形最深處離 NGZ 邊 50+ km
- 短邊（上下）幾乎貼著 NGZ 邊（offset ≈ clearance）
- 鼓的形狀像「撲克牌被彎曲」

### 證據鏈（完整推演見 `docs/ngz-taut_debug.html`）

1. Projection 是 AEQD（azimuthal equidistant），`routing_map/geom_utils.py:184`
2. AEQD 只有「通過 AOI 中心」的線同時是 metric 直線與 lon/lat 大圓；其他 metric 直線投回 lon/lat 是弧
3. `geom_to_m`（`routing_map/geom_utils.py:192`）只 per-vertex transform，沒在邊上插點
4. 在 `routing_map/ngz.py:312`，4 角矩形 → 4 個 metric vertex → 4 段長 metric 直線 → 投回 lon/lat = 4 段大圓弧
5. 鼓的幅度 ∝ (邊長)² / 距 AOI 中心距離，所以矩形長邊鼓得明顯
6. 陸地沒這問題，因為 GSHHG / Natural Earth 海岸線本身就有上千 vertex，metric 化後每段直線都是 km 級短線，AEQD 失真看不到

### 順便發現的內部不一致

不 densify 的話 `polygon_m` 跟 `polygon_ll` 在地球上代表**不同**區域：

| polygon | vertex 數 | shapely 視為的邊 | 地球上代表 |
|---|---|---|---|
| `polygon_m`（metric） | 4 | 直線 in metric | 大圓四邊形（鼓的） |
| `polygon_ll = geom_to_ll(polygon_m)`（lon/lat） | 4（round-trip 回原座標） | 直線 in lon/lat | sharp 矩形 |

不同 caller 用不同版本：
- T-ring envelope / collision buffer → `polygon_m`
- `detect_inside_ngz` / `mask_existing_nodes` / `mask_existing_edges` → `polygon_ll`

Densify 之後兩個版本在地球上趨於同一塊區域，這個不一致一併解決。

---

## Recommendation: Densify NGZ polygon edges before metric projection

採用此方案的理由（按 production use case）：

1. **真實天氣 NGZ 已 dense → 對它幾乎是 no-op**（不增 vertex、不影響效能）
2. **測試 / fallback / 手動 NGZ 是 4 角矩形 → 自動補齊，bug 消失**
3. **clearance 用 metric km，符合導航直覺**（5km 就是 5km，不會因緯度縮水）
4. **一石二鳥**：解 bow + 解 polygon_m vs polygon_ll 內部不一致
5. **地圖紅框跟氣象台原始邊界一致**（B 方案會反直覺）

備選方案均不採用，理由見 `docs/ngz-taut_debug.html` Round 2 / Round 4。

---

## Implementation

### 變動 1：新增 densify helper

**檔案**：`routing_map/ngz.py`
**位置**：跟其他 internal helper 放一起（line 96-140 區域，建議放在 `_haversine_km` 後面）

```python
def _densify_polygon_ll(poly: PolyLike, max_segment_deg: float = 0.5) -> PolyLike:
    """在 lon/lat polygon 的每條邊上插中間點，確保每段 length ≤ max_segment_deg。

    防止 4-vertex NGZ 矩形在 AEQD 投影後產生長 metric 直線，造成 polygon_m
    跟 polygon_ll 在地球上對應不同區域（撲克牌彎曲現象）。對已經 dense 的
    polygon（例如從天氣 grid 出來的）幾乎是 no-op。

    輸入可為 Polygon 或 MultiPolygon。回傳同型別；失敗回原 poly。
    """
    if poly is None or poly.is_empty:
        return poly

    def _densify_ring(coords):
        coords = list(coords)
        if len(coords) < 2:
            return coords
        result = [coords[0]]
        for i in range(1, len(coords)):
            a = coords[i - 1]
            b = coords[i]
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            length = math.hypot(dx, dy)
            if length > max_segment_deg:
                n_sub = int(math.ceil(length / max_segment_deg))
                for k in range(1, n_sub):
                    t = k / n_sub
                    result.append((a[0] + t * dx, a[1] + t * dy))
            result.append(b)
        return result

    if isinstance(poly, MultiPolygon):
        try:
            return MultiPolygon([_densify_polygon_ll(p, max_segment_deg) for p in poly.geoms])
        except Exception:
            return poly

    try:
        new_ext = _densify_ring(poly.exterior.coords)
        new_holes = [_densify_ring(h.coords) for h in poly.interiors]
        return Polygon(new_ext, new_holes)
    except Exception:
        return poly
```

不需要新 import：`math` / `Polygon` / `MultiPolygon` 在 `ngz.py` 開頭已 imported。

### 變動 2：加 config 欄位

**檔案**：`routing_map/config.py`
**位置**：`NgzRingBuildConfig` dataclass 結尾（line 71 之後）

```python
@dataclass
class NgzRingBuildConfig:
    """NGZ 專用環建構參數。"""
    clearance_m: float = 5_000.0
    ring_sample_km: float = 5.0
    taut_window_size: int = 16
    taut_max_tries: int = 8
    point_fix_step_m: float = 1_000.0
    point_fix_max_iter: int = 20
    min_island_area_km2: float = 0.0
    min_ring_length_km: float = 0.0
    visibility_k_sea: int = 8
    visibility_k_land_t: int = 4
    visibility_max_dist_km: float = 200.0
    group_merge_eps_m: float = 1_000.0
    densify_max_deg: float = 0.5  # NEW: pre-projection edge densify segment length (degrees)
```

### 變動 3：在 normalize_ngz_inputs 呼叫 densify

**檔案**：`routing_map/ngz.py`
**位置**：`normalize_ngz_inputs` 函式內，目前 line 311-313 區段

**改前**（line 311-313）：
```python
# 4. 投影到 metric，再以 unary_union + buffer(group_merge_eps_m/2) 合併連通群組
metrics: List[Polygon] = [_ensure_valid(geom_to_m(p, proj)) for (_mid, p) in sub_records]
member_ids: List[str] = [mid for (mid, _p) in sub_records]
```

**改後**：
```python
# 4. Densify lon/lat polygon 再投影到 metric。densify 防止 4-vertex 矩形
#    AEQD 失真造成 polygon_m / polygon_ll 不對齊（bow 現象）。
densified_records = [
    (mid, _densify_polygon_ll(p, max_segment_deg=cfg.densify_max_deg))
    for (mid, p) in sub_records
]
metrics: List[Polygon] = [_ensure_valid(geom_to_m(p, proj)) for (_mid, p) in densified_records]
member_ids: List[str] = [mid for (mid, _p) in densified_records]
```

注意：`cfg` 已是 `normalize_ngz_inputs` 參數（line 271 那行 `cfg = cfg or NgzRingBuildConfig()`），直接用即可。

---

## Verification

### 1. Smoke test 必過

```powershell
.venv\Scripts\python.exe ngz_smoke_test.py
```

**預期**：`Total: 15 passed, 3 skipped, 0 failed`

特別注意：
- **N2（off-baseline → result == baseline）**：densify 只影響 NGZ 內部表示，off-baseline 時 NGZ 不相交 path，不會 trigger patching，結果應該還是逐點等於 baseline。
- **N4（concave U）**：densify 後 U 字 NGZ 仍是 U 字（vertex 變多但形狀不變），T-ring 凸殼化效果不受影響。
- **N7（unreachable）**：邏輯路徑沒動，應該照樣 raise。

### 2. 手動驗證 bow 消失

重跑使用者觸發 bug 的場景（OD 與 NGZ 設定見 `docs/NGZ-handover.md` §場景），重新跑 visualization，確認：
- 紫色 taut 緊貼紅色 NGZ 矩形（offset ≈ 5km）
- 沒有 50km 弧形
- 矩形長邊不再「撲克牌彎曲」

### 3. 互動 demo 對照

打開 `docs/ngz-taut_debug.html`，在 Round 2 區段勾選「densify」checkbox，觀察弧形消失。實作後實際 NGZ T-ring 應該對應 checkbox 勾選 ON 的狀態。

---

## 不要動的東西

- **`geom_to_m` / `geom_to_ll` / `AOIProjector`**：這是 shapely 標準 transform，動了會打到陸地 pipeline 跟整個 cached graph
- **`taut_window_size`**：Round 1 假設「window 太小造成弧」其實不對，根因是 AEQD 失真不是 window；保持 16
- **`polygon_ll` 任何 caller**：densify 後 `polygon_m` 跟 `polygon_ll` 都自動跟著變多 vertex，shapely 自動處理
- **`aoi_cache/`**：禁止覆寫（CLAUDE.md §2.3）

---

## Critical files map

| 檔案 | 動作 | 大約行數 |
|---|---|---|
| `routing_map/ngz.py` | 新增 `_densify_polygon_ll` helper + 在 `normalize_ngz_inputs` 呼叫 | ~30 行新增 + 3 行替換 |
| `routing_map/config.py` | `NgzRingBuildConfig` 加 `densify_max_deg` 欄位 | 1 行 |

---

## References

- `docs/ngz-taut_debug.html` Round 2 / Round 3 / Round 4：完整證據鏈與互動演示
- `docs/NGZ flow.md`：原 NGZ pipeline 完整流程
- `docs/NGZ-handover.md`：上一輪 PR 交接 + invariants 不可破壞清單
- `docs/ngz_debug_file/Taut ring issue 1.png` / `2.png`：bug 觀察截圖
- `CLAUDE.md`：專案守則（特別 §2.3 cache 不可動、§2.4 node idx 不可變）
