# Route Planner Debug UI

一站式 web UI 工具：點擊地圖設原點 / 終點 + 任意 NGZ polygon → 跑 `run_p2p` → 即時看路徑與 pipeline log。

## 啟動

確認 `.venv` 已 activate 並裝好 `fastapi` / `uvicorn`（見 `requirements.txt`）：

```powershell
.venv\Scripts\python.exe -m uvicorn debug_ui.server:app --reload --port 8000
```

開瀏覽器 `http://localhost:8000`。

啟動需 10–30 秒（載 AOI cache + G_base）。startup log 會印出 `[startup] cache loaded in X.Xs ...`。

## 用法

1. **Set Origin** → 點地圖；或 toolbar 左下 lon/lat 欄輸入 → **Apply 套用**
2. **Set Dest** 同理
3. **Add NGZ** → 點地圖上 ≥3 個 vertex、雙擊收尾（可重複新增多個）
4. **Clear All** → 重置 markers / NGZ / route / stats / log + 清空座標欄
5. **▶ Run Route** → POST `/api/route` 後地圖畫路徑、stats 顯示距離 / 點數、Pipeline Log 顯示後端 `run_p2p` 的 debug output

## NGZ Mode 對照

| Mode | 行為 |
|---|---|
| `lenient` | origin / dest 在 NGZ 內時，該 NGZ 對該端點視為不擋（不 raise） |
| `strict` | origin / dest 在 NGZ 內時，直接 raise 錯誤 |
| `relocate` | origin / dest 移到該 NGZ 最近的 T-ring 頂點 |

## API

- `GET /api/init` → `{bbox, center, zoom, ready}`，前端啟動時設 map view
- `POST /api/route` → 傳 `{origin, dest, ngz_polygons[], ngz_mode}`，回 `{path_final, path_raw, length_km, n_points, n_ngz, error, log}`
  - query 級錯誤回 `200` + `error` 欄（**不 raise 500**），方便前端統一處理

## 注意

- Server 只 **read** `aoi_cache/`，不寫；每個 request 用 `G_base.copy()` 餵 `run_p2p`，避免 `inject_point_edges` 污染 cache 圖
- 沒做 SSE / WebSocket，log 是 `redirect_stdout` capture 完再一次性回傳
- NGZ `clearance_m` 寫死 10 km；要改去 `debug_ui/server.py` 的 `ngz_cfg`
