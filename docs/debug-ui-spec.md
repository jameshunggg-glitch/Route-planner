# 路徑規劃 Debug UI — 實作規格書 v1.0

> **目標讀者**：負責實作這個 UI 的下一個 agent。看完本檔 + `Route_Planner.ipynb` cell 20 應該能直接動工。
>
> **設計來源**：`docs/debug-ui-mockup.html` v0.2（user 已 review 拍板）。本 spec 是 mockup 的「production 翻譯」+ 後端細節。
>
> **不要做**：超出本 spec 範圍的事。MVP 後續項目見最後 §「Out of scope」。

---

## 1. Overview

**做什麼**：一站式 web UI，提供
- 滑鼠點擊 + 文字輸入兩種方式設 origin / dest
- 滑鼠點擊勾出任意 polygon、複數個 NGZ
- 一鍵呼叫 `run_p2p` → 同畫面顯示結果路徑、stats、pipeline log

**雙重用途**：對內 debug 工具 + 對外 demo / 提案展示。

**痛點**：現行 `ngz_testing_set.txt` 5 個 test case 都靠手敲座標、易輸錯、外部 demo 不直觀。

---

## 2. Architecture

```
┌────────────────────────────┐       HTTP/JSON       ┌─────────────────────────────┐
│ Browser                    │                       │ FastAPI server              │
│  Leaflet 1.9 + Draw 1.0    │  POST /api/route ────▶│  startup: 載 AOI cache + G  │
│  Left toolbar / Map        │                       │  /api/route → run_p2p       │
│  Stats / Pipeline Log      │  GET /api/init  ◀──── │  /api/init → bbox center    │
│                            │     {path, log, …}    │                             │
└────────────────────────────┘                       └─────────────────────────────┘
       GET /            static files
```

**Server-rendered first paint = 不需要**。前端就是純靜態 HTML/JS/CSS，從 server 用 StaticFiles 餵。

---

## 3. File structure

新增（全部新檔）：

```
debug_ui/
├── server.py              # FastAPI app
├── static/
│   ├── index.html         # 主頁骨架
│   ├── app.js             # 互動邏輯
│   └── style.css          # 樣式
└── README.md              # 啟動指令 + 用法
```

修改：
- `requirements.txt` — 加 `fastapi` + `uvicorn[standard]`（已和 user 確認，CLAUDE.md §4 已同意）

---

## 4. Dependencies

### Python（新增到 `requirements.txt`，跟既有同格式 `name==X.Y.Z`）

| Package | 用途 | 版本建議 |
|---|---|---|
| `fastapi` | Web framework | latest stable（pip 解析當下版本） |
| `uvicorn[standard]` | ASGI server | latest stable |

裝法：`.venv\Scripts\pip install fastapi "uvicorn[standard]"` 後 `pip freeze` 拿到實際版號回填 `requirements.txt`。

### Frontend（CDN，無 pip）

| Lib | CDN URL |
|---|---|
| Leaflet 1.9.4 | `https://unpkg.com/leaflet@1.9.4/dist/leaflet.css` + `.js` |
| Leaflet.Draw 1.0.4 | `https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css` + `.js` |

---

## 5. Backend spec (`debug_ui/server.py`)

### 5.1 Imports（最小集合）

```python
from __future__ import annotations
import gzip
import io
import os
import pickle
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from routing_map.pipeline import (
    run_p2p, GraphConfig, SnapConfig, SimplifyConfig, RunConfig,
)
from routing_map.repairer import RepairConfig
from routing_map import NgzRingBuildConfig
from routing_map.geom_utils import build_projector_from_bbox
```

### 5.2 Cache loading + hydrate（**參考 `Route_Planner.ipynb` cell 20**）

```python
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "aoi_cache"
OUT_PATH = CACHE_DIR / "out_global.pkl.gz"
G_PATH = CACHE_DIR / "G_global.pkl.gz"

# Globals 駐留 startup 期間
_state: Dict[str, Any] = {"out": None, "G_base": None, "ready": False}


def _load_gz(p: Path):
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def _hydrate_out(out: Dict[str, Any]) -> Dict[str, Any]:
    """Cell 20 pattern: 補回 proj 跟 sea_kdt（pickle 卸下後可能是 None）。"""
    if out.get("proj") is None:
        out["proj"] = build_projector_from_bbox(out["bbox_ll"])
    if out.get("sea_kdt") is None:
        S = out["S_nodes"]
        out["sea_kdt"] = cKDTree(S[["x_m", "y_m"]].to_numpy(dtype=float))
    return out
```

### 5.3 Startup hook

```python
app = FastAPI(title="路徑規劃 Debug UI")


@app.on_event("startup")
async def _startup() -> None:
    print(f"[startup] loading cache from {CACHE_DIR}...", flush=True)
    t0 = time.perf_counter()
    out = _load_gz(OUT_PATH)
    G_pack = _load_gz(G_PATH)
    G_base = G_pack["G_base"] if isinstance(G_pack, dict) and "G_base" in G_pack else G_pack
    out = _hydrate_out(out)
    _state["out"] = out
    _state["G_base"] = G_base
    _state["ready"] = True
    print(
        f"[startup] cache loaded in {time.perf_counter()-t0:.1f}s "
        f"(S_nodes={len(out['S_nodes'])}, G_base nodes={G_base.number_of_nodes()})",
        flush=True,
    )
```

**重點**：`G_base` 在 startup 載一次，後續 request 用 `G_base.copy()` 傳給 `run_p2p`（避免 `inject_point_edges` 污染 cache 圖）。

### 5.4 Pydantic models

```python
class LonLatModel(BaseModel):
    lon: float = Field(..., ge=-180, le=180)
    lat: float = Field(..., ge=-90, le=90)


class NgzPolygonModel(BaseModel):
    points: List[Tuple[float, float]] = Field(..., min_length=3)
    # 注意：tuple 是 (lon, lat)


class RouteRequest(BaseModel):
    origin: LonLatModel
    dest: LonLatModel
    ngz_polygons: List[NgzPolygonModel] = Field(default_factory=list)
    ngz_mode: Literal["lenient", "strict", "relocate"] = "lenient"


class RouteResponse(BaseModel):
    path_final: List[Tuple[float, float]]  # (lon, lat) list
    path_raw: List[Tuple[float, float]] = []  # patched baseline（debug 用）
    length_km: Optional[float] = None
    n_points: int = 0
    n_ngz: int = 0
    error: Optional[str] = None
    log: str = ""  # captured stdout from run_p2p


class InitResponse(BaseModel):
    bbox: Tuple[float, float, float, float]  # (minLon, minLat, maxLon, maxLat)
    center: Tuple[float, float]              # (lon, lat)
    zoom: int = 5
    ready: bool
```

### 5.5 Endpoints

```python
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/init", response_model=InitResponse)
async def init() -> InitResponse:
    if not _state["ready"]:
        raise HTTPException(503, "cache not loaded yet")
    bbox = _state["out"]["bbox_ll"]  # (minLon, minLat, maxLon, maxLat)
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return InitResponse(bbox=bbox, center=center, zoom=5, ready=True)


@app.post("/api/route", response_model=RouteResponse)
async def route(req: RouteRequest) -> RouteResponse:
    if not _state["ready"]:
        raise HTTPException(503, "cache not loaded yet")
    out = _state["out"]
    G_base = _state["G_base"]

    polys = [Polygon(ngz.points) for ngz in req.ngz_polygons]

    # 配 cfg（debug=True 讓 [pipeline][...] print 出來給前端 log）
    graph_cfg = GraphConfig(bbox_ll=out["bbox_ll"])
    snap_cfg = SnapConfig()
    repair_cfg = RepairConfig(debug=False)
    simplify_cfg = SimplifyConfig(enabled=True)
    run_cfg = RunConfig(do_repair=True, do_simplify=True, debug=True)
    ngz_cfg = NgzRingBuildConfig(clearance_m=10_000.0)

    # 捕 stdout（同時 capture stderr 防 ring_taut.py logging.warning）
    buf = io.StringIO()
    err_str: Optional[str] = None
    res = None
    t0 = time.perf_counter()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            res = run_p2p(
                out, (req.origin.lon, req.origin.lat), (req.dest.lon, req.dest.lat),
                graph_cfg=graph_cfg, snap_cfg=snap_cfg,
                repair_cfg=repair_cfg, simplify_cfg=simplify_cfg, run_cfg=run_cfg,
                G_in=G_base.copy(),
                ngz_polygons=polys if polys else None,
                ngz_mode=req.ngz_mode,
                ngz_cfg=ngz_cfg,
            )
        if res.error:
            err_str = res.error
    except Exception as e:
        err_str = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    elapsed = time.perf_counter() - t0
    log_str = buf.getvalue() + f"\n[server] elapsed {elapsed:.2f}s"

    path_final = list(res.path_ll_final) if (res and res.path_ll_final) else []
    path_raw = list(res.path_ll_raw) if (res and res.path_ll_raw) else []
    length_km = (res.lengths_km or {}).get("final") if res else None

    return RouteResponse(
        path_final=[(float(lo), float(la)) for (lo, la) in path_final],
        path_raw=[(float(lo), float(la)) for (lo, la) in path_raw],
        length_km=length_km,
        n_points=len(path_final),
        n_ngz=len(polys),
        error=err_str,
        log=log_str,
    )


# Static mount 放在 endpoint 之後
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
```

### 5.6 啟動指令

```powershell
.venv\Scripts\python.exe -m uvicorn debug_ui.server:app --reload --port 8000
```

開瀏覽器 `http://localhost:8000`。

---

## 6. Frontend spec

### 6.1 HTML 結構（`static/index.html`）

骨架完全照 `docs/debug-ui-mockup.html` v0.2 的 `<body>` 內容**直接拿來用**（mockup 已 lock-in 樣式）。差別只在：
1. Header 文字改為 production 版（不再寫「Mockup v0.2 預覽」）
2. 移除「試用建議」`.intro` 區塊
3. 移除 footer 的「Mockup v0.2」字樣
4. 改成從 `/static/style.css` 載 CSS、`/static/app.js` 載 JS（mockup 是 inline）

CSS / JS link：
```html
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css">
<link rel="stylesheet" href="/static/style.css">
...
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
<script src="/static/app.js"></script>
```

### 6.2 CSS（`static/style.css`）

**完全照 mockup v0.2 `<style>` 內容拷貝**。下表是 lock-in 設計值的 lookup：

#### 色彩（CSS variables）

| Token | Value | 用途 |
|---|---|---|
| `--bg` | `#ffffff` | 全域底色 |
| `--fg` | `#212121` | 主文字 |
| `--muted` | `#757575` | 次要文字、label |
| `--panel-bg` | `#f7f8fa` | 面板底色（app-shell、header bar） |
| `--border` | `#e0e0e0` | 邊框 |
| `--primary` | `#1976d2` | 主色（按鈕、focus） |
| `--primary-hover` | `#1565c0` | 主按鈕 hover |
| `--danger` | `#d32f2f` | error / invalid input |
| `--ok` | `#388e3c` | success |

#### 地圖元素配色（**寫死、不暴露 picker**）

| 元素 | 顏色 | 補充 |
|---|---|---|
| Origin marker 圓點 | `#2e7d32` | 26×26 div, label 'O' (white) |
| Dest marker 圓點 | `#c62828` | 26×26 div, label 'D' (white) |
| Route polyline | `#1976d2` | weight 4, opacity 0.9 |
| NGZ polygon stroke | `#e57373` | weight 2 |
| NGZ polygon fill | `#e57373` @ opacity 0.25 | — |

#### Log panel 暗色色板

| Token | Value | 用途 |
|---|---|---|
| Log bg | `#1e1e1e` | 終端機底色 |
| Log fg (default) | `#d4d4d4` | 一般文字 |
| `.tag` | `#569cd6` | `[pipeline][...]` 標籤 |
| `.num` | `#b5cea8` | 數字 highlight |
| `.ok` | `#4ec9b0` | ✓ 完成訊息 |
| `.warn` | `#ce9178` | warning（如 fallback 訊號） |
| `.err` | `#f48771` | error |
| `.dim` | `#808080` | dim / placeholder |

#### 字型

```css
/* UI 主字型（會 fallback 到系統字型，無需 webfont） */
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
             "PingFang TC", "Microsoft JhengHei", sans-serif;

/* 等寬字（log panel、coord input、數字） */
font-family: ui-monospace, "Consolas", "Menlo", "Monaco", monospace;
```

#### 尺寸 / Layout

| 元素 | 規格 |
|---|---|
| Body | `max-width: 1400px`, padding 24px |
| App-shell | grid `240px 1fr`, rows `520px`, gap 12px |
| Toolbar 寬 | 240px（內含 padding） |
| Map min-height | 520px |
| Stats / Log panel | 12px margin-top, padding 12px |
| Marker icon | 26×26 圓 + 3px white border + box-shadow `0 0 6px rgba(0,0,0,0.4)` |
| 一般 button height | ~36px（8px padding + 13px font） |
| Primary button height | ~42px |
| Log max-height | 280px（scrollable）|

### 6.3 JavaScript (`static/app.js`)

**主要邏輯照 mockup v0.2 `<script>` 內容**，差別在「Run Route」改成真的 POST 後端。

#### State

```js
let mode = null;           // 'origin' | 'dest' | null
let originMarker = null;
let destMarker = null;
const ngzLayers = L.featureGroup().addTo(map);
let routeLayer = null;
let routeRawLayer = null;  // optional: 顯示 patched raw path（debug 用，MVP 不一定要）
```

#### Tile options（**寫死 2 個**）

```js
const tileOptions = {
  'cartodb-light': {
    url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    attr: '© CartoDB'
  },
  'esri-sat': {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr: '© Esri'
  }
};
// default = 'cartodb-light'
```

#### Marker icon factory

```js
function mkIcon(color, label) {
  return L.divIcon({
    html: `<div style="background:${color};width:26px;height:26px;border-radius:50%;border:3px solid white;box-shadow:0 0 6px rgba(0,0,0,0.4);display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:12px">${label||''}</div>`,
    className: '',
    iconSize: [26, 26],
    iconAnchor: [13, 13]
  });
}
```

#### 啟動：fetch `/api/init`，把 bbox center 設成 map.setView

```js
async function initView() {
  try {
    const r = await fetch('/api/init');
    if (!r.ok) throw new Error(`init failed: ${r.status}`);
    const j = await r.json();
    map.setView([j.center[1], j.center[0]], j.zoom);
  } catch (e) {
    console.warn('init failed, fallback view', e);
    map.setView([28, 135], 5);
  }
}
initView();
```

#### Map click handler

```js
map.on('click', e => {
  if (mode === 'origin') {
    setOrigin(e.latlng);
    setMode(null);
  } else if (mode === 'dest') {
    setDest(e.latlng);
    setMode(null);
  }
});
```

#### Bidirectional sync（marker ↔ coord input）

- `setOrigin(latlng)` / `setDest(latlng)` 後呼叫 `syncCoordInputs()` 把 input fields 填上 `lng.toFixed(4)` / `lat.toFixed(4)`
- Coord input Apply 按下 → 驗證範圍（lon −180~180、lat −90~90）→ invalid 加 `.invalid` class（紅底框）→ valid 創 marker

#### NGZ draw

```js
btnNgz.addEventListener('click', () => {
  setMode(null);
  new L.Draw.Polygon(map, {
    shapeOptions: {
      color: '#e57373', fillColor: '#e57373',
      weight: 2, fillOpacity: 0.25
    }
  }).enable();
});

map.on(L.Draw.Event.CREATED, e => {
  ngzLayers.addLayer(e.layer);
  document.getElementById('stat-ngz').textContent = ngzLayers.getLayers().length;
});
```

#### Run handler（**真實 fetch，不要 mock**）

```js
btnRun.addEventListener('click', async () => {
  const statusEl = document.getElementById('stat-status');
  const errEl = document.getElementById('stat-error');
  if (!originMarker || !destMarker) {
    statusEl.textContent = '請先設 Origin 與 Dest';
    statusEl.className = 'v err';
    return;
  }
  statusEl.textContent = '執行中...';
  statusEl.className = 'v';
  errEl.textContent = '—';
  errEl.className = 'v';
  clearLog();
  logLineHTML('<span class="dim">--- POST /api/route ---</span>');

  const oLL = originMarker.getLatLng();
  const dLL = destMarker.getLatLng();
  const ngzs = ngzLayers.getLayers().map(layer => {
    const ring = layer.getLatLngs()[0];
    return { points: ring.map(p => [p.lng, p.lat]) };
  });

  const body = {
    origin: { lon: oLL.lng, lat: oLL.lat },
    dest:   { lon: dLL.lng, lat: dLL.lat },
    ngz_polygons: ngzs,
    ngz_mode: document.getElementById('ngz-route-mode').value
  };

  try {
    const r = await fetch('/api/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const j = await r.json();

    // 1. log: 把 j.log 解析 + 著色 + 渲染
    renderLog(j.log);

    if (j.error) {
      statusEl.textContent = '✗ Error';
      statusEl.className = 'v err';
      errEl.textContent = j.error.split('\n')[0];  // 第一行
      errEl.className = 'v err';
    } else {
      // 2. 畫路徑
      if (routeLayer) map.removeLayer(routeLayer);
      routeLayer = L.polyline(
        j.path_final.map(([lon, lat]) => [lat, lon]),
        { color: '#1976d2', weight: 4, opacity: 0.9 }
      ).addTo(map);

      statusEl.textContent = '✓ 完成';
      statusEl.className = 'v ok';
      document.getElementById('stat-npts').textContent = j.n_points;
      document.getElementById('stat-dist').textContent = j.length_km != null ? j.length_km.toFixed(1) : '—';
      errEl.textContent = 'None';
      errEl.className = 'v ok';
    }
  } catch (e) {
    statusEl.textContent = '✗ Network error';
    statusEl.className = 'v err';
    errEl.textContent = String(e);
    errEl.className = 'v err';
  }
});
```

#### Log 渲染：把 server 回傳的 log string 著色

```js
function renderLog(logStr) {
  // 接 server 的 mock：先 clear placeholder
  if (logOut.querySelector('.placeholder')) logOut.innerHTML = '';

  const lines = (logStr || '').split('\n');
  for (const line of lines) {
    let html = escapeHtml(line);
    // [pipeline][xxx] 染色
    html = html.replace(/(\[pipeline\]\[\w+\])/g, '<span class="tag">$1</span>');
    html = html.replace(/(\[server\])/g, '<span class="dim">$1</span>');
    // 數字（簡單版：=後跟數字）
    html = html.replace(/(=)(\d+)/g, '$1<span class="num">$2</span>');
    // ✓ / ✗
    if (line.includes('✓')) html = `<span class="ok">${html}</span>`;
    else if (line.includes('✗') || /Error|Exception|Traceback/i.test(line)) html = `<span class="err">${html}</span>`;
    else if (/warn|warning|fallback/i.test(line)) html = `<span class="warn">${html}</span>`;
    logLineHTML(html);
  }
}

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
```

**注意**：因為 server 是 `redirect_stdout` 一次性 capture 全部，**前端不需要 streaming**，直接 dump 渲染就好。MVP 不必 SSE / WebSocket。

#### Clear All

清掉 marker / NGZ / route / stats / log（log 還原 placeholder）+ 清空 4 個 coord input。

---

## 7. API contract（給 frontend / debugging）

### 7.1 `GET /api/init`

Response (`200`):
```json
{
  "bbox": [120.0, 18.0, 150.0, 40.0],
  "center": [135.0, 29.0],
  "zoom": 5,
  "ready": true
}
```

`503` if cache 尚未載完。

### 7.2 `POST /api/route`

Request body:
```json
{
  "origin": {"lon": 139.0, "lat": 32.0},
  "dest":   {"lon": 131.0, "lat": 23.0},
  "ngz_polygons": [
    {"points": [[137.5, 31.0], [140.0, 29.0], [132.5, 25.0], [132.5, 31.0]]}
  ],
  "ngz_mode": "lenient"
}
```

Response (`200`):
```json
{
  "path_final": [[139.0, 32.0], [136.2, 30.1], ..., [131.0, 23.0]],
  "path_raw": [[...]],
  "length_km": 1284.5,
  "n_points": 18,
  "n_ngz": 1,
  "error": null,
  "log": "[pipeline][graph] nodes=...\n[pipeline][astar] ...\n..."
}
```

Error case (200 with `error` set；**不 raise HTTP 500**，因為 query 級錯誤算正常情境）:
```json
{
  "path_final": [],
  "path_raw": [],
  "length_km": null,
  "n_points": 0,
  "n_ngz": 1,
  "error": "snap_failed: ...",
  "log": "[pipeline][graph] ...\n[pipeline][snap_inject_error] ..."
}
```

---

## 8. Verification

### 8.1 啟動

```powershell
.venv\Scripts\python.exe -m uvicorn debug_ui.server:app --reload --port 8000
```

預期 stdout：
```
[startup] loading cache from ...\aoi_cache...
[startup] cache loaded in X.Xs (S_nodes=11062, G_base nodes=12345)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

打開 `http://localhost:8000`，左 toolbar + 地圖 + 下方 stats + log placeholder 應全部就位。

### 8.2 跑 `ngz_testing_set.txt` 五個 test case

依檔內參數：

| Test | Origin | Dest | NGZ rect 點 |
|---|---|---|---|
| 1 | (139, 32) | (131, 23) | (137.5,31),(140,29),(132.5,25),(132.5,31) |
| 2 | (139, 32) | (131, 23) | (137.5,33),(137.5,25),(132.5,25),(132.5,31) |
| 3 | (139, 32) | (131, 23) | (137.5,33),(137.5,21),(132.5,21),(132.5,31) |
| 4 | (139, 32) | (131, 23) | (137.5,35),(137.5,21),(132.5,21),(132.5,35) |
| 5 | (139, 32) | (131, 23) | 5 vertex（concave，見檔內） |

**每個 test 都要試**：
- 用滑鼠點 + 文字輸入兩種設定方式
- 切換 NGZ Mode（lenient/strict/relocate）看行為差異
- 切換底圖（Carto Light ↔ Esri Satellite）
- Pipeline Log 區應顯示 `[pipeline][...]` 一系列訊息

### 8.3 錯誤狀況 verify

- Lon = 200 → input 變紅、Apply 失效
- Origin 點陸地上 → stats Error 欄顯示 `snap_failed`、log 區有完整錯誤訊息
- NGZ 完全包住 origin（strict 模式）→ error 顯示 `ngz_inside_origin: ...`
- Multi-NGZ 同段擋 → log 顯示 `pairwise_visibility=True` 或類似訊息

### 8.4 視覺驗證

- Mockup（`docs/debug-ui-mockup.html` v0.2）打開、production UI 打開並排
- 排版、配色、字型、行為應**一致**（mockup 是設計來源）

---

## 9. 重要 invariants（**不要破壞**）

從 `CLAUDE.md` 與 `docs/NGZ-handover.md`：

1. **`aoi_cache/` 禁止覆寫**（CLAUDE.md §2.3）— server 只 read，不 write
2. **不增刪 G_base nodes**（CLAUDE.md §2.4）— `G_base.copy()` 給 `run_p2p`，後者 `inject_point_edges` 只動 copy
3. **NGZ off-baseline 結果 == baseline**（N2 invariant）— server 行為應對 NGZ 不擋路 query 保持原 baseline 結果
4. **本 spec 不動 `routing_map/` 內任何檔**（純 read import）
5. **Co-Authored-By line in commits**（如最後要 commit）

---

## 10. README.md（`debug_ui/README.md`）內容建議

```markdown
# Route Planner Debug UI

一站式 web UI 工具：點擊地圖設原點/終點 + 任意 NGZ polygon → 跑 run_p2p → 即時看路徑與 pipeline log。

## 啟動

確認 .venv 已 activate 並裝好 fastapi / uvicorn（見 requirements.txt）:

```powershell
.venv\Scripts\python.exe -m uvicorn debug_ui.server:app --reload --port 8000
```

開瀏覽器 `http://localhost:8000`。

啟動需 10-30 秒（載 AOI cache + G）。

## 用法

1. **Set Origin** 按鈕 → 點地圖 / 或直接在 toolbar 內 lon/lat 欄輸入 → Apply
2. **Set Dest** 同理
3. **Add NGZ** → 點地圖上 ≥3 個 vertex、雙擊收尾（可重複新增多個）
4. **Clear All** 重置
5. **Run Route** ▶ → 後端跑 `run_p2p`，地圖畫路徑、stats 顯示距離、Pipeline Log 顯示 debug output

## NGZ Mode 對照

| Mode | 行為 |
|---|---|
| lenient | origin/dest 在 NGZ 內時，NGZ 不擋路徑 |
| strict | origin/dest 在 NGZ 內時，raise 錯誤 |
| relocate | origin/dest 移到最近 T-ring 頂點 |
```

---

## 11. Out of scope（MVP 後續，**不要**做）

- Save / load scenario JSON
- Dark 主題切換
- Baseline / repaired / patched 切換圖層（debug overlay）
- T-ring / repair midpoints 視覺化 overlay
- 多 user / 認證
- 雲端部署（Streamlit Cloud / Vercel / etc.）
- WebSocket / SSE 即時 log streaming（MVP 用 stdout capture 一次性回傳）
- 對外 demo 視覺打磨（logo / theme / 動畫）
- 自訂 NGZ clearance（目前寫死 10km）
- Coord input 多語言、單位切換

---

## 12. References

| 文件 | 用途 |
|---|---|
| `Route_Planner.ipynb` **cell 20** | Cache 載入 + hydrate + `run_p2p` 呼叫的完整 reference 範例（最重要） |
| `docs/debug-ui-mockup.html` v0.2 | UI 視覺設計來源（樣式、layout、行為皆以此為準） |
| `ngz_testing_set.txt` | 5 個 test case 用來驗收 |
| `CLAUDE.md` §2 §4 §6 §8 | 修改範圍 / dep / git / 回答風格守則 |
| `docs/NGZ-handover.md` | NGZ pipeline 整體 invariants |
| `routing_map/pipeline.py:239` | `run_p2p` 完整 signature |
| `routing_map/cache_utils.py` | 高階 cache API（替代 cell 20 的 `_load_gz`，可選） |

---

## 13. 完成標準

- [ ] `requirements.txt` 加 `fastapi` + `uvicorn[standard]` 兩行（鎖定版號）
- [ ] `debug_ui/{server.py, static/index.html, static/app.js, static/style.css, README.md}` 五檔到位
- [ ] `uvicorn` 起得來、startup log 顯示 cache loaded
- [ ] `http://localhost:8000` 開得起來、UI 看起來 = mockup v0.2
- [ ] test set 1 跑得通：Run 之後地圖顯示繞 NGZ 路徑、stats 顯示距離 / 點數、log 顯示 `[pipeline][...]` 訊息
- [ ] 手動輸入座標也能設 marker（4 input + Apply）
- [ ] 滑鼠點 marker 會回填到 coord input
- [ ] Clear All 重置乾淨
- [ ] 底圖切換 Carto Light ↔ Esri Satellite 即時生效
- [ ] NGZ mode 切換生效（lenient/strict/relocate）
- [ ] 沒動 `routing_map/`、`aoi_cache/`、`ngz_smoke_test.py`、`Route_Planner.ipynb` 任何一個

完成後跑 `git status` 確認改動範圍 = 上述 5 個新檔 + `requirements.txt` 修改。**不 commit / push**，等 user review。
