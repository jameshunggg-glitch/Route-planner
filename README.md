# Route Planner

模組化的**海上 / 沿岸路徑規劃**工具。給定起點與終點的經緯度，輸出一條避開陸地、可選擇性繞開高緯度與禁航區的航線。

底層由兩層圖組合而成：

- **Coastal layer** — 沿陸地邊界生成的環狀圖（E-ring + T-ring），處理近岸繞行。
- **Sea layer** — 來自 [`scgraph`](https://pypi.org/project/scgraph/) 的深海航網，處理跨洋遠距。

兩層之間以 **E/T transfer** 與 **T-gate connectors** 銜接。

> ⚠️ Status：研究 / 開發中。本專案最初在較緊湊的時程下完成，部分模組仍有未清理的試驗痕跡。模組之間的 API 並未穩定，僅建議從 notebook 的上層介面使用。

---

## Features

- **AOI Pipeline** — 從陸地 shapefile 一路建構到 routing graph，可快取整段結果。
- **Point-to-Point 路徑規劃** — A\* 搜尋 + 路徑修復 + 可見性簡化。
- **Multi-world 路由** — 對 origin / destination 各自嘗試 ring 或 sea 兩種策略，自動挑最佳組合。
- **Edge Hotfix 工具** — 對已建好的 cache 圖做手動補丁（封鎖航道、新增 / 修權重），不必整段重跑。
- **Folium 視覺化** — 環、節點、最終路徑都可輸出成互動式 HTML 地圖。

---

## 核心概念

| 名詞 | 說明 |
|------|------|
| **AOI** | Area of Interest，由 bbox 定義的查詢範圍；環、節點、邊都在這個框內。 |
| **E-ring** | Envelope ring，沿陸地邊界向外 buffer 的「保守安全環」，頂點密集。 |
| **T-ring** | Taut ring，E-ring 的可見性簡化版本；保留關鍵頂點，節點數較少，主要供長段使用。 |
| **C-chain** | 沿環均勻取樣的節點鏈（間距 `c_step_km`），輔助 snap 時的 nudge。 |
| **E/T transfer** | E-ring 與 T-ring 之間的橋接邊（shared pairs + ramp anchors）。 |
| **T-gate** | T-ring 上被標為「閘候選」的節點，作為近岸 ↔ 深海的出入口。 |
| **Sea node** | 來自 `scgraph` 的深海節點。 |
| **T-gate sea connector** | T-gate 直接連到 sea node 的邊，含碰撞檢查。 |
| **Multi-world** | 一條 OD 同時嘗試「ring 起、ring 終」、「ring 起、sea 終」⋯等多種策略，再選最佳。 |

---

## Pipeline 概覽

```
shapefile (land)
    ↓
build_aoi(cfg)
    ├─ 1. Land + Layers          (land_layers.py, smooth.py)
    ├─ 2. E/T Rings              (rings.py, ring_graph.py)
    ├─ 3. C-chain                (cchain.py)
    ├─ 4. E/T Transfer           (e_t_transfer_v2.py)
    ├─ 5. Sea subnet (scgraph)   (scgraph_bridge.py, sea_nodes.py)
    └─ 6. T-gate connectors      (t_gate_connectors.py)
    ↓
out (dict) + G (NetworkX graph)
    ↓                   ↓
cache_utils.save → aoi_cache/*.pkl.gz
                    ↓
            pipeline.run_p2p_multiworld(origin, dest, policy)
                    ├─ snap (snap.py)            最近 k 個節點 + nudge
                    ├─ inject                     虛擬 Q:START / Q:END
                    ├─ A* 搜尋                    haversine heuristic
                    ├─ repair (repairer.py)      碰撞修補（midpoint detour / rubberband）
                    └─ simplify                   可見性貪心簡化
                    ↓
            RouteResult（節點序列、總長、診斷資訊）
```

---

## 安裝

需要 **Python 3.10+**（建議用專案內的 `.venv`）。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` 是凍結版（含完整版本號）。關鍵依賴：

- `networkx`、`shapely`、`geopandas`、`pyproj` — 圖與幾何
- `scgraph`、`searoute` — 深海航網
- `folium` — 互動地圖
- `scipy` — KDTree 空間索引

需要外部資料：

- 陸地 shapefile（推薦 GSHHS 高解析或 Natural Earth）。路徑由 `LandConfig.shp_path` 指定，請依本機環境調整。

---

## Quick Start

主要工作流走 `Route_Planner.ipynb`。建議流程：

### 第一次跑：建 cache

```python
from routing_map import RoutingMapConfig, build_aoi
from routing_map.config import AoiConfig, LandConfig
from routing_map import cache_utils

cfg = RoutingMapConfig(
    aoi=AoiConfig(bbox_ll=(100.0, -10.0, 150.0, 40.0), pad_deg=2.0),
    land=LandConfig(shp_path="path/to/land.shp", buffer_km=20.0, avoid_km=5.0),
)

out = build_aoi(cfg)                       # 視 AOI 大小可能跑數分鐘到數十分鐘
cache_utils.save(out, "aoi_cache/")        # 存 G_global.pkl.gz + out_global.pkl.gz
```

### 之後：載 cache 直接規劃

```python
from routing_map import cache_utils
from routing_map.pipeline import run_p2p_multiworld, RoutePolicy

out, G = cache_utils.load("aoi_cache/")

policy = RoutePolicy(hard_lat_cap_deg=70.0)  # 例如禁北極

origin = (121.5, 25.0)   # (lon, lat)
dest   = (139.7, 35.7)

result = run_p2p_multiworld(out, G, origin, dest, policy=policy)
print(result.length_km, len(result.path_ll))
```

實際上建議先打開 `Route_Planner.ipynb` 跑前面幾個 cell，把 cache 載入後，再依需要修改 OD 與 policy。

---

## Typical Workflow

```
┌─ 第一次：build_aoi → 存 cache（一次性，較慢）
│
├─ 日常：載 cache → run_p2p_multiworld → 看 HTML 地圖
│         ↓
│      路徑經過禁航區或不合理的窄水道？
│         ↓
└─ Hotfix：Scgraph_hotfixer.ipynb 找出問題邊
              ↓
           apply_edge_hotfix.py 套用 patch
              ↓
           產生帶日期 suffix 的新 cache（原檔保留）
              ↓
           重新規劃驗證
```

---

## 專案結構

```
Route Planner/
├── routing_map/                  # 主套件
│   ├── build_aoi.py              # ★ AOI pipeline 主入口
│   ├── pipeline.py               # ★ run_p2p / run_p2p_multiworld
│   ├── config.py                 # 所有 dataclass config
│   ├── routing_graph.py          # NetworkX 基礎圖（layer / ban mask）
│   ├── rings.py, ring_graph.py   # E/T ring 構築與環內節點/邊
│   ├── ring_taut.py, ring_envelope.py
│   ├── cchain.py                 # C-chain 採樣
│   ├── e_t_transfer_v2.py        # E ↔ T 橋接邊（v2，舊版 e_t_transfer.py 仍保留）
│   ├── t_gate_connectors.py      # T-gate → sea 連線
│   ├── sea_nodes.py              # 海洋節點 KDTree / DataFrame
│   ├── scgraph_bridge.py         # scgraph 包裝
│   ├── snap.py, snap_link_repair.py  # OD 點吸附與注入
│   ├── repairer.py               # 路徑碰撞修復
│   ├── path_simplifier.py        # 可見性簡化
│   ├── visibility.py             # 線段碰撞檢查
│   ├── land_layers.py, io_land.py, smooth.py  # 陸地處理
│   ├── geom_utils.py, metrics.py, types.py    # 幾何 / 度量 / 型別
│   ├── cache_utils.py            # cache 存取
│   └── viz_*.py                  # Folium 視覺化
│
├── Route_Planner.ipynb           # ★ 主執行 notebook
├── Scgraph_hotfixer.ipynb        # 互動式 edge hotfix 工具
├── apply_edge_hotfix.py          # 批次套用 hotfix patch
├── research_book.ipynb           # 雜項實驗 / debug
├── compare_land.py               # 對比 Natural Earth vs GSHHS
├── pack_code.py                  # 打包整個 repo 為單一文字檔
│
├── aoi_cache/                    # （gitignore）pickled graph + bundle
│   ├── G_global.pkl.gz
│   └── out_global.pkl.gz
│
├── requirements.txt
├── CLAUDE.md                     # 給 Claude Code 的工作守則
└── README.md
```

---

## Cache 與 Hotfix 守則

- `aoi_cache/*.pkl.gz` 是長時間運算結果，**請勿直接覆寫**。
- 修圖請走 `apply_edge_hotfix.py` 的模式：讀入 → patch → 寫到帶日期 suffix 的新檔。
- `apply_edge_hotfix.py` 支援三種操作：
  - `weight` — 修改邊權重（例如把窄水道設成 `9999999` km 等同封鎖）。
  - `add` — 新增邊（不指定權重時自動取 haversine 距離）。
  - `remove` — 刪除邊。
- **預設策略是改權重，不增刪 nodes**。新增 / 刪除節點會打亂下游 idx，務必先確認影響範圍。

---

## 已知限制 / 待整理

- 程式碼是研究用，沒有測試套件。重構時要靠人工驗證。
- `e_t_transfer.py` 與 `e_t_transfer_v2.py` 並存，目前主路徑用 v2，舊版未刪。
- `Route_Planner_Full_Code.txt` 是 `pack_code.py` 自動生成的快照，不要手動編輯。
- AOI 太大時建構時間明顯變長，建議按區域分批或調小 `pad_deg`。
- 跨日期線（±180°）已在 `scgraph_bridge` 處理，但仍有極端情況需確認。

---

## License

未指定。內部使用。
