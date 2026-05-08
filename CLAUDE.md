# CLAUDE.md

這份檔案給 Claude Code 看，說明在此 repo 工作時要遵守的大原則。
僅描述方針，不重述程式碼可自行讀到的資訊。

---

## 1. 專案性質

這是一個**海上 / 沿岸路徑規劃**研究型專案，主要產物：

- `routing_map/` — 模組化的 AOI pipeline（land 載入 → ring 構築 → ring/sea graph → e-t transfer → t-gate connectors → routing graph）。
- `Route_Planner.ipynb` — 主要的執行 / 實驗 notebook。
- `Scgraph_hotfixer.ipynb` + `apply_edge_hotfix.py` — 對 cache 圖做手動補丁的工具。
- `aoi_cache/` — 預先計算好的 graph / bundle，以 `*.pkl.gz` 儲存（**已 gitignore，不要追蹤**）。

這是**研究 / 個人專案**，不是 production service。可讀性與可實驗性優先於架構潔癖。

---

## 2. 工作守則（大原則）

### 2.1 修改的範圍

- **被要求做什麼，就只做什麼。** 不要順手重構、不要清理「看起來怪怪的」程式碼、不要加抽象。
- 修 bug 時，先理解根因再下手；不要為了讓錯誤訊息消失而改判斷條件。
- 不確定使用者意圖時，先問再動手，不要猜。

### 2.2 Notebook vs Python 模組

- `routing_map/` 下的 `.py` 是「正本」，notebook 通常只是 driver / 視覺化。
- 若 notebook 與模組邏輯重複，**改模組不改 notebook**，除非使用者明確說要動 notebook。
- 編輯 `.ipynb` 時，只動使用者指定的 cell，不要動 metadata、執行 output、kernel spec。

### 2.3 Cache（`aoi_cache/`）很貴，不要亂動

- `G_global.pkl.gz`、`out_global.pkl.gz` 是長時間運算的結果，**禁止覆寫，除非使用者明確要求**。
- 要修改圖內容，採用 `apply_edge_hotfix.py` 的模式：讀入 → patch → **寫到新檔名**（例如帶日期 suffix），原檔保留。
- 讀 cache 也要走 `routing_map.cache_utils`，不要自己 `pickle.load` 繞過。

### 2.4 Index / ID 的不可變性

- 過往的事故顯示：**新增或刪除 nodes 會打亂下游所有 idx**。預設策略是「**改權重**，不增刪 nodes**」**。
- 若真的需要動 node 集合，必須先和使用者確認影響範圍，並明示哪些下游 cache 會失效。

---

## 3. 程式碼風格

- Python 風格跟現有檔案一致即可：`from __future__ import annotations`、`@dataclass` config、type hints 盡量寫但不強制完整。
- **不要新增註解，除非那個註解講的是 _why_**（隱藏約束、踩過的坑、會讓人誤解的決策）。don't narrate _what_ — 程式碼自己會講。
- 既有的中文註解保留，沿用同樣語氣（混用中英文是 OK 的，這是專案常態）。
- 避免大規模重新命名 / 換 import 路徑 — 此 repo 沒有測試套件可保護你，rename 一個函式名可能會悄悄打爛 notebook。

---

## 4. 依賴管理

- `requirements.txt` 是凍結版（含完整版本號），**不要隨手升級**。
- 需要新套件時，先在對話裡跟使用者確認；確認後再加進 requirements.txt 並用相同的固定版本格式。
- 主要關鍵依賴：`networkx`、`shapely`、`geopandas`、`pyproj`、`scgraph`、`searoute`、`folium`。動到這些之前先想清楚。

---

## 5. 大檔案 / 輸出

- `*.html`（folium 地圖輸出）、`*.pkl.gz`、`*.parquet` 都已 gitignore，**不要 git add 這類檔案**。
- 產生新的 debug HTML 時，沿用既有命名慣例（例如 `aoi_rings_p2p_debug.html`），不要創一堆同類檔案。
- `Route_Planner_Full_Code.txt` 是 `pack_code.py` 自動產生的快照，不要手動編輯。

---

## 6. Git

- 預設**不要 commit**。除非使用者明確說「commit」「推上去」之類，才動 git。
- Commit message 風格：跟最近幾筆一致（短中文敘述 + 必要時補英文細節）。
- **絕對不要** `--no-verify`、`reset --hard`、force push，除非使用者明確指示。
- `aoi_cache/`、`.venv/`、`__pycache__/`、HTML / pickle 輸出已在 `.gitignore`，commit 前再檢查一次別誤加。

---

## 7. 環境

- Windows 11 + PowerShell；路徑用反斜線、用雙引號包含空白的路徑（`"Slab Project"`）。
- 專案根目錄含空白：`C:\Users\slab\Desktop\Slab Project\Route Planner`，所有 shell 命令裡都要記得加引號。
- Python 在 `.venv/`，notebook 也跑在這個 venv。
- 跑長運算時提醒使用者「這會花很久」，不要默默開一個會跑半小時的 cell。

---

## 8. 回答風格

- 使用者偏好**簡短的中文回覆**（繁體）。技術名詞 / API 名稱保留英文不要硬翻。
- 解釋演算法或 pipeline 時可以用條列，但不要寫長篇 summary 重述剛剛做的事 — 看 diff 比較快。
- 遇到要做設計判斷的時候，給 1～2 個方案 + tradeoff，讓使用者選，不要直接幫他決定大方向。
