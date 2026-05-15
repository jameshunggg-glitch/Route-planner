# NGZ Pipeline 交接文件（agent-oriented）

> **Audience**: 下一個處理這個 repo 的 Claude agent。本檔給你看的、不
> 給人類看。寫成「事實 + 指令 + 不可破壞清單」，不寫散文。

---

## TL;DR

- **Repo**: `C:\Users\slab\Desktop\Slab Project\Route Planner`
- **Branch**: `feature/ngz-pipeline-integration`（**not** main）
- **PR**: <https://github.com/jameshunggg-glitch/Route-planner/pull/1>（已開、未 merge）
- **任務**: 修小問題 → commit → push → 觀察 PR 自動更新 → merge
- **預期 git 狀態**: working tree clean、up-to-date with origin
- **預期測試狀態**: `python ngz_smoke_test.py` → 15 pass / 3 OBSOLETE skip / 0 fail
- **Python venv**: `.venv/`（PowerShell 路徑用 `.venv\Scripts\python.exe`）
- **OS**: Windows 11 + PowerShell（路徑含空白需引號）

---

## 不要預讀（避免吃 context；只在被要求或實際需要時才 load）

| 檔案 | 行數 | 何時需要 load |
|---|---:|---|
| `docs/NGZ baseline patch plan.md` | 374 | user 問「為什麼這樣設計」、或要重新討論某個 D 決議 |
| `docs/NGZ flow.md` | 300 | user 問「舊範式（PR2）以前是怎麼做的」、追溯歷史 |
| `ngz_smoke_test.py` 的 test body | 700+ | 某個 N_n* 測試 fail、或要改測試 |
| `routing_map/ngz.py` 完整內容 | 1300 | 真的要改 NGZ helper 時才 read（先 grep / Glob 定位） |
| `routing_map/pipeline.py` 完整內容 | 1000+ | 真的要改 pipeline 時才 read 對應段落 |
| `Route_Planner.ipynb` 全檔 | ~2500 行 JSON | 永遠不要整檔 read，太佔；要動就 read 特定 cell |

**節省 context 規則**：本檔 + `CLAUDE.md`（260 行）就夠你進入狀況。不必預載
其他大檔。Grep 找目標 → Read 對應的 50-100 行區段就好。

---

## 必讀（按順序）

1. **這份檔**（你正在讀）
2. **`CLAUDE.md`**（260 行，專案守則）特別注意：
   - §2.3：`aoi_cache/` 禁止覆寫
   - §2.4：node idx 不可變（**改權重不增刪 nodes**）
   - §6：git 守則（不 force push / 不 --no-verify / 不亂 reset --hard）
   - §8：簡短繁中、技術名詞保留英文、不寫長 summary

---

## 新範式速讀（30 秒理解 NGZ 怎麼運作）

```
run_p2p(out, origin, dest, ngz_polygons=[poly]) 時的流程：

1. NGZ pre-compute (pipeline.py ~line 316-415)
   normalize_ngz_inputs → build_ngz_t_rings → build_ngz_overlay_lite
   → apply_ngz_mode（lenient/strict/relocate）
   → build_ngz_collision_geom（exempt origin/dest 所在 NGZ）
   → 準備 collision_ll_for_patch（land + NGZ_excluding_exempt 的 lon/lat union prep）

2. A* baseline P0 (pipeline.py ~line 513-527)
   pure A* on G（NGZ 沒進 G！），得到 path_ll_raw

3. NGZ patching block (pipeline.py ~line 533-580, A* 後 / repair 前)
   detect_blocked_subpaths(path_ll_raw, ngz_groups, exempt)
     → 每段 build_local_visibility_graph (A, B, T-ring 頂點)
       多 NGZ 同段時自動 pairwise_visibility=True
     → solve_local_patch (nx.shortest_path)
     → apply_patches_to_baseline (splice 回 path_ll_raw)
   若 disconnected → raise NgzPatchUnreachableError → res.error = "ngz_patch_unreachable: ..."

4. Repair (pipeline.py ~line 590)
   has_ngz=True 時 **skip**（patched path 含 T-ring lon/lat，沒對應 graph node id）

5. Simplify (pipeline.py ~line 612)
   simplify_path_visibility，collision_m 已含 NGZ
```

### 核心承諾（**這兩條是 PR3 重設計的目的，不可破壞**）

- ✅ **NGZ off-baseline** → 結果**逐點等於** baseline（N2 test 守此）
- ✅ **NGZ 擋到** → 形狀貼近 baseline + 只在被擋處鼓出去

### Settled decisions（不要重新討論，user 已拍板）

| ID | 內容 |
|---|---|
| D1a | 移除 PR2 的 compose / mask / B_NGZ-ban；保留 normalize / build_t_rings / collision_geom / apply_mode 等 |
| D2a | 多 NGZ 同擋同段 → 同一個 local visibility graph |
| D3a | A→B disconnected → `raise NgzPatchUnreachableError`（無 fallback） |
| D4 | snap 不認 T-ring 頂點；lenient 模式靠 collision exempt |
| D5a | baseline = pure A\*（不 repair / 不 simplify）；patching 後做一次 final simplify |

---

## 檔案 map

### `routing_map/ngz.py`

| 區段 | 用途 |
|---|---|
| `NgzInput / NgzGroup / NgzRingResult / NgzOverlay` | dataclasses（保留） |
| `_iter_polygons / _ensure_valid / _haversine_km / _to_ngz_input` | internal helpers |
| `clip_collision_to_ngz_bbox` | 把 global land collision clip 成 local 版（性能關鍵） |
| `split_polygon_at_antimeridian` | dateline 切分 |
| `normalize_ngz_inputs` | dateline split + land subtract + group merge |
| `build_ngz_t_rings` | 對 group 建 T-ring（凸殼化過的） |
| `_node_id` | "NGZ:{gid}:{seq}" 格式 |
| `build_ngz_collision_geom` | NGZ union（lon/lat + metric + prepared） |
| **`BlockedSubpath` (dataclass)** | 新增：baseline 中跟 NGZ 相交的子段 |
| **`NgzPatchUnreachableError`** | 新增：A↔B 不連通時 raise |
| **`detect_blocked_subpaths`** | 新增：找 path_ll 與 NGZ 相交的子段 |
| **`build_local_visibility_graph`** | 新增：local A/B + T-ring 頂點 visibility graph |
| **`solve_local_patch`** | 新增：nx.shortest_path + id→ll 翻譯 |
| **`apply_patches_to_baseline`** | 新增：reverse-order splice |
| **`build_ngz_overlay_lite`** | 新增：只填 groups + rings + nodes 的 lite overlay |
| `NgzInsideError / detect_inside_ngz / _nearest_taut_vertex_ll / apply_ngz_mode` | 保留 |
| ~~`build_ngz_overlay`~~ | **已刪**（舊範式） |
| ~~`_kdt_query`~~ | **已刪** |
| ~~`compose_ngz_into_graph`~~ | **已刪** |

### `routing_map/pipeline.py`（NGZ 相關段落速查）

| 大約行號 | 內容 |
|---|---|
| ~10-32 | imports（NGZ helpers + `NgzPatchUnreachableError`） |
| ~40-69 | `RoutePolicy`（`active_ban_mask = B_HIGH_LAT`，已刪 B_NGZ） |
| ~316-415 | **NGZ pre-compute block**（normalize / rings / overlay_lite / mode / collision） |
| ~489-529 | A\* baseline（不擴增 G） |
| ~533-580 | **NGZ patching block**（新增；detect → local graph → solve → splice） |
| ~590 | `if ... and not has_ngz` ← skip repair |
| ~612 | simplify（不變，collision_m 已含 NGZ） |

精確行號用 `Grep` 找 `# NGZ pre-compute` / `# NGZ local patching` / `# repair` / `# simplify` 標記。

### `routing_map/routing_graph.py`

刪掉的 4 個常數：
- `L_NGZ_RING = 1 << 9`（保留位）
- `L_NGZ_GATE = 1 << 10`（保留位）
- `L_NGZ_NGZ_GATE = 1 << 11`（保留位）
- `B_NGZ = 1 << 1`（保留位）

### `ngz_smoke_test.py`

| Test | 性質 | 需 cache |
|---|---|---|
| T1-T5, T7, T8 | 純模組（geom / collision） | 否 |
| T6, T9, T10 | **OBSOLETE skip** | - |
| N1 degenerate | ngz=None → == baseline | 是 |
| N2 off-baseline | **核心 win**：結果逐點 == baseline | 是 |
| N3 single NGZ | 擋 baseline、繞行 | 是 |
| N4 concave (U) | T-ring 凸殼化 → 走 U 外 | 是 |
| N5 multi-NGZ same seg | 自動 pairwise=True、兩 group_id 都偵測到 | 是 |
| N6 lenient origin | origin in NGZ-A，NGZ-A 不擋路 | 是 |
| N7 unreachable (unit) | fake disconnected g_local → raise | 否 |
| N8 final simplify | simplified 不交 NGZ | 是 |

預設整合測試 OD：`(130.0, 22.0)` → `(145.0, 32.0)`（純太平洋）。

---

## 已知 invariants（**不要破壞**）

1. **`NGZ off-baseline → 結果逐點 == baseline`**（N2 test 守此；用浮點 tolerance 1e-9）
2. **`has_ngz=True → repair skipped`**（pipeline.py 內 `and not has_ngz` 條件守此）
3. **`multi-NGZ same segment → pairwise_visibility=True`**（pipeline.py 內 `len(blocked.ngz_group_ids) > 1` 條件自動觸發）
4. **不擴增 G、不增刪 sea node**（CLAUDE.md §2.4；之前事故：增刪 node 會打亂下游 idx）
5. **`aoi_cache/G_global.pkl.gz` / `out_global.pkl.gz` 禁止覆寫**（CLAUDE.md §2.3）
6. **patching 全在 lon/lat 空間** + 本地小 networkx graph，不碰 cached G

---

## 驗證指令（新 session 第一步跑這些）

```bash
# 確認狀態
git status                          # 應該 clean
git log --oneline -6                # 應該看到 d8f4861 / a109b65 / 523cd9b / ...
git branch -vv                      # 確認在 feature/ngz-pipeline-integration

# 跑測試（會載 cache，~5 秒）
.venv\Scripts\python.exe ngz_smoke_test.py
# 預期：
#   Total: 15 passed, 3 skipped, 0 failed

# 看看 GitHub 那邊
# 開 https://github.com/jameshunggg-glitch/Route-planner/pull/1
```

如果 working tree **不** clean、測試**不**全綠 → 先停下來告訴 user，不要直接動。

---

## 第一步（給下個 agent 的 action）

1. 跑上面的驗證指令，把結果報告給 user。
2. 請 user 列出小問題。建議用這個 template 問：
   > 請列出要修的小問題，每個附：
   > - **症狀**（看到什麼異常）
   > - **預期**（應該怎樣）
   > - **重現步驟**（如何觸發；OD / NGZ 設定）
3. 每個 issue：
   - 討論根因（不要為了訊息消失就改判斷條件——CLAUDE.md §2.1）
   - 改 code
   - 跑 `python ngz_smoke_test.py` 確認沒打壞 invariants
   - `git add <specific files>` + `git commit -m "fix: ..."` + `git push`
   - 提醒 user 上 PR #1 看新 commit 已自動更新

---

## Commit / PR 慣例（user 偏好）

- Commit message 風格：短中文標題 + 必要時 body（看 `git log` 最近幾筆對照）
- Co-Authored-By line：`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
- HEREDOC 寫 message（避免引號 escape 問題）
- 一個 logical fix 一個 commit；不要混 unrelated fix
- 改完 push → 觀察 PR 自動更新（這是 user 練 PR 流程的重點）

---

## 觀察：本檔自身不在版控

`docs/NGZ-handover.md` 是 untracked（git status 會 list）；user 明確要求
不 commit。Merge PR #1 前若沒清掉，會殘留 untracked，但**不會**進 main。
若 user 嫌 git status 噪音，可加 `docs/NGZ-handover.md` 到 `.gitignore`。

---

## TODO: 待修問題（placeholder）

由 user 在下個 session 填入。建議格式：

```
### Issue 1: <短描述>
- 症狀:
- 預期:
- 重現步驟:
- 涉及檔案 (猜測):
- 狀態: [ ] 待討論 / [ ] in progress / [x] commit XXXX 修掉
```
