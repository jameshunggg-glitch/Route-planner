import os
import gzip
import pickle
import numpy as np
import networkx as nx
from datetime import datetime

# 匯入專案內部組件
from routing_map import cache_utils
from routing_map.routing_graph import haversine_km, compute_edge_masks
from scipy.spatial import cKDTree
from routing_map.geom_utils import build_projector_from_bbox


# ==========================================
# 1. 配置路徑與設定
# ==========================================
CACHE_DIR = r"C:\Users\slab\Desktop\Slab Project\Route Planner\aoi_cache"
IN_OUT_PKL = os.path.join(CACHE_DIR, "out_global.pkl.gz")
IN_G_PKL = os.path.join(CACHE_DIR, "G_global.pkl.gz")

OUT_OUT_PKL = os.path.join(
    CACHE_DIR, f"out_global_patched_{datetime.now().strftime('%m%d')}.pkl.gz"
)
OUT_G_PKL = os.path.join(
    CACHE_DIR, f"G_global_patched_{datetime.now().strftime('%m%d')}.pkl.gz"
)


# ==========================================
# 2. 預設補丁清單
# action: "weight" | "add" | "remove"
# value:
#   - weight: 新的權重 (公里)
#   - add: 若為 None，則自動計算 haversine 距離
#   - remove: 可忽略
# ==========================================
PATCH_LIST = [
    {
        "name": "封鎖馬尼拉特定水道 (Penalty)",
        "p1": (120.354, 14.382),  # (lon, lat)
        "p2": (120.582, 14.521),
        "action": "weight",
        "value": 9999999,
    },
    {
        "name": "手動打通窄水道 (Add Edge)",
        "p1": (118.421, 24.512),
        "p2": (118.550, 24.630),
        "action": "add",
        "value": None,
    },
]


def find_nearest_sea_node_id(p_ll, out):
    """根據座標在 S_nodes 中尋找最近的 sea node ID"""
    S_nodes = out["S_nodes"]
    sea_kdt = out["sea_kdt"]
    proj = out["proj"]

    # 轉換為投影座標 (meters)
    x, y = proj.ll2m(float(p_ll[0]), float(p_ll[1]))

    # 查詢 KDTree
    try:
        # sklearn KDTree 格式
        dists, idxs = sea_kdt.query(np.array([[x, y]]), k=1)
        idx = int(idxs[0][0])
    except Exception:
        # scipy cKDTree 格式
        dist, idx = sea_kdt.query([x, y], k=1)
        idx = int(idx)

    node_id = str(S_nodes.iloc[idx]["node_id"])
    node_ll = (
        float(S_nodes.iloc[idx]["lon"]),
        float(S_nodes.iloc[idx]["lat"]),
    )

    print(
        f"  -> 座標 {p_ll} 匹配到節點: {node_id} "
        f"(距離補丁座標約 {haversine_km(p_ll, node_ll):.2f} km)"
    )
    return node_id, node_ll


def _validate_patch(patch, i):
    """檢查 patch 格式是否合理"""
    required_keys = {"name", "p1", "p2", "action"}
    missing = required_keys - set(patch.keys())
    if missing:
        raise ValueError(f"Patch #{i} 缺少必要欄位: {sorted(missing)}")

    if not isinstance(patch["p1"], (tuple, list)) or len(patch["p1"]) != 2:
        raise ValueError(f"Patch #{i} 的 p1 格式錯誤，需為 (lon, lat)")

    if not isinstance(patch["p2"], (tuple, list)) or len(patch["p2"]) != 2:
        raise ValueError(f"Patch #{i} 的 p2 格式錯誤，需為 (lon, lat)")

    action = str(patch["action"]).lower()
    if action not in {"weight", "add", "remove"}:
        raise ValueError(
            f"Patch #{i} 的 action={patch['action']} 不支援，"
            f"只能是 weight / add / remove"
        )

    if action == "weight" and "value" not in patch:
        raise ValueError(f"Patch #{i} action=weight 時必須提供 value")


def apply_edge_hotfix(custom_patches=None):
    """
    套用 edge hotfix 到全域海圖。

    Parameters
    ----------
    custom_patches : list[dict] | None
        若提供，優先使用此外部補丁清單；
        若為 None，則使用本檔案中的 PATCH_LIST。
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 開始執行海網路邊級補丁...")

    # 決定本次要使用的 patch 清單
    active_patches = custom_patches if custom_patches is not None else PATCH_LIST

    if not isinstance(active_patches, (list, tuple)):
        raise TypeError("custom_patches 必須是 list[dict] 或 None")

    if len(active_patches) == 0:
        print("未提供任何 patch，流程結束。")
        return

    for i, patch in enumerate(active_patches, start=1):
        _validate_patch(patch, i)

    # --- 1. 載入原始資料 ---
    if not os.path.exists(IN_OUT_PKL) or not os.path.exists(IN_G_PKL):
        print(f"錯誤: 找不到原始檔案於 {CACHE_DIR}")
        return

    with gzip.open(IN_OUT_PKL, "rb") as f:
        out = pickle.load(f)
    
    with gzip.open(IN_OUT_PKL, "rb") as f:
        out = pickle.load(f)
    
    # --- 補水邏輯 (Hydration) ---
    # 因為 .pkl.gz 為了節省空間不儲存 KDTree 和 Projector，這裡需要重建
        if out.get("proj") is None:
            bbox_ll = out.get("bbox_ll")
            out["proj"] = build_projector_from_bbox(bbox_ll)
        
        if out.get("sea_kdt") is None:
            print("正在重建 sea_kdt 空間索引...")
            S_nodes = out["S_nodes"]
            # 提取投影座標 (x_m, y_m)
            xy = S_nodes[["x_m", "y_m"]].to_numpy(dtype=float)
            out["sea_kdt"] = cKDTree(xy)

    with gzip.open(IN_G_PKL, "rb") as f:
        G_pack = pickle.load(f)
        G_base = G_pack["G_base"] if isinstance(G_pack, dict) and "G_base" in G_pack else G_pack

    print(f"已載入圖資: Nodes={G_base.number_of_nodes()}, Edges={G_base.number_of_edges()}")
    print(f"本次補丁數量: {len(active_patches)}")

    # 用於同步 out["S_edges"]
    edges_to_add_to_out = []
    edges_to_remove_from_out = []

    # --- 2. 執行補丁清單 ---
    for patch in active_patches:
        print(f"\n執行補丁項目: {patch['name']}")
        u_id, u_ll = find_nearest_sea_node_id(patch["p1"], out)
        v_id, v_ll = find_nearest_sea_node_id(patch["p2"], out)

        if u_id == v_id:
            print(f"  警告: 補丁端點匹配到同一個節點 {u_id}，略過。")
            continue

        action = str(patch["action"]).lower()

        if action == "weight":
            new_weight = patch["value"]
            if G_base.has_edge(u_id, v_id):
                old_w = G_base[u_id][v_id].get("weight")
                G_base[u_id][v_id]["weight"] = new_weight
                if "length_km" in G_base[u_id][v_id]:
                    G_base[u_id][v_id]["length_km"] = new_weight
                print(f"  更新權重: {u_id} <-> {v_id} | {old_w} -> {new_weight}")
            else:
                print(f"  錯誤: 找不到邊 {u_id} <-> {v_id}，無法修改權重。")

        elif action == "remove":
            if G_base.has_edge(u_id, v_id):
                G_base.remove_edge(u_id, v_id)
                edges_to_remove_from_out.append((u_ll, v_ll))
                print(f"  已移除邊: {u_id} <-> {v_id}")
            else:
                print(f"  警告: 邊 {u_id} <-> {v_id} 本就不存在，無需移除。")

        elif action == "add":
            w = patch.get("value")
            if w is None:
                w = haversine_km(u_ll, v_ll)

            layer, ban, lat_max = compute_edge_masks(
                G_base,
                u_id,
                v_id,
                etype="sea",
                hard_lat_cap_deg=70.0,
            )

            G_base.add_edge(
                u_id,
                v_id,
                weight=w,
                length_km=w,
                etype="sea",
                layer_mask=layer,
                ban_mask=ban,
                lat_max_abs=lat_max,
            )
            edges_to_add_to_out.append((u_ll, v_ll))
            print(f"  已新增邊: {u_id} <-> {v_id} (weight={w:.2f} km)")

    # --- 3. 同步 out 字典 ---
    if edges_to_add_to_out or edges_to_remove_from_out:
        print("\n同步 out['S_edges'] 清單...")
        s_edges = list(out["S_edges"])

        for rem in edges_to_remove_from_out:
            s_edges = [e for e in s_edges if not (set(e) == set(rem))]

        for add in edges_to_add_to_out:
            s_edges.append(add)

        out["S_edges"] = s_edges

    # 重新計算 sea_ok_set
    print("重新計算連通分量 (sea_ok_set)...")
    sea_subgraph = G_base.subgraph(
        [n for n, d in G_base.nodes(data=True) if d.get("kind") == "sea"]
    )
    components = list(nx.connected_components(sea_subgraph))

    if components:
        largest_comp = max(components, key=len)
        S_nodes = out["S_nodes"]
        id2idx = {str(row.node_id): idx for idx, row in S_nodes.iterrows()}
        new_ok_set = {id2idx[nid] for nid in largest_comp if nid in id2idx}
        out["sea_ok_set"] = new_ok_set
        print(f"  連通性更新完成: 最大分量節點數={len(new_ok_set)}")

    # --- 4. 存檔 ---
    print("\n正在儲存補丁檔案...")
    cache_utils.save_graph_cache(
        G_base,
        cfg=out.get("cfg"),
        graph_build_args={},
        cache_dir=CACHE_DIR,
        filename=os.path.basename(OUT_G_PKL),
    )
    cache_utils.save_out_cache(
        out,
        cache_dir=CACHE_DIR,
        filename=os.path.basename(OUT_OUT_PKL),
    )

    print("成功！補丁已存至:")
    print(f"  - {OUT_OUT_PKL}")
    print(f"  - {OUT_G_PKL}")
    print("\n請使用 Route_Planner.ipynb 測試新路徑。")


if __name__ == "__main__":
    apply_edge_hotfix()