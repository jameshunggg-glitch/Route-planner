import os
import webbrowser
import folium
from pathlib import Path
from shapely.geometry import mapping
import geopandas as gpd

# 匯入你專案中的讀取邏輯
from routing_map.io_land import load_polys_in_bbox

# ==================================================
# 1. 配置設定 (請在此填寫你的檔案路徑)
# ==================================================
OLD_LAND_PATH = Path(r"C:\Users\slab\Desktop\Slab Project\Stage1\data\Land\ne_10m_land.shp")
NEW_LAND_PATH = Path(r"C:\Users\slab\Desktop\Slab Project\Stage1\data\GSHHS\GSHHS_shp\h\GSHHS_h_L1.shp") 

OUTPUT_HTML = "land_comparison_result.html"

# 你指定的 AOI 範圍 (min_lon, min_lat, max_lon, max_lat)
# Point 1: 7.58799, 116.42335 (Lat, Lon)
# Point 2: 21.81055, 128.31758 (Lat, Lon)
BBOX = (116.42335, 7.58799, 128.31758, 21.81055)

# ==================================================
# 2. 讀取與處理資料
# ==================================================
print(f"正在掃描範圍: {BBOX}")

# 讀取舊地圖
print("正在讀取舊地圖 (Natural Earth)...")
polys_old = load_polys_in_bbox(OLD_LAND_PATH, BBOX)

# 讀取新地圖
print("正在讀取新地圖 (GSHHS High Res)...")
polys_new = load_polys_in_bbox(NEW_LAND_PATH, BBOX)

print(f"完成！讀取到舊地圖多邊形數量: {len(polys_old)}")
print(f"完成！讀取到新地圖多邊形數量: {len(polys_new)}")

# ==================================================
# 3. 建立 Folium 地圖
# ==================================================
# 計算地圖中心
center_lat = (BBOX[1] + BBOX[3]) / 2
center_lon = (BBOX[0] + BBOX[2]) / 2

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=7,
    tiles="CartoDB positron" # 使用簡潔底圖
)

# 建立圖層組
fg_old = folium.FeatureGroup(name="舊地圖 (ne_10m) - 紅色", show=True)
fg_new = folium.FeatureGroup(name="新地圖 (GSHHS_h) - 綠色", show=True)

# 繪製舊地圖 (紅色半透明)
for p in polys_old:
    # 稍微簡化頂點以利於網頁流暢度
    p_sim = p.simplify(0.0005, preserve_topology=True)
    folium.GeoJson(
        mapping(p_sim),
        style_function=lambda x: {
            "fillColor": "red",
            "color": "red",
            "weight": 1,
            "fillOpacity": 0.4
        }
    ).add_to(fg_old)

# 繪製新地圖 (綠色不透明)
for p in polys_new:
    p_sim = p.simplify(0.0005, preserve_topology=True)
    folium.GeoJson(
        mapping(p_sim),
        style_function=lambda x: {
            "fillColor": "#228B22", # Forest Green
            "color": "#006400",
            "weight": 1,
            "fillOpacity": 0.8
        }
    ).add_to(fg_new)

# 加入地圖
fg_old.add_to(m)
fg_new.add_to(m)

# 加入圖層控制
folium.LayerControl(collapsed=False).add_to(m)

# ==================================================
# 4. 儲存並自動開啟
# ==================================================
m.save(OUTPUT_HTML)
print(f"\n對比地圖已生成: {os.path.abspath(OUTPUT_HTML)}")

webbrowser.open(os.path.abspath(OUTPUT_HTML))