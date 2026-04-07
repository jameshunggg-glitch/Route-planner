from __future__ import annotations
import folium
from shapely.geometry import LineString, Polygon, MultiPolygon, GeometryCollection

def add_points_layer(m: folium.Map, df, *, lon_col="lon", lat_col="lat", name="points", show=True):
    fg = folium.FeatureGroup(name=name, show=show)
    for _, r in df.iterrows():
        folium.CircleMarker(location=[float(r[lat_col]), float(r[lon_col])], radius=3).add_to(fg)
    fg.add_to(m)
    return fg

def add_lines_layer(m: folium.Map, lines_ll, *, name="lines", show=True, weight=2, opacity=0.8):
    fg = folium.FeatureGroup(name=name, show=show)
    for ln in lines_ll:
        if ln is None or ln.is_empty:
            continue
        coords = [(lat, lon) for lon, lat in ln.coords]
        folium.PolyLine(locations=coords, weight=weight, opacity=opacity).add_to(fg)
    fg.add_to(m)
    return fg
