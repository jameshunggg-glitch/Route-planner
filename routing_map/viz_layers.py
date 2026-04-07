from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math

import folium
import pandas as pd
from folium import MacroElement
from jinja2 import Template

LonLat = Tuple[float, float]
BBoxLL = Tuple[float, float, float, float]

import folium
from folium import MacroElement
from jinja2 import Template


class _SelectAllNoneControl(MacroElement):
    """
    Add two buttons (All / None) that toggle all Overlay layers in folium.LayerControl.
    """
    def __init__(self, position="topright", all_text="All", none_text="None"):
        super().__init__()
        self._name = "SelectAllNoneControl"
        self.position = position
        self.all_text = all_text
        self.none_text = none_text

        self._template = Template(
            """
            {% macro script(this, kwargs) %}
            (function() {
              // map variable name in folium
              var map = {{ this._parent.get_name() }};

              function setAllOverlays(checked) {
                // LayerControl creates checkboxes under:
                // .leaflet-control-layers-overlays input[type=checkbox]
                var overlays = document.querySelectorAll(
                  ".leaflet-control-layers-overlays input[type='checkbox']"
                );
                overlays.forEach(function(cb) {
                  // Click to ensure Leaflet fires handlers properly
                  if (cb.checked !== checked) cb.click();
                });
              }

              var control = L.control({position: "{{ this.position }}"});
              control.onAdd = function() {
                var div = L.DomUtil.create("div", "leaflet-bar leaflet-control");
                div.style.background = "white";
                div.style.padding = "6px";
                div.style.borderRadius = "4px";

                var btnAll = L.DomUtil.create("a", "", div);
                btnAll.href = "#";
                btnAll.innerHTML = "{{ this.all_text }}";
                btnAll.style.display = "block";
                btnAll.style.textAlign = "center";
                btnAll.style.padding = "2px 8px";
                btnAll.style.textDecoration = "none";

                var btnNone = L.DomUtil.create("a", "", div);
                btnNone.href = "#";
                btnNone.innerHTML = "{{ this.none_text }}";
                btnNone.style.display = "block";
                btnNone.style.textAlign = "center";
                btnNone.style.padding = "2px 8px";
                btnNone.style.textDecoration = "none";

                // Prevent map dragging/zooming when clicking the buttons
                L.DomEvent.disableClickPropagation(div);
                L.DomEvent.disableScrollPropagation(div);

                L.DomEvent.on(btnAll, "click", function(e) {
                  L.DomEvent.preventDefault(e);
                  setAllOverlays(true);
                });

                L.DomEvent.on(btnNone, "click", function(e) {
                  L.DomEvent.preventDefault(e);
                  setAllOverlays(false);
                });

                return div;
              };

              control.addTo(map);
            })();
            {% endmacro %}
            """
        )


def add_select_all_none_layer_control(
    m: folium.Map,
    *,
    position: str = "topright",
    all_text: str = "All",
    none_text: str = "None",
) -> None:
    """
    Add 'All/None' buttons to toggle all overlay layers in folium.LayerControl.

    Call this AFTER you've added folium.LayerControl to the map
    (or at least after layers are created).
    """
    _SelectAllNoneControl(position=position, all_text=all_text, none_text=none_text).add_to(m)



# === Dateline visualization mode (Folium/Leaflet) ===
# Leaflet repeats the world horizontally. When an AOI bbox crosses the antimeridian
# (min_lon > max_lon), we can render everything continuously by shifting the left-side
# longitudes ([-180, max_lon]) by +360 so bbox/path/nodes/edges live in one world copy.
DATELINE_VIZ_MODE = "unwrap360"  # or "split"

def _bbox_crosses_dateline(bbox_ll: Optional[BBoxLL]) -> bool:
    if bbox_ll is None:
        return False
    min_lon, _, max_lon, _ = map(float, bbox_ll)
    return min_lon > max_lon

def _lon_viz(lon: float, bbox_ll: Optional[BBoxLL]) -> float:
    x = float(lon)
    if DATELINE_VIZ_MODE == "unwrap360" and _bbox_crosses_dateline(bbox_ll) and bbox_ll is not None:
        min_lon, _, max_lon, _ = map(float, bbox_ll)
        if x <= max_lon:
            x += 360.0
    return x

def _bbox_bounds_viz(bbox_ll: BBoxLL) -> BBoxLL:
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)
    if DATELINE_VIZ_MODE == "unwrap360" and min_lon > max_lon:
        return (min_lon, min_lat, max_lon + 360.0, max_lat)
    return (min_lon, min_lat, max_lon, max_lat)

def _in_bbox(p: LonLat, bbox_ll: Optional[BBoxLL]) -> bool:
    if bbox_ll is None:
        return True
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)
    lon, lat = float(p[0]), float(p[1])
    if not (min_lat <= lat <= max_lat):
        return False
    if min_lon <= max_lon:
        return (min_lon <= lon <= max_lon)
    # crosses dateline
    return (lon >= min_lon) or (lon <= max_lon)

def make_base_map(bbox_ll: BBoxLL, *, zoom_start: int = 5, control_scale: bool = True) -> folium.Map:
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)
    crosses = (min_lon > max_lon)

    if not crosses:
        center_lon = (min_lon + max_lon) / 2.0
    else:
        center_lon = (min_lon + (max_lon + 360.0)) / 2.0

    v_min_lon, v_min_lat, v_max_lon, v_max_lat = _bbox_bounds_viz(bbox_ll)
    center = [(min_lat + max_lat) / 2.0, _lon_viz(center_lon, bbox_ll)]

    m = folium.Map(location=center, zoom_start=zoom_start, control_scale=control_scale)
    setattr(m, "_routing_bbox_ll", tuple(map(float, bbox_ll)))

    if DATELINE_VIZ_MODE == "unwrap360" and crosses:
        # === AOI bbox (debug) ===
        # folium.Rectangle(
        #     bounds=[[v_min_lat, v_min_lon], [v_max_lat, v_max_lon]],
        #     fill=False, weight=3, opacity=0.9, color="blue"
        # ).add_to(m)

        m.fit_bounds([[v_min_lat, v_min_lon], [v_max_lat, v_max_lon]])
    else:
        # === AOI bbox (debug) ===
        # folium.Rectangle(
        #     bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        #     fill=False, weight=3, opacity=0.9, color="blue"
        # ).add_to(m)

        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    return m



def add_points_layer(
    m: folium.Map,
    pts_ll: Sequence[LonLat],
    *,
    name: str,
    radius: int = 4,
    show: bool = True,
    bbox_ll: Optional[BBoxLL] = None,
) -> None:
    if bbox_ll is None:
        bbox_ll = getattr(m, "_routing_bbox_ll", None)
    fg = folium.FeatureGroup(name=name, show=show)
    for lon, lat in pts_ll:
        p = (float(lon), float(lat))
        if not _in_bbox(p, bbox_ll):
            continue
        lon_v = _lon_viz_point(p[0], bbox_ll, m)
        folium.CircleMarker([p[1], lon_v], radius=int(radius)).add_to(fg)
    fg.add_to(m)


# === Great-circle densification (for visualization) ===
_EARTH_R_KM = 6371.0088

def _ll_to_unitvec(lon_deg: float, lat_deg: float) -> Tuple[float, float, float]:
    lon = math.radians(float(lon_deg))
    lat = math.radians(float(lat_deg))
    clat = math.cos(lat)
    return (clat * math.cos(lon), clat * math.sin(lon), math.sin(lat))

def _unitvec_to_ll(v: Tuple[float, float, float]) -> LonLat:
    x, y, z = v
    lon = math.degrees(math.atan2(y, x))
    hyp = math.hypot(x, y)
    lat = math.degrees(math.atan2(z, hyp))
    return (lon, lat)

def _dot(u: Tuple[float, float, float], v: Tuple[float, float, float]) -> float:
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def _norm(u: Tuple[float, float, float]) -> float:
    return math.sqrt(_dot(u, u))

def _scale(u: Tuple[float, float, float], a: float) -> Tuple[float, float, float]:
    return (u[0]*a, u[1]*a, u[2]*a)

def _add(u: Tuple[float, float, float], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (u[0]+v[0], u[1]+v[1], u[2]+v[2])

def _normalize(u: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = _norm(u)
    if n <= 1e-15:
        return (0.0, 0.0, 0.0)
    return (u[0]/n, u[1]/n, u[2]/n)

def _central_angle_rad(u: Tuple[float, float, float], v: Tuple[float, float, float]) -> float:
    d = max(-1.0, min(1.0, _dot(u, v)))
    return math.acos(d)

def _slerp(u: Tuple[float, float, float], v: Tuple[float, float, float], t: float, omega: float) -> Tuple[float, float, float]:
    if omega < 1e-12:
        return u
    so = math.sin(omega)
    a = math.sin((1.0 - t) * omega) / so
    b = math.sin(t * omega) / so
    return _normalize(_add(_scale(u, a), _scale(v, b)))

def _densify_great_circle_ll(p0: LonLat, p1: LonLat, *, step_km: float = 20.0) -> List[LonLat]:
    """Densify the great-circle segment between two lon/lat points.

    Notes:
    - Works with "continuous" longitudes (e.g. 190 == -170) and preserves continuity
      relative to p0 to avoid jumps in Leaflet visualization.
    """
    lon0, lat0 = float(p0[0]), float(p0[1])
    lon1, lat1 = float(p1[0]), float(p1[1])

    u = _ll_to_unitvec(lon0, lat0)
    v = _ll_to_unitvec(lon1, lat1)
    omega = _central_angle_rad(u, v)
    dist_km = _EARTH_R_KM * omega

    if step_km <= 0 or dist_km <= step_km:
        return [(lon0, lat0), (lon1, lat1)]

    n_seg = max(1, int(math.ceil(dist_km / float(step_km))))
    out: List[LonLat] = []
    for k in range(n_seg + 1):
        t = k / n_seg
        w = _slerp(u, v, t, omega)
        lon, lat = _unitvec_to_ll(w)

        # keep lon continuous w.r.t lon0
        d = lon - lon0
        if d > 180.0:
            lon -= 360.0
        elif d < -180.0:
            lon += 360.0

        out.append((lon, lat))
    return out

def _densify_polyline_gc_ll(path_ll: Sequence[LonLat], *, step_km: float = 20.0) -> List[LonLat]:
    if not path_ll or len(path_ll) < 2:
        return []
    out: List[LonLat] = []
    for a, b in zip(path_ll, path_ll[1:]):
        seg = _densify_great_circle_ll(a, b, step_km=step_km)
        if out:
            seg = seg[1:]  # avoid duplicating join point
        out.extend(seg)
    return out

def _split_polyline_at_antimeridian_ll(path_ll: Sequence[LonLat]) -> List[List[LonLat]]:
    """
    Split a polyline into multiple segments so that no segment crosses the antimeridian.
    This avoids Leaflet drawing a huge 360°-wrapping line when lon jumps (e.g. 170 -> -170).
    """
    if not path_ll or len(path_ll) < 2:
        return [list(path_ll)] if path_ll else []

    seg: List[LonLat] = [ (float(path_ll[0][0]), float(path_ll[0][1])) ]
    out: List[List[LonLat]] = []

    for (lon1, lat1), (lon2, lat2) in zip(path_ll, path_ll[1:]):
        lon1, lat1 = float(lon1), float(lat1)
        lon2, lat2 = float(lon2), float(lat2)
        d = lon2 - lon1

        if abs(d) <= 180.0:
            seg.append((lon2, lat2))
            continue

        # We need to split at +/-180.
        if d > 180.0:
            # Example: lon1=-170 -> lon2=+170 (jump +340). Short way crosses -180.
            lon2_adj = lon2 - 360.0  # make it continuous going "left"
            # interpolate where lon hits -180
            f = (-180.0 - lon1) / (lon2_adj - lon1)
            latc = lat1 + f * (lat2 - lat1)

            seg.append((-180.0, float(latc)))
            out.append(seg)

            seg = [(180.0, float(latc)), (lon2, lat2)]
        else:
            # Example: lon1=+170 -> lon2=-170 (jump -340). Short way crosses +180.
            lon2_adj = lon2 + 360.0  # make it continuous going "right"
            # interpolate where lon hits +180
            f = (180.0 - lon1) / (lon2_adj - lon1)
            latc = lat1 + f * (lat2 - lat1)

            seg.append((180.0, float(latc)))
            out.append(seg)

            seg = [(-180.0, float(latc)), (lon2, lat2)]

    out.append(seg)
    return out

def _unwrap_polyline_lon_ll(path_ll: Sequence[LonLat]) -> List[LonLat]:
    """Make lon sequence continuous by adding/subtracting 360 when needed."""
    if not path_ll:
        return []
    out = [(float(path_ll[0][0]), float(path_ll[0][1]))]
    prev = out[0][0]
    for lon, lat in path_ll[1:]:
        lon = float(lon); lat = float(lat)
        # shift lon by multiples of 360 so it's closest to prev
        while lon - prev > 180.0:
            lon -= 360.0
        while lon - prev < -180.0:
            lon += 360.0
        out.append((lon, lat))
        prev = lon
    return out

def _lon_to_near_ref(lon: float, ref_lon: float) -> float:
    """Shift lon by multiples of 360 so it's closest to ref_lon."""
    x = float(lon)
    ref = float(ref_lon)
    while x - ref > 180.0:
        x -= 360.0
    while x - ref < -180.0:
        x += 360.0
    return x

def _lon_viz_point(lon: float, bbox_ll: Optional[BBoxLL], m: Optional[folium.Map] = None) -> float:
    """Pick a visualization longitude consistent with the currently drawn path (if any)."""
    x0 = float(lon)

    # If bbox itself crosses the dateline, the existing bbox-based rule is enough.
    if DATELINE_VIZ_MODE == "unwrap360" and _bbox_crosses_dateline(bbox_ll):
        return _lon_viz(x0, bbox_ll)

    # Otherwise, if a path has already been drawn in an unwrapped world copy, align points to it.
    if DATELINE_VIZ_MODE == "unwrap360" and m is not None:
        refs = []
        for k in ("_routing_viz_ref_lon_start", "_routing_viz_ref_lon_end"):
            if hasattr(m, k):
                try:
                    refs.append(float(getattr(m, k)))
                except Exception:
                    pass
        if refs:
            cands = []
            for r in refs:
                xr = _lon_to_near_ref(x0, r)
                cands.append((abs(xr - r), xr))
            cands.sort(key=lambda t: t[0])
            return cands[0][1]

    return x0


def add_path_layer(
    m: folium.Map,
    path_ll: Sequence[LonLat],
    *,
    name: str,
    weight: int = 4,
    opacity: float = 0.95,
    show: bool = True,
    bbox_ll: Optional[BBoxLL] = None,
    geodesic: bool = False,
    geodesic_step_km: float = 20.0,
) -> None:
    if not path_ll or len(path_ll) < 2:
        return
    if bbox_ll is None:
        bbox_ll = getattr(m, "_routing_bbox_ll", None)
    fg = folium.FeatureGroup(name=name, show=show)

    # Leaflet draws straight segments in (WebMercator) map space. If you want the path
    # to *look* like a great-circle, we densify each segment along the great-circle
    # before drawing (purely for visualization; this does not change routing).
    
    # Case 1: bbox crosses dateline AND unwrap360 mode -> keep your existing "single-world copy" behavior
    if bbox_ll is not None and DATELINE_VIZ_MODE == "unwrap360" and _bbox_crosses_dateline(bbox_ll):
        pts_ll_viz = [(_lon_viz(float(lon), bbox_ll), float(lat)) for (lon, lat) in path_ll]
        if geodesic:
            pts_ll_viz = _densify_polyline_gc_ll(pts_ll_viz, step_km=float(geodesic_step_km))
        folium.PolyLine([[lat, lon] for (lon, lat) in pts_ll_viz],
                    color="red", weight=int(weight), opacity=float(opacity)).add_to(fg)

    # Case 2: bbox does NOT cross dateline (e.g. global bbox) -> split the path itself at the antimeridian
    else:
        # Global bbox / non-dateline bbox case:
        # Draw as a single continuous polyline by unwrapping lon sequence (no split).
        pts = _unwrap_polyline_lon_ll(path_ll)

        if geodesic:
            pts = _densify_polyline_gc_ll(pts, step_km=float(geodesic_step_km))

        # Save a reference so point markers can be shifted into the same world copy.
        try:
            if pts:
                setattr(m, "_routing_viz_ref_lon_start", float(pts[0][0]))
                setattr(m, "_routing_viz_ref_lon_end", float(pts[-1][0]))
        except Exception:
            pass

        # NOTE: do NOT split; just draw the unwrapped lon directly.
        # Leaflet can render lon outside [-180, 180] as a different world copy.
        folium.PolyLine([[lat, lon] for (lon, lat) in pts],
                        color="red", weight=int(weight), opacity=float(opacity)).add_to(fg)


    fg.add_to(m)


def add_sea_layers(
    m: folium.Map,
    out: Dict[str, Any],
    *,
    show_nodes: bool = True,
    show_edges: bool = True,
    node_sample: Optional[int] = None,
    max_edges: int = 6000,
    show: bool = True,
    bbox_ll: Optional[BBoxLL] = None,
) -> None:
    if bbox_ll is None:
        bbox_ll = getattr(m, "_routing_bbox_ll", None)

    S_nodes = out.get("S_nodes")
    S_edges = out.get("S_edges")

    if isinstance(S_nodes, pd.DataFrame) and len(S_nodes) > 0 and show_nodes:
        df = S_nodes
        if node_sample is not None and len(df) > int(node_sample):
            df = df.sample(int(node_sample), random_state=7)
        fgN = folium.FeatureGroup(name=f"S_nodes ({len(df)}/{len(S_nodes)})", show=show)
        for _, r in df.iterrows():
            p = (float(r["lon"]), float(r["lat"]))
            if not _in_bbox(p, bbox_ll):
                continue
            folium.CircleMarker([p[1], _lon_viz(p[0], bbox_ll)], color="blue", radius=3).add_to(fgN)
        fgN.add_to(m)

    if not show_edges:
        return
    if not isinstance(S_nodes, pd.DataFrame) or S_edges is None:
        return

    # Build a lookup for index->lonlat if S_edges uses indices
    idx2ll: Dict[int, LonLat] = {}
    if {"lon", "lat"}.issubset(S_nodes.columns):
        for i, r in S_nodes.iterrows():
            try:
                idx2ll[int(i)] = (float(r["lon"]), float(r["lat"]))
            except Exception:
                pass

    def _resolve(v) -> Optional[LonLat]:
        if v is None:
            return None
        # already lonlat-like
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                return (float(v[0]), float(v[1]))
            except Exception:
                return None
        # index
        try:
            iv = int(v)
        except Exception:
            return None
        return idx2ll.get(iv)

    # Normalize edge container to an iterable of 2-endpoints.
    if isinstance(S_edges, pd.DataFrame):
        if {"u", "v"}.issubset(S_edges.columns):
            take_iter = S_edges[["u", "v"]].head(int(max_edges)).itertuples(index=False, name=None)
            total_edges = len(S_edges)
        elif {"a", "b"}.issubset(S_edges.columns):
            take_iter = S_edges[["a", "b"]].head(int(max_edges)).itertuples(index=False, name=None)
            total_edges = len(S_edges)
        else:
            take_iter = []
            total_edges = len(S_edges)
    else:
        take_iter = list(S_edges)[: int(max_edges)]
        total_edges = len(S_edges)

    fgE = folium.FeatureGroup(name=f"S_edges ({min(int(max_edges), total_edges)}/{total_edges})", show=False)
    drawn = 0
    for uv in take_iter:
        if not isinstance(uv, (list, tuple)) or len(uv) < 2:
            continue
        a = _resolve(uv[0])
        b = _resolve(uv[1])
        if a is None or b is None:
            continue
        if not (_in_bbox(a, bbox_ll) and _in_bbox(b, bbox_ll)):
            continue
        a_v = (_lon_viz(a[0], bbox_ll), a[1])
        b_v = (_lon_viz(b[0], bbox_ll), b[1])
        folium.PolyLine([[a_v[1], a_v[0]], [b_v[1], b_v[0]]], weight=2, opacity=0.8, color="#2b8cbe").add_to(fgE)
        drawn += 1
    fgE.add_to(m)

def add_ring_layers(
    m: folium.Map,
    out: Dict[str, Any],
    *,
    show_e: bool = True,
    show_t: bool = True,
    e_node_sample: Optional[int] = None,
    t_node_sample: Optional[int] = None,
    max_e_edges: int = 4000,
    max_t_edges: int = 4000,
    show: bool = True,
    bbox_ll: Optional[BBoxLL] = None,
) -> None:
    if bbox_ll is None:
        bbox_ll = getattr(m, "_routing_bbox_ll", None)

    rg = out.get("ring_graph", {}) or {}
    E_nodes = rg.get("E_nodes")
    T_nodes = rg.get("T_nodes")
    E_edges = rg.get("E_edges")
    T_edges = rg.get("T_edges")

    def _nodes_layer(df: pd.DataFrame, title: str, sample: Optional[int], radius: int, color: str):
        dd = df
        if sample is not None and len(dd) > int(sample):
            dd = dd.sample(int(sample), random_state=7)
        fg = folium.FeatureGroup(name=f"{title} ({len(dd)}/{len(df)})", show=show)
        for _, r in dd.iterrows():
            p = (float(r.get("lon")), float(r.get("lat")))
            if not _in_bbox(p, bbox_ll):
                continue
            folium.CircleMarker([p[1], _lon_viz(p[0], bbox_ll)], radius=radius, color=color).add_to(fg)
        fg.add_to(m)

    def _edge_layer(edges: pd.DataFrame, nodes: pd.DataFrame, title: str, max_edges: int, color: str):
        # node_id -> lonlat
        nid2ll = {}
        if "node_id" in nodes.columns:
            nid2ll = {int(r["node_id"]): (float(r.get("lon")), float(r.get("lat"))) for _, r in nodes.iterrows()}
        take = edges.head(int(max_edges))
        fg = folium.FeatureGroup(name=f"{title} ({len(take)}/{len(edges)})", show=False)
        for _, r in take.iterrows():
            try:
                u = int(r.get("u"))
                v = int(r.get("v"))
            except Exception:
                continue
            a = nid2ll.get(u)
            b = nid2ll.get(v)
            if a is None or b is None:
                continue
            if not (_in_bbox(a, bbox_ll) and _in_bbox(b, bbox_ll)):
                continue
            a_v = (_lon_viz(a[0], bbox_ll), a[1])
            b_v = (_lon_viz(b[0], bbox_ll), b[1])
            folium.PolyLine([[a_v[1], a_v[0]], [b_v[1], b_v[0]]], weight=2, opacity=0.7, color=color).add_to(fg)
        fg.add_to(m)

    if show_e and isinstance(E_nodes, pd.DataFrame) and len(E_nodes) > 0:
        _nodes_layer(E_nodes, "E_nodes", e_node_sample, radius=2, color="#3182bd")
    if show_t and isinstance(T_nodes, pd.DataFrame) and len(T_nodes) > 0:
        _nodes_layer(T_nodes, "T_nodes", t_node_sample, radius=2, color="#31a354")

    if show_e and isinstance(E_edges, pd.DataFrame) and isinstance(E_nodes, pd.DataFrame):
        _edge_layer(E_edges, E_nodes, "E_edges", max_e_edges, color="#6baed6")
    if show_t and isinstance(T_edges, pd.DataFrame) and isinstance(T_nodes, pd.DataFrame):
        _edge_layer(T_edges, T_nodes, "T_edges", max_t_edges, color="#74c476")

def add_connector_layers(
    m: folium.Map,
    out: Dict[str, Any],
    *,
    show_et: bool = True,
    show_tgate_sea: bool = True,
    max_et: int = 2000,
    max_tgate: int = 2000,
    show: bool = True,
    bbox_ll: Optional[BBoxLL] = None,
) -> None:
    if bbox_ll is None:
        bbox_ll = getattr(m, "_routing_bbox_ll", None)

    rg = out.get("ring_graph", {}) or {}
    E_nodes = rg.get("E_nodes")
    T_nodes = rg.get("T_nodes")
    ET = rg.get("ET_edges")
    dfTG = out.get("tgate_sea_connectors")
    S_nodes = out.get("S_nodes")

    e_map: Dict[int, LonLat] = {}
    t_map: Dict[int, LonLat] = {}
    if isinstance(E_nodes, pd.DataFrame) and "node_id" in E_nodes.columns:
        e_map = {int(r["node_id"]): (float(r.get("lon")), float(r.get("lat"))) for _, r in E_nodes.iterrows()}
    if isinstance(T_nodes, pd.DataFrame) and "node_id" in T_nodes.columns:
        t_map = {int(r["node_id"]): (float(r.get("lon")), float(r.get("lat"))) for _, r in T_nodes.iterrows()}

    # Sea node index -> lonlat
    s_map: Dict[int, LonLat] = {}
    if isinstance(S_nodes, pd.DataFrame) and {"lon", "lat"}.issubset(S_nodes.columns):
        for i, r in S_nodes.iterrows():
            try:
                s_map[int(i)] = (float(r["lon"]), float(r["lat"]))
            except Exception:
                pass

    if show_et and isinstance(ET, pd.DataFrame) and e_map and t_map:
        take = ET.head(int(max_et))
        fg = folium.FeatureGroup(name=f"E<->T ({len(take)}/{len(ET)})", show=False)
        for _, r in take.iterrows():
            try:
                u = int(r.get("u"))
                v = int(r.get("v"))
            except Exception:
                continue
            a = e_map.get(u)
            b = t_map.get(v)
            if a is None or b is None:
                continue
            if not (_in_bbox(a, bbox_ll) and _in_bbox(b, bbox_ll)):
                continue
            a_v = (_lon_viz(a[0], bbox_ll), a[1])
            b_v = (_lon_viz(b[0], bbox_ll), b[1])
            folium.PolyLine([[a_v[1], a_v[0]], [b_v[1], b_v[0]]], weight=2, opacity=0.85, color="#756bb1").add_to(fg)
        fg.add_to(m)

    if show_tgate_sea and isinstance(dfTG, pd.DataFrame) and len(dfTG) > 0 and t_map and s_map:
        take = dfTG.head(int(max_tgate))
        fg = folium.FeatureGroup(name=f"Tgate->Sea ({len(take)}/{len(dfTG)})", show=False)
        for _, r in take.iterrows():
            try:
                t_id = int(r.get("t_node_id"))
                sea_idx = int(r.get("sea_idx"))
            except Exception:
                continue
            a = t_map.get(t_id)
            b = s_map.get(sea_idx)
            if a is None or b is None:
                continue
            if not (_in_bbox(a, bbox_ll) and _in_bbox(b, bbox_ll)):
                continue
            a_v = (_lon_viz(a[0], bbox_ll), a[1])
            b_v = (_lon_viz(b[0], bbox_ll), b[1])
            folium.PolyLine([[a_v[1], a_v[0]], [b_v[1], b_v[0]]], weight=2, opacity=0.85, color="#e6550d").add_to(fg)
        fg.add_to(m)

def finalize_map(m: folium.Map, *, html_path: str) -> str:
    m.add_child(folium.LayerControl(collapsed=False))

    # add all/none toggle buttons
    add_select_all_none_layer_control(m, position="topright", all_text="ALL", none_text="NONE")

    m.save(html_path)
    return html_path



__all__ = [
    "make_base_map",
    "add_points_layer",
    "add_path_layer",
    "add_sea_layers",
    "add_ring_layers",
    "add_connector_layers",
    "finalize_map",
]
