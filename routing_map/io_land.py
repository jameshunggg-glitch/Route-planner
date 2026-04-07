# routing_map/io_land.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Iterable
from shapely.ops import unary_union
import pandas as pd


import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.base import BaseGeometry

def _split_bbox_dateline(
    bbox_ll: Tuple[float, float, float, float]
) -> List[Tuple[float, float, float, float]]:
    min_lon, min_lat, max_lon, max_lat = [float(x) for x in bbox_ll]
    if min_lon <= max_lon:
        return [(min_lon, min_lat, max_lon, max_lat)]
    # crossing dateline
    return [
        (min_lon, min_lat, 180.0, max_lat),
        (-180.0,  min_lat, max_lon, max_lat),
    ]


def _bbox_geom_from_parts(
    parts: List[Tuple[float, float, float, float]]
):
    # union of one or two boxes (covers dateline-crossing window correctly)
    return unary_union([box(*p) for p in parts])


def _iter_polygons(geom: BaseGeometry) -> Iterable[Polygon]:
    """Yield Polygon parts from Polygon / MultiPolygon / GeometryCollection."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
        return
    if isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            if p is not None and (not p.is_empty):
                yield p
        return
    if isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            yield from _iter_polygons(g)
        return


def load_polys_in_bbox(
    shp_path: Path,
    bbox_ll: Tuple[float, float, float, float],
    *,
    fix_invalid: bool = True,
    debug: bool = False,
) -> List[Polygon]:
    """
    Load land polygons for AOI bbox (lon/lat), WITHOUT clipping to bbox edges.
    Key idea:
      - spatial filter by bbox (fast path if supported)
      - then explode MultiPolygons into Polygon parts
      - keep ONLY parts that intersect bbox (still no clipping)

    This avoids AOI-frame artifacts while preventing far-away MultiPolygon parts
    from leaking into the AOI pipeline (the cause of "nodes appear in Americas").
    """
    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"Land shapefile not found: {shp_path}")

    parts = _split_bbox_dateline(bbox_ll)
    bbox_geom = _bbox_geom_from_parts(parts)


    # ---- read ----

    read_used_bbox = True
    gdfs = []
    try:
        for p in parts:
            g = gpd.read_file(shp_path, bbox=p)  # geopandas bbox: (minx,miny,maxx,maxy)
            if not g.empty:
                gdfs.append(g)
    except Exception:
        # 如果 bbox 讀取整個失敗，就 fallback 讀全檔（跟你原本一致）
        read_used_bbox = False
        gdf = gpd.read_file(shp_path)
    else:
        # bbox 讀取成功：concat
        if not gdfs:
            return []
        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

    if gdf.empty:
        return []
    

    # ---- CRS safety ----
    try:
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        # if CRS missing/odd, we still proceed (best effort)
        pass

    # ---- optional: fix invalid geometries ----
    if fix_invalid:
        try:
            # buffer(0) trick to fix self-intersections etc.
            gdf["geometry"] = gdf.geometry.buffer(0)
        except Exception:
            pass

    # ---- coarse filter by bbox intersection (feature-level) ----
    # IMPORTANT: do not swallow errors silently; if this fails, we would leak global land.
    try:
        gdf = gdf[gdf.geometry.intersects(bbox_geom)]
    except Exception as e:
        raise RuntimeError(
            "Failed to bbox-filter land geometries via intersects(). "
            "This would leak global geometries into AOI. "
            f"Original error: {type(e).__name__}: {e}"
        )

    if gdf.empty:
        return []

    # ---- explode MultiPolygons into polygon parts and keep only parts intersecting bbox ----
    kept: List[Polygon] = []
    for geom in gdf.geometry.values:
        for p in _iter_polygons(geom):
            # keep only polygon parts that actually intersect bbox
            # (still NO clipping)
            try:
                if p.intersects(bbox_geom):
                    kept.append(p)
            except Exception:
                # If a single part is problematic, skip it rather than leaking global.
                continue

    if debug:
        print(
            f"[io_land] read_used_bbox={read_used_bbox} "
            f"features_after_filter={len(gdf)} polygon_parts_kept={len(kept)} "
            f"parts={parts} bbox_ll={bbox_ll}"
        )

    return kept
