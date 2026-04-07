"""Example main (rings-only) using the refactored modules.

Assumes you already ran build_aoi(...) and have `out`.

This script intentionally focuses on:
- building the graph (Sea + E/T rings + connectors)
- snapping & injecting endpoints
- A* + repair + simplify
- folium visualization

Adjust import paths to match your project layout.
"""

from __future__ import annotations

from pathlib import Path

from routing_map.pipeline import GraphConfig, SnapConfig, SimplifyConfig, RunConfig, run_p2p
from routing_map.viz_layers import (
    make_base_map,
    add_sea_layers,
    add_ring_layers,
    add_connector_layers,
    add_points_layer,
    add_path_layer,
    finalize_map,
)


def run_example(out, origin_ll, dest_ll, *, html_path="aoi_debug_map.html"):
    bbox_ll = out.get("bbox_ll")
    if bbox_ll is None:
        raise ValueError("out['bbox_ll'] is required")

    res = run_p2p(
        out,
        origin_ll,
        dest_ll,
        graph_cfg=GraphConfig(bbox_ll=bbox_ll, max_sea_edges=None, max_ring_edges=None),
        snap_cfg=SnapConfig(),
        simplify_cfg=SimplifyConfig(enabled=True),
        run_cfg=RunConfig(debug=True),
    )

    m = make_base_map(bbox_ll=bbox_ll, zoom_start=6)
    add_sea_layers(m, out, bbox_ll=bbox_ll, edge_limit=4000, show=True)
    add_ring_layers(m, out, bbox_ll=bbox_ll, e_show=True, t_show=True)
    add_connector_layers(m, out, bbox_ll=bbox_ll, et_show=True, tgate_show=True)

    # points
    add_points_layer(m, [origin_ll], name="start_raw", radius=6, show=True)
    add_points_layer(m, [dest_ll], name="end_raw", radius=6, show=True)
    if res.start_ll_snap and res.end_ll_snap:
        add_points_layer(m, [res.start_ll_snap], name="start_snap", radius=5, show=True)
        add_points_layer(m, [res.end_ll_snap], name="end_snap", radius=5, show=True)

    # paths
    if res.path_ll_raw:
        add_path_layer(m, res.path_ll_raw, name="path_raw", weight=3, opacity=0.6, show=False)
    if res.path_ll_repaired:
        add_path_layer(m, res.path_ll_repaired, name="path_repaired", weight=4, opacity=0.7, show=False)
    if res.path_ll_simplified:
        add_path_layer(m, res.path_ll_simplified, name="path_simplified", weight=5, opacity=0.9, show=True)
    elif res.path_ll_final:
        add_path_layer(m, res.path_ll_final, name="path_final", weight=5, opacity=0.9, show=True)

    return finalize_map(m, html_path=html_path)


if __name__ == "__main__":
    raise SystemExit("This is a template. Call run_example(out, origin_ll, dest_ll) from your notebook/script.")
