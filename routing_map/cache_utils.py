"""
routing_map.cache_utils

Cache helpers for:
- out (result of build_aoi)
- G_base (NetworkX base routing graph built from out)

Design goals:
- Stable pickles: store only serializable data; rebuild accelerators after load.
- Cache is optional: caller decides whether to use it (Solution A).
- G_base should be treated as immutable; routing runs must use a copy because inject mutates graphs.

Default cache layout (single "global" cache):
  cache_dir/
    out_global.pkl.gz
    G_global.pkl.gz
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import gzip
import hashlib
import json
import pickle
import datetime

import networkx as nx

def ensure_graph_edge_masks(G: nx.Graph, *, hard_lat_cap_deg: float = 70.0) -> nx.Graph:
    """Backfill edge mask attributes for cached graphs built with older code."""
    from .routing_graph import infer_layer_mask_from_etype, B_HIGH_LAT
    hard_lat = float(hard_lat_cap_deg)
    for u, v, attr in G.edges(data=True):
        try:
            layer = int(attr.get("layer_mask") or 0)
            if layer == 0:
                layer = int(infer_layer_mask_from_etype(str(attr.get("etype", ""))))
            attr["layer_mask"] = layer

            lat_max = attr.get("lat_max_abs", None)
            if lat_max is None:
                try:
                    lat_max = max(abs(float(G.nodes[u].get("lat"))), abs(float(G.nodes[v].get("lat"))))
                except Exception:
                    lat_max = None
            attr["lat_max_abs"] = (float(lat_max) if lat_max is not None else None)

            ban = int(attr.get("ban_mask") or 0)
            if lat_max is not None and float(lat_max) > hard_lat:
                ban |= int(B_HIGH_LAT)
            attr["ban_mask"] = int(ban)
        except Exception:
            # best-effort only
            continue
    return G


# Optional imports; only used in rebuild step.
try:
    from shapely.prepared import prep  # type: ignore
except Exception:  # pragma: no cover
    prep = None

try:
    from sklearn.neighbors import KDTree  # type: ignore
except Exception:  # pragma: no cover
    KDTree = None

# local utilities
from .geom_utils import build_projector_from_bbox


OUT_CACHE_VERSION = "out_cache_v1"
G_CACHE_VERSION = "graph_cache_v1"


def _safe_getattr(obj: Any, dotted: str, default: Any = None) -> Any:
    cur = obj
    for part in dotted.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    """Best-effort conversion of cfg object into JSON-serializable dict."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    # pydantic v2
    md = getattr(cfg, "model_dump", None)
    if callable(md):
        try:
            return md()
        except Exception:
            pass
    # dataclass
    try:
        if is_dataclass(cfg):
            return asdict(cfg)
    except Exception:
        pass
    # fallback: shallow attrs we care about
    out: Dict[str, Any] = {}
    for k in ["aoi", "land", "rings", "sea"]:
        v = getattr(cfg, k, None)
        if v is None:
            continue
        mdv = getattr(v, "model_dump", None)
        if callable(mdv):
            try:
                out[k] = mdv()
                continue
            except Exception:
                pass
        try:
            if is_dataclass(v):
                out[k] = asdict(v)
                continue
        except Exception:
            pass
        out[k] = repr(v)
    return out


def cfg_fingerprint(cfg: Any, *, extra: Optional[Dict[str, Any]] = None) -> str:
    """Fingerprint for cache validation. Small and stable."""
    cfgd = cfg_to_dict(cfg)
    payload = {"cfg": cfgd, "extra": extra or {}}
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _gzip_pickle_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _gzip_pickle_load(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _make_out_payload(out: Dict[str, Any], *, cache_version: str, fp: str) -> Dict[str, Any]:
    # Do NOT store accelerator objects; they are cheap to rebuild and brittle to pickle.
    DROP = {"collision_prep", "sea_kdt", "sea_kdtree"}

    out_save: Dict[str, Any] = {}
    for k, v in out.items():
        if k in DROP:
            continue
        out_save[k] = v

    # proj can be rebuilt; store bbox only. We'll rebuild proj on load.
    if "proj" in out_save:
        out_save.pop("proj", None)

    # If layers is a dict but contains unpicklable objects, keep only required geometry keys.
    layers = out_save.get("layers", None)
    if layers is not None:
        try:
            pickle.dumps(layers, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            if isinstance(layers, dict):
                keep = {}
                for kk in ["UNION_M", "COLLISION_M"]:
                    if kk in layers:
                        keep[kk] = layers[kk]
                out_save["layers"] = keep
            else:
                out_save.pop("layers", None)

    meta = {
        "version": cache_version,
        "fingerprint": fp,
        "created_at": _now_iso(),
        "bbox_ll": out.get("bbox_ll", None),
    }
    cfg_obj = out.get("cfg", None)
    meta["cfg_dict"] = cfg_to_dict(cfg_obj)
    # keep a short repr for debugging
    try:
        meta["cfg_repr"] = repr(cfg_obj)
    except Exception:
        meta["cfg_repr"] = "<unrepr>"

    # Store meta alongside saved out
    out_save["_cache_meta"] = meta
    return out_save


def save_out_cache(
    out: Dict[str, Any],
    *,
    cache_dir: str | Path = "aoi_cache",
    filename: str = "out_global.pkl.gz",
) -> Path:
    cache_dir = Path(cache_dir)
    cfg_obj = out.get("cfg", None)
    fp = cfg_fingerprint(cfg_obj, extra={"bbox_ll": out.get("bbox_ll", None), "ver": OUT_CACHE_VERSION})
    payload = _make_out_payload(out, cache_version=OUT_CACHE_VERSION, fp=fp)
    path = cache_dir / filename
    _gzip_pickle_dump(payload, path)
    return path


def load_out_cache(
    cfg: Any,
    *,
    cache_dir: str | Path = "aoi_cache",
    filename: str = "out_global.pkl.gz",
    strict: bool = True,
) -> Optional[Dict[str, Any]]:
    cache_dir = Path(cache_dir)
    path = cache_dir / filename
    if not path.exists():
        return None

    out = _gzip_pickle_load(path)
    meta = out.get("_cache_meta", {}) if isinstance(out, dict) else {}
    if not isinstance(meta, dict):
        return None

    if meta.get("version") != OUT_CACHE_VERSION:
        return None

    expected_fp = cfg_fingerprint(cfg, extra={"bbox_ll": meta.get("bbox_ll", None), "ver": OUT_CACHE_VERSION})
    if strict and meta.get("fingerprint") != expected_fp:
        return None

    # Rebuild projector
    try:
        bbox_ll = out.get("bbox_ll", None)
        if bbox_ll is not None:
            out["proj"] = build_projector_from_bbox(bbox_ll)
    except Exception:
        pass

    # Rebuild collision_prep (from layers["COLLISION_M"] if available)
    if prep is not None:
        try:
            layers = out.get("layers", None)
            if isinstance(layers, dict) and ("COLLISION_M" in layers):
                out["collision_prep"] = prep(layers["COLLISION_M"])
        except Exception:
            pass

    # Rebuild sea_kdt from S_nodes if present
    if KDTree is not None:
        try:
            S_nodes = out.get("S_nodes", None)
            if S_nodes is not None and hasattr(S_nodes, "__getitem__") and ("x_m" in S_nodes.columns) and ("y_m" in S_nodes.columns):
                out["sea_kdt"] = KDTree(S_nodes[["x_m", "y_m"]].to_numpy(dtype=float))
        except Exception:
            pass

    return out


def get_out(
    cfg: Any,
    *,
    cache_dir: str | Path = "aoi_cache",
    use_cache: bool = True,
    strict: bool = True,
    build_fn=None,
) -> Dict[str, Any]:
    """Return out. If use_cache, load; otherwise build. If cache miss, build and save.

    build_fn: function(cfg)->out. If None, imported lazily from routing_map.build_aoi.
    """
    if use_cache:
        out = load_out_cache(cfg, cache_dir=cache_dir, strict=strict)
        if out is not None:
            return out

    if build_fn is None:
        from .build_aoi import build_aoi  # local import
        build_fn = build_aoi

    out = build_fn(cfg)
    # Ensure cfg is attached (some code expects it)
    out["cfg"] = cfg
    if use_cache:
        save_out_cache(out, cache_dir=cache_dir)
    return out


def save_graph_cache(
    G_base: nx.Graph,
    *,
    cfg: Any,
    graph_build_args: Dict[str, Any],
    cache_dir: str | Path = "aoi_cache",
    filename: str = "G_global.pkl.gz",
    stats: Any = None,
) -> Path:
    cache_dir = Path(cache_dir)
    fp = cfg_fingerprint(cfg, extra={"graph_build_args": graph_build_args, "ver": G_CACHE_VERSION})
    payload = {
        "version": G_CACHE_VERSION,
        "fingerprint": fp,
        "created_at": _now_iso(),
        "graph_build_args": graph_build_args,
        "stats": stats,
        "G_base": G_base,
    }
    path = cache_dir / filename
    _gzip_pickle_dump(payload, path)
    return path


def load_graph_cache(
    *,
    cfg: Any,
    graph_build_args: Dict[str, Any],
    cache_dir: str | Path = "aoi_cache",
    filename: str = "G_global.pkl.gz",
    strict: bool = True,
) -> Optional[Tuple[nx.Graph, Dict[str, Any]]]:
    cache_dir = Path(cache_dir)
    path = cache_dir / filename
    if not path.exists():
        return None

    payload = _gzip_pickle_load(path)
    if not isinstance(payload, dict):
        return None

    if payload.get("version") != G_CACHE_VERSION:
        return None

    expected_fp = cfg_fingerprint(cfg, extra={"graph_build_args": graph_build_args, "ver": G_CACHE_VERSION})
    if strict and payload.get("fingerprint") != expected_fp:
        return None

    G_base = payload.get("G_base", None)
    if G_base is None:
        return None

    return G_base, payload


def get_graph(
    out: Dict[str, Any],
    *,
    cfg: Any,
    graph_build_args: Dict[str, Any],
    cache_dir: str | Path = "aoi_cache",
    use_cache: bool = True,
    strict: bool = True,
    build_fn=None,
) -> nx.Graph:
    """Return immutable base graph G_base (do NOT inject into it; copy before routing).

    build_fn: function(out, **graph_build_args)->(G, stats). If None, imported from routing_map.routing_graph.build_base_graph
    """
    if use_cache:
        hit = load_graph_cache(cfg=cfg, graph_build_args=graph_build_args, cache_dir=cache_dir, strict=strict)
        if hit is not None:
            G_base, _meta = hit
            ensure_graph_edge_masks(G_base, hard_lat_cap_deg=float(graph_build_args.get('hard_lat_cap_deg', 70.0)))
            return G_base

    if build_fn is None:
        from .routing_graph import build_base_graph  # local import
        build_fn = build_base_graph

    G_base, stats = build_fn(out, **graph_build_args)
    ensure_graph_edge_masks(G_base, hard_lat_cap_deg=float(graph_build_args.get('hard_lat_cap_deg', 70.0)))
    if use_cache:
        save_graph_cache(G_base, cfg=cfg, graph_build_args=graph_build_args, cache_dir=cache_dir, stats=stats)
    return G_base


def copy_graph_for_run(G_base: nx.Graph) -> nx.Graph:
    """Return a per-run working copy of G_base (inject mutates the graph)."""
    return G_base.copy()
