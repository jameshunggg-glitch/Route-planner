from __future__ import annotations

import gzip
import io
import pickle
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from routing_map.pipeline import (
    run_p2p, GraphConfig, SnapConfig, SimplifyConfig, RunConfig,
)
from routing_map.repairer import RepairConfig
from routing_map import NgzRingBuildConfig
from routing_map.geom_utils import build_projector_from_bbox

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "aoi_cache"
OUT_PATH = CACHE_DIR / "out_global.pkl.gz"
G_PATH = CACHE_DIR / "G_global.pkl.gz"
STATIC_DIR = Path(__file__).resolve().parent / "static"

_state: Dict[str, Any] = {"out": None, "G_base": None, "ready": False}


def _load_gz(p: Path) -> Any:
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def _hydrate_out(out: Dict[str, Any]) -> Dict[str, Any]:
    # pickle 卸載後 proj / sea_kdt 可能是 None，跟 cell 20 同樣補回。
    if out.get("proj") is None:
        out["proj"] = build_projector_from_bbox(out["bbox_ll"])
    if out.get("sea_kdt") is None:
        S = out["S_nodes"]
        out["sea_kdt"] = cKDTree(S[["x_m", "y_m"]].to_numpy(dtype=float))
    return out


app = FastAPI(title="路徑規劃 Debug UI")


@app.on_event("startup")
async def _startup() -> None:
    print(f"[startup] loading cache from {CACHE_DIR}...", flush=True)
    t0 = time.perf_counter()
    out = _load_gz(OUT_PATH)
    G_pack = _load_gz(G_PATH)
    G_base = G_pack["G_base"] if isinstance(G_pack, dict) and "G_base" in G_pack else G_pack
    out = _hydrate_out(out)
    _state["out"] = out
    _state["G_base"] = G_base
    _state["ready"] = True
    print(
        f"[startup] cache loaded in {time.perf_counter()-t0:.1f}s "
        f"(S_nodes={len(out['S_nodes'])}, G_base nodes={G_base.number_of_nodes()})",
        flush=True,
    )


class LonLatModel(BaseModel):
    lon: float = Field(..., ge=-180, le=180)
    lat: float = Field(..., ge=-90, le=90)


class NgzPolygonModel(BaseModel):
    points: List[Tuple[float, float]] = Field(..., min_length=3)


class RouteRequest(BaseModel):
    origin: LonLatModel
    dest: LonLatModel
    ngz_polygons: List[NgzPolygonModel] = Field(default_factory=list)
    ngz_mode: Literal["lenient", "strict", "relocate"] = "lenient"


class RouteResponse(BaseModel):
    path_final: List[Tuple[float, float]]
    path_raw: List[Tuple[float, float]] = []
    length_km: Optional[float] = None
    n_points: int = 0
    n_ngz: int = 0
    error: Optional[str] = None
    log: str = ""


class InitResponse(BaseModel):
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    zoom: int = 5
    ready: bool


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/init", response_model=InitResponse)
async def init() -> InitResponse:
    if not _state["ready"]:
        raise HTTPException(503, "cache not loaded yet")
    bbox = tuple(_state["out"]["bbox_ll"])
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return InitResponse(bbox=bbox, center=center, zoom=5, ready=True)


@app.post("/api/route", response_model=RouteResponse)
async def route(req: RouteRequest) -> RouteResponse:
    if not _state["ready"]:
        raise HTTPException(503, "cache not loaded yet")
    out = _state["out"]
    G_base = _state["G_base"]

    polys = [Polygon(ngz.points) for ngz in req.ngz_polygons]

    graph_cfg = GraphConfig(bbox_ll=out["bbox_ll"])
    snap_cfg = SnapConfig()
    repair_cfg = RepairConfig(debug=False)
    simplify_cfg = SimplifyConfig(enabled=True)
    run_cfg = RunConfig(do_repair=True, do_simplify=True, debug=True)
    ngz_cfg = NgzRingBuildConfig(clearance_m=10_000.0)

    buf = io.StringIO()
    err_str: Optional[str] = None
    res = None
    t0 = time.perf_counter()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            res = run_p2p(
                out,
                (req.origin.lon, req.origin.lat),
                (req.dest.lon, req.dest.lat),
                graph_cfg=graph_cfg,
                snap_cfg=snap_cfg,
                repair_cfg=repair_cfg,
                simplify_cfg=simplify_cfg,
                run_cfg=run_cfg,
                G_in=G_base.copy(),  # 不污染 cache 的 G_base
                ngz_polygons=polys if polys else None,
                ngz_mode=req.ngz_mode,
                ngz_cfg=ngz_cfg,
            )
        if res is not None and res.error:
            err_str = res.error
    except Exception as e:
        err_str = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    elapsed = time.perf_counter() - t0
    log_str = buf.getvalue() + f"\n[server] elapsed {elapsed:.2f}s"

    if err_str and err_str.startswith("ngz_patch_unreachable"):
        log_str += (
            "\n[server] ngz_patch_unreachable detected"
            "\n  原因：local visibility patching disconnected"
            "\n  常見原因：NGZ 範圍過大、anchor 附近視線全被 T-ring 頂點以外的地形 / 其他 NGZ 阻擋"
            "\n\n建議行動："
            "\n  1. 評估該 NGZ 範圍是否合理（可能設太大或包含不必要的水域）"
            "\n  2. 等候情境改善（颱風 / 海況），延後出航"
            "\n  3. 重新考慮起終點或繞行策略"
            "\n  4. 拆分為多段中繼點分別規劃"
        )

    path_final = list(res.path_ll_final) if (res and res.path_ll_final) else []
    path_raw = list(res.path_ll_raw) if (res and res.path_ll_raw) else []
    length_km = (res.lengths_km or {}).get("final") if res else None

    return RouteResponse(
        path_final=[(float(lo), float(la)) for (lo, la) in path_final],
        path_raw=[(float(lo), float(la)) for (lo, la) in path_raw],
        length_km=length_km,
        n_points=len(path_final),
        n_ngz=len(polys),
        error=err_str,
        log=log_str,
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
