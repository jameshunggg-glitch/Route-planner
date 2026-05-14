from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from .types import LonLat, BBoxLL
from dataclasses import dataclass, field
from typing import List
from routing_map.ring_types import RingBuildConfig  


@dataclass
class LandConfig:
    shp_path: Path
    buffer_km: float = 20.0
    avoid_km: float = 5.0
    collision_safety_km: float = 2.0
    precision_grid_m: float = 5.0  # set_precision grid size

@dataclass
class AoiConfig:
    origin_ll: Optional[LonLat] = None
    dest_ll: Optional[LonLat] = None
    bbox_ll: Optional[BBoxLL] = None
    pad_deg: float = 2.0

@dataclass
class SmoothConfig:
    a2_smooth_km: float = 5.0
    a2_tol_km: float = 8.0

@dataclass
class CChainConfig:
    c_step_km: float = 20.0
    round_decimals: int = 5
    island_area_min_km2: float = 20.0

@dataclass
class SeaConfig:
    # Gate-B v1 parameters
    r_max_km: float = 300.0
    candidate_top_n: int = 40
    k_connect: int = 3
    deg_min: int = 1
    aoi_pad_deg: float = 3.0
    use_largest_component_only: bool = True

@dataclass
class RoutingMapConfig:
    aoi: AoiConfig
    land: LandConfig
    smooth: SmoothConfig = SmoothConfig()
    cchain: CChainConfig = CChainConfig()  # 保留，因為 C-node 還在用
    sea: SeaConfig = SeaConfig()
    rings: Optional[RingBuildConfig] = None


@dataclass
class NgzRingBuildConfig:
    """NGZ 專用環建構參數。預設值參考 RingBuildConfig 但 NGZ 通常需要不同 clearance。"""
    clearance_m: float = 5_000.0
    ring_sample_km: float = 5.0
    taut_window_size: int = 16
    taut_max_tries: int = 8
    point_fix_step_m: float = 1_000.0
    point_fix_max_iter: int = 20
    min_island_area_km2: float = 0.0
    min_ring_length_km: float = 0.0
    visibility_k_sea: int = 8
    visibility_k_land_t: int = 4
    visibility_max_dist_km: float = 200.0
    group_merge_eps_m: float = 1_000.0
    densify_max_deg: float = 0.5  # NEW: pre-projection edge densify segment length (degrees)

