from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Points are always (x, y) in meters unless explicitly named *_ll.
XY = Tuple[float, float]
LL = Tuple[float, float]


@dataclass(frozen=True)
class RingBuildConfig:
    """
    Config for envelope/taut rings (metric CRS).

    Notes:
    - smooth_m is expected to have already been applied upstream (A2 smooth union),
      but can be used by a caller that wants to do smoothing inside ring builder.
    - clearance_m is the key safety distance for envelope and taut collision checks.
    """
    smooth_m: float = 0.0
    clearance_m: float = 2000.0

    # Envelope sampling
    ring_sample_km: float = 5.0

    # Envelope point fixing if a sampled point lands inside collision
    point_fix_step_m: float = 250.0
    point_fix_max_iter: int = 40

    # Taut (visibility simplification)
    taut_window_size: int = 80
    taut_max_tries: int = 8
    cut_strategy: str = "best_gap"  # "best_gap" (recommended)
    taut_use_clearance_buffer: bool = True
    taut_collision_buffer_m: Optional[float] = None

    # Filter
    min_island_area_km2: float = 5.0
    min_ring_length_km: float = 20.0


@dataclass
class RingResult:
    ring_id: int
    is_mainland: bool

    # Envelope
    envelope_poly_m: Any  # shapely Polygon
    envelope_pts_m: List[XY] = field(default_factory=list)

    # Taut (subsequence of envelope_pts_m)
    taut_pts_m: List[XY] = field(default_factory=list)

    # Optional lon/lat forms for visualization (filled by caller that has proj)
    envelope_ll: Optional[List[LL]] = None
    taut_ll: Optional[List[LL]] = None

    stats: Dict[str, Any] = field(default_factory=dict)
