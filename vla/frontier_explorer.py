"""
Frontier Explorer and Workspace Coverage Tracker for AERO Workcell.
Maintains discrete occupancy and visibility grid across tabletop workspace.
Calculates exploratory camera / arm inspection viewpoints when confidence is low or objects are occluded.
Also preserves TabletopFrontierExplorer for existing dashboard compatibility.
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from simulation.constants import TABLE_HALF_EXTENTS, TABLE_TOP_Y, SAFE_HOVER_Y, STACK_ORIGIN

def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


class FrontierExplorer:
    """
    Tabletop 2D coverage grid tracker.
    Represents explored vs unexplored spatial cells.
    """
    def __init__(
        self,
        grid_res: Tuple[int, int] = (10, 10),
        table_half_x: float = TABLE_HALF_EXTENTS[0] * 0.8,
        table_half_z: float = TABLE_HALF_EXTENTS[2] * 0.8,
    ):
        self.rows, self.cols = grid_res
        self.x_bounds = (-table_half_x, table_half_x)
        self.z_bounds = (-table_half_z, table_half_z)
        # 0.0 = completely unknown, 1.0 = fully observed
        self.coverage_grid = np.zeros(grid_res, dtype=np.float32)

    def mark_observed_region(self, center_x: float, center_z: float, radius: float = 0.35):
        """Mark cells within the observation footprint as observed."""
        for r in range(self.rows):
            for c in range(self.cols):
                cx, cz = self._cell_to_metric(r, c)
                dist = np.hypot(cx - center_x, cz - center_z)
                if dist <= radius:
                    weight = max(0.0, 1.0 - (dist / radius) * 0.5)
                    self.coverage_grid[r, c] = min(1.0, self.coverage_grid[r, c] + weight)

    def get_coverage_percentage(self) -> float:
        """Percentage of tabletop workspace observed."""
        return float(np.mean(self.coverage_grid) * 100.0)

    def get_frontier_inspection_pose(self) -> Optional[Tuple[float, float, float]]:
        """
        Finds centroid of least observed region to recommend an inspection viewpoint.
        """
        min_idx = np.unravel_index(np.argmin(self.coverage_grid), self.coverage_grid.shape)
        if self.coverage_grid[min_idx] > 0.85:
            # Entire table is well-observed
            return None
        fx, fz = self._cell_to_metric(min_idx[0], min_idx[1])
        return (float(fx), float(SAFE_HOVER_Y), float(fz))

    def _cell_to_metric(self, r: int, c: int) -> Tuple[float, float]:
        frac_x = (c + 0.5) / float(self.cols)
        frac_z = (r + 0.5) / float(self.rows)
        x = self.x_bounds[0] + frac_x * (self.x_bounds[1] - self.x_bounds[0])
        z = self.z_bounds[0] + frac_z * (self.z_bounds[1] - self.z_bounds[0])
        return x, z


class TabletopFrontierExplorer:
    """
    Manages active exploration of the table work-cell for dashboard studio.
    Plans multi-view scanning trajectories to inspect unmapped or occluded table zones.
    Maintains a 10x10 metric occupancy & exploration grid.
    """
    def __init__(self):
        self.is_exploring: bool = False
        self.current_waypoint_idx: int = 0
        self.explored_cells: Dict[str, float] = {
            "quadrant_nw": 0.25, # Northwest
            "quadrant_ne": 0.25, # Northeast
            "quadrant_sw": 0.25, # Southwest
            "quadrant_se": 0.25, # Southeast (Pad area)
        }
        self.exploration_pct: float = 25.0
        self.rows: int = 10
        self.cols: int = 10
        # 0: UNKNOWN, 1: FREE (observed), 2: OCCUPIED
        self.grid = np.zeros((10, 10), dtype=np.int32)
        # Mark center workspace as initially observed free space
        for r in range(2, 8):
            for c in range(2, 8):
                self.grid[r, c] = 1

        # Multi-point exploration scanning waypoints (safe hover transit)
        self.scan_waypoints: List[Dict[str, Any]] = [
            {"name": "Center Table Overview", "target": (0.25, SAFE_HOVER_Y, 0.0), "quadrant": "quadrant_nw"},
            {"name": "Left Table Boundary Scan", "target": (0.15, SAFE_HOVER_Y, -0.22), "quadrant": "quadrant_sw"},
            {"name": "Far Edge Inspection", "target": (0.38, SAFE_HOVER_Y, -0.15), "quadrant": "quadrant_nw"},
            {"name": "Center-Right Table Scan", "target": (0.35, SAFE_HOVER_Y, 0.12), "quadrant": "quadrant_ne"},
            {"name": "Target Pad & Landing Zone Verification", "target": (STACK_ORIGIN[0], SAFE_HOVER_Y, STACK_ORIGIN[2]), "quadrant": "quadrant_se"},
            {"name": "Home Overview Survey", "target": (0.20, SAFE_HOVER_Y, 0.0), "quadrant": "quadrant_nw"},
        ]

    def start_exploration(self):
        """Initiates autonomous scanning sequence."""
        self.is_exploring = True
        self.current_waypoint_idx = 0

    def get_current_waypoint(self) -> Dict[str, Any]:
        if not self.is_exploring or self.current_waypoint_idx >= len(self.scan_waypoints):
            return self.scan_waypoints[0]
        return self.scan_waypoints[self.current_waypoint_idx]

    def advance_waypoint(self):
        """Advances to the next exploration waypoint and increases coverage."""
        if not self.is_exploring:
            return

        cur = self.scan_waypoints[self.current_waypoint_idx]
        quad = cur.get("quadrant", "quadrant_nw")
        self.explored_cells[quad] = min(1.0, self.explored_cells[quad] + 0.35)

        tgt = cur.get("target", (0.25, SAFE_HOVER_Y, 0.0))
        self.mark_region(tgt[0], tgt[2], radius=0.22)

        total = sum(self.explored_cells.values()) / len(self.explored_cells) * 100.0
        self.exploration_pct = min(100.0, total)

        self.current_waypoint_idx += 1
        if self.current_waypoint_idx >= len(self.scan_waypoints):
            self.is_exploring = False
            self.exploration_pct = 100.0
            # Fully explored
            self.grid[self.grid == 0] = 1

    def mark_region(self, cx: float, cz: float, radius: float = 0.20):
        """Marks cells around coordinate as observed free space."""
        half_x = TABLE_HALF_EXTENTS[0] * 0.9
        half_z = TABLE_HALF_EXTENTS[2] * 0.9
        for r in range(self.rows):
            for c in range(self.cols):
                gx = -half_x + (c + 0.5) / self.cols * (2 * half_x)
                gz = -half_z + (r + 0.5) / self.rows * (2 * half_z)
                if np.hypot(gx - cx, gz - cz) <= radius and self.grid[r, c] == 0:
                    self.grid[r, c] = 1

    def update_occupancy_from_tracks(self, tracks: List[Any]):
        """Sets cells with perceived objects as OCCUPIED (2)."""
        half_x = TABLE_HALF_EXTENTS[0] * 0.9
        half_z = TABLE_HALF_EXTENTS[2] * 0.9
        # Reset previous occupied to free if still within observed area
        self.grid[self.grid == 2] = 1
        for t in tracks:
            pos = getattr(t, "position", None) or (t.get("position") if isinstance(t, dict) else None)
            if pos:
                col = int(clamp((pos[0] + half_x) / (2 * half_x) * self.cols, 0, self.cols - 1))
                row = int(clamp((pos[2] + half_z) / (2 * half_z) * self.rows, 0, self.rows - 1))
                self.grid[row, col] = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_exploring": self.is_exploring,
            "exploration_pct": round(self.exploration_pct, 1),
            "current_waypoint": self.get_current_waypoint()["name"] if self.is_exploring else "Idle",
            "waypoints_total": len(self.scan_waypoints),
            "waypoint_progress": self.current_waypoint_idx,
            "quadrant_coverage": {k: round(v * 100.0, 1) for k, v in self.explored_cells.items()},
            "occupancy_grid": self.grid.tolist(),
        }
