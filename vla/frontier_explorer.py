"""
Tabletop Frontier Exploration Module.
Autonomous active visual scanning and spatial coverage for the robot arm.
Analogous to 2D/3D frontier exploration in autonomous mobile robot navigation.
"""

import math
from typing import List, Dict, Tuple, Any
from simulation.constants import (
    TABLE_HALF_EXTENTS,
    SAFE_HOVER_Y,
    STACK_ORIGIN,
)

class TabletopFrontierExplorer:
    """
    Manages active exploration of the table work-cell.
    Plans multi-view scanning trajectories to inspect unmapped or occluded table zones.
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

        total = sum(self.explored_cells.values()) / len(self.explored_cells) * 100.0
        self.exploration_pct = min(100.0, total)

        self.current_waypoint_idx += 1
        if self.current_waypoint_idx >= len(self.scan_waypoints):
            self.is_exploring = False
            self.exploration_pct = 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_exploring": self.is_exploring,
            "exploration_pct": round(self.exploration_pct, 1),
            "current_waypoint": self.get_current_waypoint()["name"] if self.is_exploring else "Idle",
            "waypoints_total": len(self.scan_waypoints),
            "waypoint_progress": self.current_waypoint_idx,
            "quadrant_coverage": {k: round(v * 100.0, 1) for k, v in self.explored_cells.items()}
        }
