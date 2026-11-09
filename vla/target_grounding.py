"""
Semantic Target Grounding for AERO Workcell.
Resolves natural-language references to tracked entities in the estimated scene map.
Replaces hard-coded coordinates with perception-grounded entities.
"""

import math
from typing import Optional, List, Dict, Tuple
from vla.semantic_mapper import SemanticSpatialMapper, SemanticEntity
from simulation.constants import STACK_ORIGIN, SAFE_CORRIDOR_STAGING_SLOT


class GroundingResult:
    def __init__(
        self,
        success: bool,
        entity: Optional[SemanticEntity] = None,
        position: Optional[Tuple[float, float, float]] = None,
        target_id: Optional[str] = None,
        status: str = "MATCHED", # "MATCHED", "AMBIGUOUS", "NOT_FOUND", "OCCLUDED"
        message: str = ""
    ):
        self.success = success
        self.entity = entity
        self.position = position
        self.target_id = target_id
        self.status = status
        self.message = message


class TargetGrounder:
    """
    Grounds high-level semantic descriptions to tracked entities in the semantic map.
    """
    def __init__(self, mapper: SemanticSpatialMapper):
        self.mapper = mapper

    def ground(self, description: str, tcp_pos: Optional[Tuple[float, float, float]] = None) -> GroundingResult:
        """
        Resolve natural language target description to a tracked entity or landmark coordinate.
        """
        desc = description.strip().lower()
        active_blocks = [e for e in self.mapper.entities.values() if e.category == "block" and e.visibility != "LOST"]

        # 1. Direct Color Matches
        colors = ["red", "blue", "yellow", "green", "orange", "purple"]
        matched_color = None
        for c in colors:
            if c in desc:
                matched_color = c
                break

        if matched_color:
            color_matches = [b for b in active_blocks if matched_color in b.name.lower()]
            if len(color_matches) == 1:
                b = color_matches[0]
                return GroundingResult(
                    success=True,
                    entity=b,
                    position=tuple(b.position),
                    target_id=b.id,
                    status="MATCHED",
                    message=f"Grounded '{description}' to {b.name} at {b.position}"
                )
            elif len(color_matches) > 1:
                # If multiple (e.g. two towers), resolve by context
                if "highest" in desc or "top" in desc:
                    b = max(color_matches, key=lambda x: x.position[1])
                    return GroundingResult(True, b, tuple(b.position), b.id, "MATCHED", f"Resolved highest {matched_color} block: {b.name}")
                return GroundingResult(
                    success=False,
                    status="AMBIGUOUS",
                    message=f"Multiple blocks matched color '{matched_color}'"
                )

        # 2. Geometric Relations / Spatial Properties
        if "highest" in desc or "top" in desc:
            if active_blocks:
                b = max(active_blocks, key=lambda x: x.position[1])
                return GroundingResult(True, b, tuple(b.position), b.id, "MATCHED", f"Resolved highest block: {b.name}")

        if "lowest" in desc or "bottom" in desc:
            if active_blocks:
                b = min(active_blocks, key=lambda x: x.position[1])
                return GroundingResult(True, b, tuple(b.position), b.id, "MATCHED", f"Resolved lowest block: {b.name}")

        if "nearest" in desc or "closest" in desc:
            ref_pos = tcp_pos or [0.0, 0.85, 0.0]
            if active_blocks:
                b = min(active_blocks, key=lambda x: math.dist(x.position, ref_pos))
                return GroundingResult(True, b, tuple(b.position), b.id, "MATCHED", f"Resolved nearest block to TCP: {b.name}")

        # 3. Static Workcell Landmarks
        if "pad" in desc or "target zone" in desc or "stacking area" in desc:
            pad = self.mapper.entities.get("target_pad")
            pos = tuple(pad.position) if pad else STACK_ORIGIN
            return GroundingResult(True, pad, pos, "target_pad", "MATCHED", "Grounded to Target Pad")

        if "staging" in desc or "safe corridor" in desc:
            return GroundingResult(True, None, SAFE_CORRIDOR_STAGING_SLOT, "staging_slot", "MATCHED", "Grounded to Staging Slot")

        return GroundingResult(
            success=False,
            status="NOT_FOUND",
            message=f"Could not ground target description '{description}' to any tracked entity or landmark"
        )

