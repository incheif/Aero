"""
Semantic Spatial Mapper for 3D Tabletop Robot Arm Workcell.
Maintains persistent 3D spatial memory of objects, bounding volumes,
semantic states, grasp affordances, and spatial topological relations.
Operates strictly on estimated perceptual observations and robot state telemetry.
"""

import math
from typing import Dict, List, Any, Optional, Tuple
from simulation.constants import (
    STACK_ORIGIN,
    STACK_TOLERANCE,
    TABLE_TOP_Y,
    CUBE_SIZE,
    CUBE_HALF,
)
from perception.models import TrackedObject
from simulation.sensor_interfaces import RobotStateObservation


class SemanticEntity:
    def __init__(
        self,
        entity_id: str,
        name: str,
        category: str,
        color: str,
        position: List[float],
        bounding_box: Tuple[float, float, float] = (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        confidence: float = 0.95
    ):
        self.id = entity_id
        self.name = name
        self.category = category  # 'block', 'pad', 'tcp', 'obstacle'
        self.color = color
        self.position = list(position)
        self.bounding_box = bounding_box
        self.semantic_state = "ON_TABLE"
        self.grasp_affordance = 1.0  # 1.0 = clear top grasp, 0.0 = occluded or stacked beneath
        self.stability_score = 1.0
        self.supported_by = "table"
        self.supporting: List[str] = []
        self.in_target_zone = False
        self.confidence = confidence
        self.visibility = "VISIBLE"  # "VISIBLE", "OCCLUDED", "LOST"
        self.time_since_seen = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "color": self.color,
            "position": [round(p, 4) for p in self.position],
            "bounding_box": [round(b, 4) for b in self.bounding_box],
            "semantic_state": self.semantic_state,
            "grasp_affordance": round(self.grasp_affordance, 2),
            "stability_score": round(self.stability_score, 2),
            "supported_by": self.supported_by,
            "supporting": self.supporting,
            "in_target_zone": self.in_target_zone,
            "confidence": round(self.confidence, 3),
            "visibility": self.visibility,
        }


class SemanticSpatialMapper:
    """
    Tabletop 3D Semantic Memory & Affordance Graph.
    Integrates pure perception observations (TrackedObject list) and robot state.
    Infers support relations, stacking layers, grasp affordances, and goal conditions geometrically.
    """
    def __init__(self):
        self.entities: Dict[str, SemanticEntity] = {}
        self.spatial_relations: List[Dict[str, str]] = []
        self.total_scanned_area_pct: float = 85.0
        self.init_static_landmarks()

    def init_static_landmarks(self):
        # Target Zone / Stacking Pad Landmark
        self.entities["target_pad"] = SemanticEntity(
            entity_id="target_pad",
            name="Target Zone (Pad)",
            category="pad",
            color="#234B5F",
            position=[STACK_ORIGIN[0], STACK_ORIGIN[1], STACK_ORIGIN[2]],
            bounding_box=(0.096, 0.005, 0.096),
            confidence=1.0
        )
        self.entities["target_pad"].semantic_state = "DESIGNATED_GOAL"
        self.entities["target_pad"].grasp_affordance = 0.0

    def update_from_perception(
        self,
        tracks: List[TrackedObject],
        robot_state: Optional[RobotStateObservation] = None
    ):
        """
        Updates 3D semantic memory using strictly perception tracks and robot telemetry.
        Never accesses simulator ground truth.
        """
        # 1. Update Robot TCP Entity
        if robot_state and robot_state.tcp_pose_estimated:
            tcp_pos = [robot_state.tcp_pose_estimated[0], robot_state.tcp_pose_estimated[1], robot_state.tcp_pose_estimated[2]]
        else:
            tcp_pos = [0.0, 0.85, 0.0]

        if "tcp_gripper" not in self.entities:
            self.entities["tcp_gripper"] = SemanticEntity(
                entity_id="tcp_gripper",
                name="Gripper TCP",
                category="tcp",
                color="#FFFFFF",
                position=tcp_pos,
                bounding_box=(0.04, 0.04, 0.04),
                confidence=1.0
            )
        else:
            self.entities["tcp_gripper"].position = list(tcp_pos)

        # 2. Synchronize dynamic block tracks
        seen_track_ids = set()
        for track in tracks:
            t_id = track.track_id
            seen_track_ids.add(t_id)

            # Name formatting
            c_name = track.color_name.capitalize() if track.color_name else "Unknown"
            name = f"{c_name} Cube"
            color_hex = "#FFFFFF"
            if track.color_rgb:
                color_hex = f"#{track.color_rgb[0]:02X}{track.color_rgb[1]:02X}{track.color_rgb[2]:02X}"

            if t_id not in self.entities:
                self.entities[t_id] = SemanticEntity(
                    entity_id=t_id,
                    name=name,
                    category="block",
                    color=color_hex,
                    position=track.position,
                    bounding_box=track.dimensions,
                    confidence=track.confidence
                )
            ent = self.entities[t_id]
            ent.name = name
            ent.color = color_hex
            ent.position = list(track.position)
            ent.confidence = track.confidence
            ent.visibility = "VISIBLE" if track.state == "ACTIVE" else "OCCLUDED"
            ent.time_since_seen = track.time_since_update

        # Prune or decay entities not present in tracks
        for ent_id in list(self.entities.keys()):
            if ent_id in ("target_pad", "tcp_gripper"):
                continue
            if ent_id not in seen_track_ids:
                ent = self.entities[ent_id]
                ent.time_since_seen += 1
                ent.confidence = max(0.1, ent.confidence * 0.8)
                ent.visibility = "LOST"
                if ent.time_since_seen > 10:
                    del self.entities[ent_id]

        # Reset hierarchy links
        for ent in self.entities.values():
            ent.supporting = []
            ent.supported_by = "table"
            ent.grasp_affordance = 1.0

        # 3. Geometric reasoning: infer support relations from 3D positions
        blocks = [e for e in self.entities.values() if e.category == "block" and e.visibility != "LOST"]
        for ent in blocks:
            # Check if grasped by gripper: close to TCP and gripper effort > 0.5
            dist_to_tcp = math.dist(ent.position, tcp_pos)
            if dist_to_tcp < 0.055 and robot_state and robot_state.gripper_effort > 0.5:
                ent.semantic_state = "IN_GRIPPER"
                ent.supported_by = "tcp_gripper"
                ent.grasp_affordance = 0.0
                continue

            # Check target zone proximity
            dist_to_pad = math.hypot(ent.position[0] - STACK_ORIGIN[0], ent.position[2] - STACK_ORIGIN[2])
            ent.in_target_zone = dist_to_pad < (STACK_TOLERANCE * 1.8)

            # Check support from other blocks (within 0.04m horizontal, altitude difference ~ CUBE_SIZE)
            found_support = False
            for other in blocks:
                if other.id == ent.id:
                    continue
                h_dist = math.hypot(ent.position[0] - other.position[0], ent.position[2] - other.position[2])
                v_diff = ent.position[1] - other.position[1]

                if h_dist < 0.038 and (0.040 <= v_diff <= 0.070):
                    ent.supported_by = other.id
                    other.supporting.append(ent.name)
                    other.grasp_affordance = 0.0
                    found_support = True
                    break

            if not found_support:
                ent.supported_by = "table"

            # Classify semantic state
            if ent.in_target_zone:
                layer = max(0, int(round((ent.position[1] - TABLE_TOP_Y - CUBE_HALF) / CUBE_SIZE)))
                if layer == 0:
                    ent.semantic_state = "STACKED_L0 (Base)"
                elif layer == 1:
                    ent.semantic_state = "STACKED_L1 (Mid)"
                else:
                    ent.semantic_state = f"STACKED_L{layer} (Top)"
                ent.grasp_affordance = 0.85 if not ent.supporting else 0.0
            else:
                ent.semantic_state = "ON_TABLE"
                ent.grasp_affordance = 1.0 if not ent.supporting else 0.0

        # 4. Build natural-language spatial relations
        self.build_spatial_relations()

    def build_spatial_relations(self):
        relations = []
        for ent_id, ent in self.entities.items():
            if ent.category != "block" or ent.visibility == "LOST":
                continue

            if ent.semantic_state == "IN_GRIPPER":
                relations.append({"subject": ent.name, "relation": "is held by", "object": "Gripper TCP"})
            elif ent.in_target_zone:
                if ent.supported_by == "table":
                    relations.append({"subject": ent.name, "relation": "rests on", "object": "Target Pad (Base layer)"})
                elif ent.supported_by in self.entities:
                    relations.append({"subject": ent.name, "relation": "is stacked on top of", "object": self.entities[ent.supported_by].name})
            else:
                dist_to_pad = math.hypot(ent.position[0] - STACK_ORIGIN[0], ent.position[2] - STACK_ORIGIN[2])
                relations.append({
                    "subject": ent.name,
                    "relation": "rests on table",
                    "object": f"{dist_to_pad:.2f}m from Target Pad"
                })

        self.spatial_relations = relations

    def get_semantic_summary(self) -> str:
        """Generates natural language scene description from estimated memory."""
        lines = ["Current 3D Tabletop Semantic Spatial Memory (Perception Estimates):"]
        for ent in self.entities.values():
            if ent.category == "block" and ent.visibility != "LOST":
                pos = f"[{ent.position[0]:.3f}, {ent.position[1]:.3f}, {ent.position[2]:.3f}]"
                lines.append(f"- {ent.name} (TrackID: {ent.id}): State={ent.semantic_state}, Pos={pos}, Affordance={ent.grasp_affordance:.1f}, Conf={ent.confidence:.2f}")
        lines.append(f"- Target Pad: Located at [{STACK_ORIGIN[0]:.3f}, {STACK_ORIGIN[1]:.3f}, {STACK_ORIGIN[2]:.3f}]")
        return "\n".join(lines)

    def get_obstacles_for_motion_planning(self, exclude_entity_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Returns estimated 3D bounding obstacles for collision-avoidance motion planner.
        """
        exclude_set = set(exclude_entity_ids or [])
        obstacles = []
        for ent_id, ent in self.entities.items():
            if ent_id in exclude_set or ent.category != "block" or ent.visibility == "LOST":
                continue
            obstacles.append({
                "id": ent.id,
                "name": ent.name,
                "position": list(ent.position),
                "bounding_box": list(ent.bounding_box),
            })
        return obstacles

    def find_block_by_color(self, color_name: str) -> Optional[SemanticEntity]:
        """Find active block matching color query (case-insensitive)."""
        query = color_name.strip().lower()
        for ent in self.entities.values():
            if ent.category == "block" and ent.visibility != "LOST":
                if query in ent.name.lower() or query in ent.id.lower():
                    return ent
        return None

    def update_from_simulation(self, snapshot: Dict[str, Any]):
        """Legacy reflection adapter preserved for backward compatibility."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "relations": self.spatial_relations,
            "scanned_area_pct": round(self.total_scanned_area_pct, 1),
            "target_pad_pos": STACK_ORIGIN,
        }
