"""
Semantic Spatial Mapper for 3D Tabletop Robot Arm Workcell.
Maintains persistent 3D spatial memory of objects, bounding volumes,
semantic states, grasp affordances, and spatial topological relations.
Analogous to 3D Semantic SLAM in autonomous navigation.
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

class SemanticEntity:
    def __init__(
        self,
        entity_id: str,
        name: str,
        category: str,
        color: str,
        position: List[float],
        bounding_box: Tuple[float, float, float] = (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
    ):
        self.id = entity_id
        self.name = name
        self.category = category  # 'block', 'pad', 'tcp', 'zone'
        self.color = color
        self.position = list(position)
        self.bounding_box = bounding_box
        self.semantic_state = "ON_TABLE"
        self.grasp_affordance = 1.0  # 1.0 = clear top grasp, 0.0 = occluded or stacked beneath
        self.stability_score = 1.0
        self.supported_by = "table"
        self.supporting = []
        self.in_target_zone = False
        self.confidence = 0.98

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
            "confidence": self.confidence,
        }

class SemanticSpatialMapper:
    """
    Tabletop 3D Semantic Memory & Affordance Graph.
    Continuously updates spatial observations, classifies tower layers,
    calculates clearance affordances, and checks goal states.
    """
    def __init__(self):
        self.entities: Dict[str, SemanticEntity] = {}
        self.spatial_relations: List[Dict[str, str]] = []
        self.total_scanned_area_pct: float = 25.0  # Initial known view
        self.init_static_landmarks()

    def init_static_landmarks(self):
        # Target Zone / Stacking Pad Landmark
        self.entities["target_pad"] = SemanticEntity(
            entity_id="target_pad",
            name="Target Zone (Pad)",
            category="pad",
            color="#234B5F",
            position=[STACK_ORIGIN[0], STACK_ORIGIN[1], STACK_ORIGIN[2]],
            bounding_box=(0.096, 0.005, 0.096)
        )
        self.entities["target_pad"].semantic_state = "DESIGNATED_GOAL"
        self.entities["target_pad"].grasp_affordance = 0.0

    def update_from_simulation(self, snapshot: Dict[str, Any]):
        """Integrates physical telemetry into 3D semantic memory graph."""
        blocks = snapshot.get("blocks", [])
        grasped_id = snapshot.get("grasped_block_id")
        tcp_pos = snapshot.get("tcp", {}).get("position", [0.0, 0.8, 0.0])

        # Register or update TCP entity
        if "tcp_gripper" not in self.entities:
            self.entities["tcp_gripper"] = SemanticEntity(
                entity_id="tcp_gripper",
                name="Gripper TCP",
                category="tcp",
                color="#FFFFFF",
                position=tcp_pos,
                bounding_box=(0.04, 0.04, 0.04)
            )
        else:
            self.entities["tcp_gripper"].position = list(tcp_pos)

        # Clear supporting relationships
        for entity in self.entities.values():
            entity.supporting = []

        # Update dynamic blocks
        for b in blocks:
            b_id = b["id"]
            if b_id not in self.entities:
                self.entities[b_id] = SemanticEntity(
                    entity_id=b_id,
                    name=b.get("name", b_id),
                    category="block",
                    color=b.get("color", "#FFFFFF"),
                    position=b["position"]
                )

            ent = self.entities[b_id]
            ent.position = list(b["position"])
            ent.supported_by = b.get("supported_by", "table")
            ent.in_target_zone = b.get("in_pad", False)

            # Classify semantic state
            if b.get("is_grasped", False):
                ent.semantic_state = "IN_GRIPPER"
                ent.grasp_affordance = 0.0
            elif ent.in_target_zone:
                layer = b.get("stacked_layer", 0)
                if layer == 0:
                    ent.semantic_state = "STACKED_L0 (Base)"
                elif layer == 1:
                    ent.semantic_state = "STACKED_L1 (Mid)"
                else:
                    ent.semantic_state = f"STACKED_L{layer} (Top)"
                ent.grasp_affordance = 0.85
            else:
                ent.semantic_state = "ON_TABLE"
                ent.grasp_affordance = 1.0

        # Compute stacking hierarchy and occlusion
        for b_id, ent in self.entities.items():
            if ent.category != "block":
                continue
            if ent.supported_by in self.entities and ent.supported_by != "table":
                parent = self.entities[ent.supported_by]
                parent.supporting.append(ent.name)
                # If something is resting on this block, top-grasp affordance drops to 0
                parent.grasp_affordance = 0.0

        # Build natural-language spatial relations list
        self.build_spatial_relations()

    def build_spatial_relations(self):
        relations = []
        for ent_id, ent in self.entities.items():
            if ent.category != "block":
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
        """Generates natural language scene description for Google VLA prompt."""
        lines = ["Current 3D Tabletop Semantic Spatial Memory:"]
        for ent in self.entities.values():
            if ent.category == "block":
                pos = f"[{ent.position[0]:.3f}, {ent.position[1]:.3f}, {ent.position[2]:.3f}]"
                lines.append(f"- {ent.name} ({ent.color}): State={ent.semantic_state}, Position={pos}, GraspAffordance={ent.grasp_affordance:.1f}")
        lines.append(f"- Target Pad: Located at [{STACK_ORIGIN[0]:.3f}, {STACK_ORIGIN[1]:.3f}, {STACK_ORIGIN[2]:.3f}]")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "relations": self.spatial_relations,
            "scanned_area_pct": round(self.total_scanned_area_pct, 1),
            "target_pad_pos": STACK_ORIGIN,
        }
