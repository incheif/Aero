"""
Unit tests for Semantic Spatial Mapper, Geometric Support Inference & Target Grounding.
Verifies Stage 3 & 4 requirements without simulator ground-truth dependencies.
"""

import pytest
from perception.models import TrackedObject
from simulation.constants import TABLE_TOP_Y, CUBE_SIZE, CUBE_HALF, STACK_ORIGIN
from vla.semantic_mapper import SemanticSpatialMapper
from vla.target_grounding import TargetGrounder


class TestSemanticMapperAndTargetGrounding:
    def test_01_entities_created_from_perception_tracks(self):
        """Verifies mapper ingests pure perception tracks and creates SemanticEntity instances."""
        mapper = SemanticSpatialMapper()

        tracks = [
            TrackedObject(
                track_id="track_1",
                class_name="block",
                color_name="red",
                color_rgb=(255, 30, 39),
                position=[0.25, TABLE_TOP_Y + CUBE_HALF, 0.10],
                confidence=0.96
            ),
            TrackedObject(
                track_id="track_2",
                class_name="block",
                color_name="blue",
                color_rgb=(30, 96, 255),
                position=[0.20, TABLE_TOP_Y + CUBE_HALF, -0.15],
                confidence=0.94
            )
        ]

        mapper.update_from_perception(tracks)

        assert "track_1" in mapper.entities
        assert "track_2" in mapper.entities

        e1 = mapper.entities["track_1"]
        assert e1.name == "Red Cube"
        assert e1.semantic_state == "ON_TABLE"
        assert e1.supported_by == "table"
        assert e1.grasp_affordance == 1.0

    def test_02_geometric_stacking_and_support_inference(self):
        """
        Tests that mapper geometrically infers stacking without simulator metadata.
        Base block at altitude Y0; Mid block directly above at Y0 + CUBE_SIZE.
        """
        mapper = SemanticSpatialMapper()

        base_y = TABLE_TOP_Y + CUBE_HALF
        top_y = base_y + CUBE_SIZE

        tracks = [
            # Lower block
            TrackedObject(
                track_id="track_blue",
                class_name="block",
                color_name="blue",
                color_rgb=(30, 96, 255),
                position=[0.45, base_y, 0.20],
                confidence=0.98
            ),
            # Stacked upper block directly on top
            TrackedObject(
                track_id="track_yellow",
                class_name="block",
                color_name="yellow",
                color_rgb=(255, 195, 0),
                position=[0.45, top_y, 0.20],
                confidence=0.95
            ),
        ]

        mapper.update_from_perception(tracks)

        blue_ent = mapper.entities["track_blue"]
        yellow_ent = mapper.entities["track_yellow"]

        # Support inference
        assert yellow_ent.supported_by == "track_blue"
        assert "Yellow Cube" in blue_ent.supporting
        # Affordance: lower block cannot be grasped from above
        assert blue_ent.grasp_affordance == 0.0
        # Upper block can be grasped
        assert yellow_ent.grasp_affordance > 0.5

    def test_03_target_grounder_resolves_semantic_queries(self):
        """Verifies TargetGrounder maps natural language queries to tracked entities."""
        mapper = SemanticSpatialMapper()

        base_y = TABLE_TOP_Y + CUBE_HALF
        tracks = [
            TrackedObject(track_id="t1", class_name="block", color_name="red", color_rgb=(255,0,0), position=[0.20, base_y, 0.0]),
            TrackedObject(track_id="t2", class_name="block", color_name="green", color_rgb=(0,255,0), position=[0.30, base_y + CUBE_SIZE, 0.10]),
        ]
        mapper.update_from_perception(tracks)

        grounder = TargetGrounder(mapper)

        # 1. Color query
        res_red = grounder.ground("Pick up the red block")
        assert res_red.success is True
        assert res_red.target_id == "t1"
        assert res_red.entity.name == "Red Cube"

        # 2. Highest query
        res_high = grounder.ground("Pick the highest block")
        assert res_high.success is True
        assert res_high.target_id == "t2"
        assert res_high.entity.name == "Green Cube"

        # 3. Target pad
        res_pad = grounder.ground("Place it on the target pad")
        assert res_pad.success is True
        assert res_pad.position == STACK_ORIGIN

        # 4. Unknown query
        res_none = grounder.ground("Pick the turquoise cylinder")
        assert res_none.success is False
        assert res_none.status == "NOT_FOUND"

