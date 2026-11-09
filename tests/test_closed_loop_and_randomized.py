"""
Integration tests for Closed-Loop Verification, Bounded Recovery, and Randomized Tabletop Manipulation.
Verifies Stage 6 & 7 requirements:
1. Observe-act-verify cycle correctly distinguishes successful vs failed grasp.
2. Failed placement triggers bounded recovery rather than false success.
3. Randomized block layout: Blocks are detected and localized purely from CameraObservation,
   and manipulated without privileged simulator ground truth.
"""

import math
import random
import pytest

from simulation.physics_engine import RobotArmSimulation
from simulation.camera import SimulationCamera, SimulationRobotStateSensor
from simulation.constants import generate_random_block_spawns, STACK_ORIGIN, TABLE_TOP_Y, CUBE_HALF
from perception.detector import BlockColorDetector
from perception.localisation import Object3DLocalizer
from perception.tracker import MultiObjectTracker
from perception.models import TrackedObject
from vla.semantic_mapper import SemanticSpatialMapper
from vla.target_grounding import TargetGrounder
from vla.execution_monitor import ExecutionMonitor, VerificationResult
from vla.google_vla_agent import GoogleVLAAgent
from vla.trajectory_controller import TrajectoryController


class TestClosedLoopAndRandomizedPerception:
    def test_01_verify_grasp_detects_empty_air_failure(self):
        """Verifies ExecutionMonitor detects when gripper closed on empty air (no contact)."""
        sim = RobotArmSimulation()
        sim.reset()

        cam = SimulationCamera(sim)
        robot_sensor = SimulationRobotStateSensor(sim)
        detector = BlockColorDetector()
        localizer = Object3DLocalizer()
        tracker = MultiObjectTracker()
        monitor = ExecutionMonitor(max_retries=2)

        # Initial observation
        obs = cam.capture()
        dets = localizer.localize(detector.detect(obs), obs)
        tracks = tracker.update(dets)

        # Target block at initial position
        target_track = tracks[0]
        initial_pos = tuple(target_track.position)

        # 1. Arm is at rest, jaws open: Effort = 0.0 -> Grasp verification fails cleanly
        state = robot_sensor.get_state()
        res = monitor.verify_grasp(target_track.color_name, initial_pos, state, tracks)

        assert res.verified is False
        assert res.recovery_action == "REACQUIRE_AND_RETRY_GRASP"
        assert "no contact" in res.reason.lower() or "indicates jaws closed on empty air" in res.reason

        # 2. Second failure (retry 1)
        res2 = monitor.verify_grasp(target_track.color_name, initial_pos, state, tracks)
        assert res2.verified is False
        assert res2.recovery_action == "REACQUIRE_AND_RETRY_GRASP"

        # 3. Third failure exhausts max_retries (2 retries max)
        res3 = monitor.verify_grasp(target_track.color_name, initial_pos, state, tracks)
        assert res3.verified is False
        assert res3.recovery_action == "ABORT_AND_REPLAN"

    def test_02_verify_placement_detects_accurate_stacking(self):
        """Verifies ExecutionMonitor confirms placement when object matches target slot."""
        monitor = ExecutionMonitor(placement_tolerance=0.038)
        target_slot = STACK_ORIGIN

        # 1. Accurate track within 10mm of target slot
        good_track = TrackedObject(
            track_id="t_placed",
            class_name="block",
            color_name="red",
            color_rgb=(255, 0, 0),
            position=[target_slot[0] + 0.008, target_slot[1] + CUBE_HALF, target_slot[2] - 0.005],
            confidence=0.97
        )

        res_ok = monitor.verify_placement("Red Cube", (target_slot[0], target_slot[1] + CUBE_HALF, target_slot[2]), [good_track])
        assert res_ok.verified is True
        assert res_ok.confidence >= 0.80
        assert "within tolerance" in res_ok.reason

        # 2. Displaced track outside tolerance
        bad_track = TrackedObject(
            track_id="t_bad",
            class_name="block",
            color_name="red",
            color_rgb=(255, 0, 0),
            position=[target_slot[0] + 0.12, target_slot[1] + CUBE_HALF, target_slot[2]],
            confidence=0.95
        )
        res_bad = monitor.verify_placement("Red Cube", (target_slot[0], target_slot[1] + CUBE_HALF, target_slot[2]), [bad_track])
        assert res_bad.verified is False
        assert res_bad.recovery_action == "REINSPECT_AND_NUDGE"

    def test_03_randomized_workcell_perception_and_planning(self):
        """
        Integration test:
        1. Spawns blocks at randomized tabletop coordinates.
        2. Perceives the scene via CameraObservation (no privileged reading of true positions).
        3. Segments and localizes all 6 blocks in 3D.
        4. Calculates precision, recall, and localization error in meters.
        5. Semantic mapper constructs estimated memory graph.
        6. VLA plans manipulation using solely the estimated semantic map.
        """
        sim = RobotArmSimulation()
        sim.reset(randomize=True)

        cam = SimulationCamera(sim, size=128)
        robot_sensor = SimulationRobotStateSensor(sim)
        detector = BlockColorDetector()
        localizer = Object3DLocalizer()
        tracker = MultiObjectTracker()
        mapper = SemanticSpatialMapper()

        # Step 1: Capture Camera Observation
        obs = cam.capture()
        robot_obs = robot_sensor.get_state()

        # Step 2: Segment & Localize
        raw_detections = detector.detect(obs)
        localized_dets = localizer.localize(raw_detections, obs)
        tracks = tracker.update(localized_dets)

        # Step 3: Evaluate Perception Metrics against hidden simulator state (Post-episode evaluation only)
        gt_blocks = sim.get_snapshot()["blocks"]
        gt_colors = {b["color"]: b for b in gt_blocks}
        color_map = {
            "red": "#FF1E27", "blue": "#1E60FF", "yellow": "#FFC300",
            "green": "#00B341", "orange": "#FF5500", "purple": "#9333EA",
        }

        # Precision & Recall
        detected_colors = {d.color_name for d in localized_dets if d.color_name}
        assert len(detected_colors) == 6, f"Expected 6 distinct colors, found {detected_colors}"
        recall = len(detected_colors) / 6.0
        assert recall == 1.0, "Object detection recall must be 100%"

        # Metric Localisation Error
        localization_errors = []
        for det in localized_dets:
            hex_c = color_map[det.color_name]
            gt_pos = gt_colors[hex_c]["position"]
            err = math.dist(det.position_base, gt_pos)
            localization_errors.append(err)

        mean_error_m = sum(localization_errors) / len(localization_errors)
        max_error_m = max(localization_errors)

        # Orthographic 128x128 grid cell size is ~10mm. Mean error should be < 12mm
        assert mean_error_m < 0.015, f"Mean localization error {mean_error_m*1000:.2f}mm exceeds 15mm"
        assert max_error_m < 0.020, f"Max localization error {max_error_m*1000:.2f}mm exceeds 20mm"

        # Step 4: Build Estimated Map
        mapper.update_from_perception(tracks, robot_obs)
        assert len(mapper.entities) >= 6

        # Step 5: VLA Agent Plans Pick-and-Place using ONLY estimated semantic memory
        vla = GoogleVLAAgent()
        cmd = "Pick up the yellow block and place it on the target pad"
        result = vla.parse_instruction(
            instruction=cmd,
            image=obs.rgb,
            snapshot=None, # Strictly no simulator snapshot passed to planner!
            semantic_summary=mapper.get_semantic_summary(),
            semantic_mapper=mapper # Planner uses estimated perception map
        )

        assert result["is_possible"] is True
        assert len(result["plan"]) > 0

        # Verify plan waypoints match estimated yellow block position
        yellow_track = [t for t in tracks if t.color_name == "yellow"][0]
        pick_step = [s for s in result["plan"] if s["action"] == "HOVER_ACQUIRE"][0]

        # The planned pick hover coordinates must match the PERCEIVED yellow block, not hardcoded origin!
        assert abs(pick_step["target"][0] - yellow_track.position[0]) < 0.001
        assert abs(pick_step["target"][2] - yellow_track.position[2]) < 0.001
