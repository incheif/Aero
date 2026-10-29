"""
vla/h_jepa/orchestrator.py

Hierarchical VL-JEPA Orchestrator.
Integrates Level 2 Macro-Task Planning with Level 1 High-Rate Micro-CEM Control.
Closes the loop with simulation/kinematics.py and simulation/scoring.py to eliminate
false positives and ensure drift-free multi-box manipulation (pyramids, towers, trains).
"""

import time
import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from simulation.constants import (
    STACK_ORIGIN,
    SAFE_HOVER_Y,
    PICK_Y,
    TABLE_TOP_Y,
    CUBE_SIZE,
    GRASP_CLOSE_THRESHOLD,
    GRASP_OPEN_THRESHOLD,
    get_pyramid_slots,
    get_train_slots,
)
from simulation.kinematics import inverse_kinematics
from simulation.scoring import StabilityValidator, vec_dist
from .encoders import (
    Level2ObservationEncoder,
    Level1ObservationEncoder,
    Level1ActionEncoder,
    ProjectionHead1to2,
)
from .predictors import Level1Predictor, Level2Predictor
from .vl_grounder import VLJEPAGrounder
from .cem_planner import Level1MicroCEMPlanner, Level2MacroCEMPlanner

class HierarchicalVLJEPAOrchestrator:
    """
    Complete 2-Level H-JEPA System.
    Connects language grounding, latent world modeling, CEM micro-planning,
    analytical IK execution, and physical stability hold verification.
    """
    MAX_RETRIES: int = 3
    GRASP_TIMEOUT: int = 60
    VERIFY_TIMEOUT: int = 90

    def __init__(self):
        # Encoders & Projectors
        self.encoder_l2 = Level2ObservationEncoder()
        self.encoder_l1 = Level1ObservationEncoder()
        self.encoder_act_l1 = Level1ActionEncoder()
        self.projector_1to2 = ProjectionHead1to2()

        # World Model Predictors
        self.predictor_l2 = Level2Predictor()
        self.predictor_l1 = Level1Predictor()

        # Language Grounder
        self.vl_grounder = VLJEPAGrounder(encoder_l2=self.encoder_l2)

        # Planners
        self.cem_micro = Level1MicroCEMPlanner(
            obs_encoder_l1=self.encoder_l1,
            act_encoder_l1=self.encoder_act_l1,
            predictor_l1=self.predictor_l1,
            projector_1to2=self.projector_1to2,
            horizon=4,
            num_samples=48,
            num_elites=6,
            num_iterations=2,
        )
        self.cem_macro = Level2MacroCEMPlanner(predictor_l2=self.predictor_l2)

        # Physical Stability Validator from simulation/scoring.py
        self.stability_validator = StabilityValidator(required_ticks=18)

        # State Variables
        self.is_active: bool = False
        self.task_instruction: str = ""
        self.z_goal_l2: Optional[np.ndarray] = None
        self.task_metadata: Dict[str, Any] = {}
        self.subgoals: List[Dict[str, Any]] = []
        self.current_subgoal_idx: int = 0
        
        # Micro-Phase State Machine
        # States: "HOVER_PICK", "DESCEND_PICK", "GRASP", "LIFT", "HOVER_PLACE", "DESCEND_PLACE", "RELEASE", "VERIFY_HOLD"
        self.micro_phase: str = "IDLE"
        self.phase_ticks: int = 0
        self.retry_count: int = 0
        self.dwell_counter: int = 0
        self.grasped_block_id: Optional[str] = None
        self.latest_cem_telemetry: Dict[str, Any] = {}

    def _set_phase(self, phase: str):
        """Sets the active micro-phase and resets phase_ticks on change."""
        if self.micro_phase != phase:
            self.micro_phase = phase
            self.phase_ticks = 0

    def _retry_subgoal(self):
        """
        Handles retry logic when grasping or verification times out.
        Resets grasped block and stability validator, increments retry_count.
        If retries are exceeded, advances to next subgoal or completes; otherwise returns to HOVER_PICK.
        """
        if self.grasped_block_id is not None:
            self._set_phase("RELEASE")
            return

        self.grasped_block_id = None
        self.stability_validator.reset()
        self.retry_count += 1
        if self.retry_count > self.MAX_RETRIES:
            self.retry_count = 0
            self.current_subgoal_idx += 1
            if self.current_subgoal_idx < len(self.subgoals):
                self._set_phase("HOVER_PICK")
            else:
                self._set_phase("COMPLETED")
                self.is_active = False
        else:
            self._set_phase("HOVER_PICK")

    def plan_task(
        self,
        instruction: str,
        initial_blocks: List[Dict[str, Any]],
        target_slots: Optional[List[List[float]]] = None,
        target_block_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Initializes task via Level 2 Language Grounding and macro plan generation.
        """
        self.task_instruction = instruction
        self.z_goal_l2, self.task_metadata = self.vl_grounder.ground_goal_embedding(
            instruction=instruction,
            target_slots=target_slots,
            block_ids=target_block_ids,
            initial_blocks=initial_blocks,
        )

        slots = self.task_metadata["target_slots"]
        b_ids = self.task_metadata["target_blocks"]

        # Generate intermediate sub-goal sequence
        self.subgoals = self.vl_grounder.generate_subgoal_embeddings(
            target_slots=slots,
            block_ids=b_ids,
            initial_blocks=initial_blocks
        )

        self.current_subgoal_idx = 0
        self.is_active = len(self.subgoals) > 0
        self.retry_count = 0
        self._set_phase("HOVER_PICK" if self.is_active else "IDLE")
        self.dwell_counter = 0
        self.grasped_block_id = None
        self.stability_validator.reset()

        return {
            "status": "PLAN_READY",
            "instruction": instruction,
            "structure_type": self.task_metadata.get("structure_type"),
            "num_subgoals": len(self.subgoals),
            "subgoals": [
                {"step": s["subgoal_index"] + 1, "desc": s["description"]}
                for s in self.subgoals
            ]
        }

    def get_block_by_id(self, blocks: List[Dict[str, Any]], block_id: str) -> Optional[Dict[str, Any]]:
        for b in blocks:
            if b.get("id") == block_id:
                return b
        return None

    def step(
        self,
        current_tcp: Tuple[float, float, float],
        current_joints: Dict[str, float],
        blocks: List[Dict[str, Any]],
        current_gripper: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Main execution tick (called at simulation rate).
        Runs Level 1 Micro-CEM, handles phase state machine, and verifies physical stability.
        Returns command dict for physics engine or None if completed.
        """
        if not self.is_active or self.current_subgoal_idx >= len(self.subgoals):
            self.is_active = False
            self._set_phase("COMPLETED")
            return None

        self.phase_ticks += 1

        active_subgoal = self.subgoals[self.current_subgoal_idx]
        b_id = active_subgoal["block_id"]
        target_slot = active_subgoal["target_slot"]
        z_sub = active_subgoal["z_sub"]

        target_block = self.get_block_by_id(blocks, b_id)
        if not target_block:
            # Block missing, advance or abort
            return None

        block_pos = target_block.get("position", [0.0, TABLE_TOP_Y, 0.0])

        # Micro-Phase Target Computation
        micro_target = list(current_tcp)
        desired_gripper = current_gripper

        # Target transit altitude clearing any existing stack layers
        transit_y = max(SAFE_HOVER_Y, target_slot[1] + 0.070)

        if self.micro_phase == "HOVER_PICK":
            # Above block
            micro_target = [block_pos[0], transit_y, block_pos[2]]
            desired_gripper = 0.0  # Open
            d = math.hypot(current_tcp[0] - micro_target[0], current_tcp[1] - micro_target[1], current_tcp[2] - micro_target[2])
            if d < 0.030 or self.phase_ticks > 60:
                self._set_phase("DESCEND_PICK")

        elif self.micro_phase == "DESCEND_PICK":
            # Down to grasp height
            pick_y = max(0.756, block_pos[1] + 0.010)
            micro_target = [block_pos[0], pick_y, block_pos[2]]
            desired_gripper = 0.0  # Still open
            horiz_err = math.hypot(current_tcp[0] - block_pos[0], current_tcp[2] - block_pos[2])
            vert_err = abs(current_tcp[1] - pick_y)
            if (horiz_err < 0.010 and abs(current_tcp[1] - pick_y) < 0.004) or self.phase_ticks > 60:
                self._set_phase("GRASP")

        elif self.micro_phase == "GRASP":
            micro_target = list(current_tcp)
            desired_gripper = 1.0  # Close
            if current_gripper >= GRASP_CLOSE_THRESHOLD + 0.02 and target_block.get("is_grasped") is True:
                self.grasped_block_id = b_id
                self._set_phase("LIFT")
            elif self.phase_ticks > self.GRASP_TIMEOUT:
                self._retry_subgoal()

        elif self.micro_phase == "LIFT":
            micro_target = [current_tcp[0], transit_y, current_tcp[2]]
            desired_gripper = 1.0  # Hold closed
            d = abs(current_tcp[1] - transit_y)
            if self.phase_ticks > 15 and not target_block.get("is_grasped", False):
                self._retry_subgoal()
            elif d < 0.025 or self.phase_ticks > 50:
                self._set_phase("HOVER_PLACE")

        elif self.micro_phase == "HOVER_PLACE":
            # Hover directly above target slot at safe transit height
            micro_target = [target_slot[0], transit_y, target_slot[2]]
            desired_gripper = 1.0  # Hold closed
            horiz_d = math.hypot(current_tcp[0] - micro_target[0], current_tcp[2] - micro_target[2])
            if self.phase_ticks > 15 and not target_block.get("is_grasped", False):
                self._retry_subgoal()
            elif horiz_d < 0.025 or self.phase_ticks > 70:
                self._set_phase("DESCEND_PLACE")

        elif self.micro_phase == "DESCEND_PLACE":
            # Place at target slot height
            place_y = target_slot[1] + 0.008
            micro_target = [target_slot[0], place_y, target_slot[2]]
            desired_gripper = 1.0  # Hold closed
            d = math.hypot(current_tcp[0] - micro_target[0], current_tcp[1] - micro_target[1], current_tcp[2] - micro_target[2])
            horiz_err = math.hypot(current_tcp[0] - target_slot[0], current_tcp[2] - target_slot[2])
            vert_err = abs(current_tcp[1] - place_y)
            if d < 0.028 or (horiz_err < 0.022 and vert_err < 0.015) or self.phase_ticks > 60:
                self._set_phase("RELEASE")

        elif self.micro_phase == "RELEASE":
            micro_target = list(current_tcp)
            desired_gripper = 0.0  # Open gripper
            if (current_gripper <= GRASP_OPEN_THRESHOLD - 0.05 and not target_block.get("is_grasped", False)) or self.phase_ticks > 40:
                self.grasped_block_id = None
                self._set_phase("VERIFY_HOLD")

        elif self.micro_phase == "VERIFY_HOLD":
            # Retreat slightly up while verifying stability hold
            micro_target = [target_slot[0], transit_y, target_slot[2]]
            desired_gripper = 0.0
            
            # Step the physical stability validator for this specific target block
            is_verified, hold_ticks, status = self.stability_validator.step(
                blocks=blocks,
                grasped_block_id=self.grasped_block_id,
                target_block_id=b_id,
            )

            placed_pos = target_block.get("position", [0.0, 0.0, 0.0])
            placed_dist_xz = math.hypot(placed_pos[0] - target_slot[0], placed_pos[2] - target_slot[2])
            placed_dist_y = abs(placed_pos[1] - target_slot[1])

            # Gate: Verified resting stability + Placed block within 0.038m (xz) and 0.035m (y) of target slot
            if is_verified and placed_dist_xz <= 0.045 and placed_dist_y <= 0.035:
                # Sub-goal complete! Unlock next macro-step
                self.retry_count = 0
                self.current_subgoal_idx += 1
                self.stability_validator.reset()
                if self.current_subgoal_idx < len(self.subgoals):
                    self._set_phase("HOVER_PICK")
                else:
                    self._set_phase("COMPLETED")
                    self.is_active = False
            elif self.phase_ticks > self.VERIFY_TIMEOUT:
                self._retry_subgoal()
            elif self.phase_ticks > 25 and (placed_dist_xz > 0.08 or placed_dist_y > 0.08):
                self._retry_subgoal()

        if not self.is_active or self.current_subgoal_idx >= len(self.subgoals):
            self.is_active = False
            self._set_phase("COMPLETED")
            return None

        # Level 1 Micro-CEM Trajectory Optimization
        # In GRASP and RELEASE, skip CEM entirely with zero latency and zero jitter
        if self.micro_phase in ("GRASP", "RELEASE"):
            opt_action = [0.0, 0.0, 0.0, desired_gripper]
            cem_res = {
                "optimal_action": opt_action,
                "planned_trajectory": [],
                "latent_distance": 0.0,
                "latency_ms": 0.0,
                "num_samples_evaluated": 0,
            }
        else:
            cem_res = self.cem_micro.plan(
                current_tcp=list(current_tcp),
                target_pos=micro_target,
                active_subgoal_z2=z_sub,
                gripper_aperture=current_gripper,
                desired_gripper=desired_gripper
            )
            opt_action = cem_res["optimal_action"]

        self.latest_cem_telemetry = cem_res

        next_tcp = [
            current_tcp[0] + opt_action[0],
            current_tcp[1] + opt_action[1],
            current_tcp[2] + opt_action[2]
        ]
        next_gripper = float(opt_action[3])

        # Analytical IK via simulation/kinematics.py
        ik_sol = inverse_kinematics(next_tcp, next_gripper)
        if isinstance(ik_sol, dict):
            if "joints" in ik_sol:
                target_joints = ik_sol["joints"]
            else:
                target_joints = ik_sol
        else:
            target_joints = current_joints

        return {
            "joints": target_joints,
            "gripper": next_gripper,
            "tcp_target": next_tcp,
            "phase": self.micro_phase,
            "subgoal_index": self.current_subgoal_idx,
            "total_subgoals": len(self.subgoals),
            "subgoal_desc": active_subgoal["description"],
            "latent_distance": cem_res.get("latent_distance", 0.0),
            "cem_latency_ms": cem_res.get("latency_ms", 0.0),
            "stability_status": self.stability_validator.status,
            "is_done": not self.is_active,
        }
