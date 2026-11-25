"""
FastAPI Server for Robot Arm VLA Simulation & Three.js Studio.
Streams real-time 60 FPS physics telemetry over WebSockets and handles
Google VLA natural-language commands, 3D semantic mapping, and frontier exploration.

Supports multi-tenant Session ID isolation: each browser tab / visitor gets their own
private robot arm, physics environment, execution logs, and telemetry stream.
"""

import os
import sys
import io
import asyncio
import time
from typing import Dict, Any, Optional, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from simulation.constants import (
    FIXED_DT,
    PHYS_HZ,
    STACK_ORIGIN,
    SAFE_HOVER_Y,
    DEFAULT_JOINTS,
    stack_slot_y,
)
from simulation.physics_engine import RobotArmSimulation
from simulation.camera import (
    render_camera_frame,
    render_depth_frame,
    render_depth_visual,
    render_wrist_camera_frame,
    render_occupancy_map_visual,
    encode_image_base64,
    pixel_to_world,
    world_to_pixel,
    SimulationCamera,
    SimulationRobotStateSensor,
)
from simulation.kinematics import inverse_kinematics
from perception.detector import BlockColorDetector
from perception.localisation import Object3DLocalizer
from perception.tracker import MultiObjectTracker
from vla.semantic_mapper import SemanticSpatialMapper
from vla.frontier_explorer import TabletopFrontierExplorer
from vla.target_grounding import TargetGrounder
from vla.execution_monitor import ExecutionMonitor
from vla.google_vla_agent import GoogleVLAAgent
from vla.trajectory_controller import TrajectoryController
from vla.color_seek_policy import ColorSeekPolicy, find_blobs
from vla.closed_loop_recovery import ClosedLoopRecoveryOrchestrator
from vla.agents import (
    EpisodicMemoryAgent,
    NarrativeLoggerAgent,
    InstructionDecomposerAgent,
    HumanDialogueSupervisorAgent,
    FailureDetectorAgent,
)

VLA_STRIDE = 12  # 5 Hz policy updates over 60 Hz physics (vsarena standard)
SESSION_MAX_DURATION = 25 * 60  # 25 minutes (1500 seconds) auto-suspension timeout

class CommandRequest(BaseModel):
    command: str
    api_key: Optional[str] = None
    model: Optional[str] = "gemini-2.5-flash"
    session_id: Optional[str] = None

class PauseRequest(BaseModel):
    reason: Optional[str] = "Operator requested pause"
    session_id: Optional[str] = None

class GuidanceRequest(BaseModel):
    guidance: str
    session_id: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class TestKeyRequest(BaseModel):
    api_key: str
    model: Optional[str] = "gemini-2.5-flash"
    session_id: Optional[str] = None

class TeleopRequest(BaseModel):
    joints: Optional[Dict[str, float]] = None
    gripper: Optional[float] = None
    session_id: Optional[str] = None


class UserSession:
    """Encapsulates an isolated simulation and cognitive pipeline for one client session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.sim = RobotArmSimulation()
        self.camera = SimulationCamera(self.sim)
        self.robot_sensor = SimulationRobotStateSensor(self.sim)
        self.detector = BlockColorDetector()
        self.localizer = Object3DLocalizer()
        self.tracker = MultiObjectTracker()
        self.mapper = SemanticSpatialMapper()
        self.target_grounder = TargetGrounder(self.mapper)
        self.execution_monitor = ExecutionMonitor()
        self.explorer = TabletopFrontierExplorer()
        self.vla_agent = GoogleVLAAgent()
        self.controller = TrajectoryController()
        self.color_seek = ColorSeekPolicy()
        self.orchestrator = ClosedLoopRecoveryOrchestrator()
        self.prev_subgoal_id: Optional[str] = None

        # Multi-Agent Collaborative Framework
        self.memory_agent = EpisodicMemoryAgent()
        self.narrator_agent = NarrativeLoggerAgent()
        self.decomposer_agent = InstructionDecomposerAgent()
        self.failure_detector = FailureDetectorAgent()
        self.supervisor = HumanDialogueSupervisorAgent(
            memory_agent=self.memory_agent,
            narrator_agent=self.narrator_agent,
            decomposer_agent=self.decomposer_agent,
        )

        self.is_paused: bool = False
        self.manual_mode: bool = False
        self.color_seek_active: bool = False
        self.active_target_block_id: Optional[str] = None
        self.detected_blobs: Dict[str, Any] = {}

        self.websockets: Set[WebSocket] = set()
        self.recent_logs: list[Dict[str, Any]] = []
        self.pending_logs: list[Dict[str, Any]] = []
        self.created_at: float = time.time()
        self.first_connected_at: Optional[float] = None
        self.session_start_time: float = time.time()
        self.last_active: float = time.time()
        self.is_suspended: bool = False
        self.suspension_reason: Optional[str] = None

        self.prev_step_idx: int = -1
        self.prev_stability_verified: bool = False
        self.frame_count: int = 0
        self.fps_timer: float = time.time()
        self.current_fps: float = 60.0

        # H-JEPA Level 1 & 2 Live Predictive Telemetry
        self.latest_hjepa_residual: float = 0.0
        self.hjepa_anomaly_count: int = 0
        self.active_subgoal_desc: str = "Idle"
        self.total_subgoals_count: int = 0

        self.log_event(
            "system",
            "STARTUP",
            f"Session {session_id} initialized with private workcell.",
            "Rapier3D physics running at 60 FPS (dt=16.6ms). 6 blocks spawned at canonical table coordinates."
        )

    def log_event(self, category: str, badge: str, what: str, how: str) -> Dict[str, Any]:
        """Records an execution or cognitive event for this session."""
        entry = {
            "id": f"log_{int(time.time()*1000)}_{len(self.recent_logs)}",
            "timestamp": time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 100):02d}",
            "category": category,
            "badge": badge,
            "what": what,
            "how": how,
        }
        self.recent_logs.append(entry)
        if len(self.recent_logs) > 250:
            self.recent_logs.pop(0)
        self.pending_logs.append(entry)
        if len(self.pending_logs) > 50:
            self.pending_logs.pop(0)
        return entry

    async def suspend(self, reason: str = "Session auto-suspended after 25 minutes to conserve cloud bandwidth."):
        """Suspends the session to halt continuous physics streaming and bandwidth consumption."""
        if self.is_suspended:
            return
        self.is_suspended = True
        self.suspension_reason = reason
        self.controller.stop()
        self.color_seek_active = False
        self.explorer.is_exploring = False
        self.is_paused = True

        self.log_event(
            "system",
            "SUSPENDED",
            "Session auto-suspended (25 min limit reached).",
            "Telemetry streaming halted to conserve outbound cloud bandwidth."
        )

        notification = {
            "type": "session_suspended",
            "is_suspended": True,
            "session_id": self.session_id,
            "reason": reason,
            "duration_minutes": 25,
        }
        if self.pending_logs:
            notification["logs"] = list(self.pending_logs)
            self.pending_logs.clear()

        for ws in list(self.websockets):
            try:
                await ws.send_json(notification)
            except Exception:
                self.websockets.discard(ws)

    async def close_connections(self, reason: str = "Maximum of 5 sessions reached. Earliest connection closed."):
        """Closes all active WebSockets and halts running tasks for this session."""
        for ws in list(self.websockets):
            try:
                await ws.send_json({
                    "type": "session_evicted",
                    "reason": reason,
                    "session_id": self.session_id,
                })
                await ws.close(code=4001, reason="Session limit exceeded")
            except Exception:
                pass
            finally:
                self.websockets.discard(ws)
        self.controller.stop()
        self.orchestrator.stop_task()
        self.color_seek_active = False
        self.explorer.is_exploring = False

    def close_connections_sync(self, reason: str = "Maximum of 5 sessions reached. Earliest connection closed."):
        """Synchronously schedules WebSocket closure in active event loop, if running."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.close_connections(reason))
        except RuntimeError:
            self.controller.stop()
            self.orchestrator.stop_task()
            self.color_seek_active = False
            self.explorer.is_exploring = False
            self.websockets.clear()

    def resume(self):
        """Resumes a suspended session and resets the 25-minute timer."""
        was_suspended = self.is_suspended
        self.is_suspended = False
        self.suspension_reason = None
        self.session_start_time = time.time()
        self.last_active = time.time()
        self.is_paused = False
        if was_suspended:
            self.log_event(
                "system",
                "RESUMED",
                "Session resumed by user interaction.",
                "Telemetry streaming re-engaged (new 25-minute window)."
            )


    def is_active(self) -> bool:
        return (
            len(self.websockets) > 0
            or self.controller.is_active
            or self.orchestrator.is_task_active
            or self.color_seek_active
            or self.explorer.is_exploring
        )

    def handle_reset(self, randomize: bool = False, reset_blocks: bool = True):
        if self.is_suspended:
            self.resume()
        self.last_active = time.time()
        self.session_start_time = time.time()
        self.color_seek_active = False
        self.color_seek.reset()
        self.controller.stop()
        self.orchestrator.stop_task()
        self.active_target_block_id = None
        self.detected_blobs = {}
        self.explorer.is_exploring = False
        self.explorer.exploration_pct = 25.0
        for k in self.explorer.explored_cells:
            self.explorer.explored_cells[k] = 0.25
        if reset_blocks:
            self.sim.reset(randomize=randomize)
        obs = self.camera.capture()
        robot_obs = self.robot_sensor.get_state()
        dets = self.localizer.localize(self.detector.detect(obs), obs)
        tracks = self.tracker.update(dets)
        self.mapper.update_from_perception(tracks, robot_obs)
        self.vla_agent.execution_status = "IDLE"
        self.vla_agent.latest_thought = "Arm halted and returned to baseline pose. Ready for new command."
        self.supervisor.reset()
        self.narrator_agent.flush_pending()
        self.failure_detector.reset()
        if reset_blocks:
            self.log_event(
                "system",
                "RESET",
                "Workspace and arm reset." if not randomize else "Workspace randomized with collision-free spawns.",
                "Blocks spawned at baseline coordinates. Trajectory queue stopped." if not randomize else "Blocks randomized within 5-DOF reach envelope. Camera perception initialized."
            )

    def handle_failure_detected(self, failure: Dict[str, Any]):
        """
        Failure Detector Agent Action:
        Immediately stops execution, records diagnostics in logs,
        and posts an explainable failure alert to the bottom conversational chat window.
        """
        # 1. Immediately halt all active controllers and policies
        self.controller.stop()
        self.orchestrator.stop_task()
        self.color_seek_active = False
        self.active_target_block_id = None
        self.failure_detector.stop_monitoring()

        f_type = failure.get("type", "TASK_FAILURE")
        f_reason = failure.get("reason", "Robot failed to execute task")
        f_expl = failure.get("explanation", "Failure detected. Execution halted.")

        # 2. Update states
        self.vla_agent.execution_status = "FAILED"
        self.vla_agent.latest_thought = f"Failure Detector: {f_reason}"
        self.supervisor.state = "FAILED"

        # 3. Log event into Execution Story with high-visibility FAILURE badge
        self.log_event(
            "vla",
            "FAILURE",
            f"Execution Halted: {f_type} - {f_reason}",
            f"{f_expl} Autonomous safety stop triggered to protect robot hardware and workcell."
        )
        self.narrator_agent.log(
            category="anomaly",
            title=f"Failure: {f_type}",
            explanation=f_expl,
            severity="error",
        )

        # 4. Post prominently into bottom chat window
        self.supervisor.post_robot_message(
            f"Execution stopped due to failure: {f_reason}\n\nDiagnostics: {f_expl}\n\nPlease click Reset Arm or provide a corrective instruction.",
            status_tag="FAILED"
        )

    async def handle_vla_command(self, cmd: str) -> Dict[str, Any]:
        if self.is_suspended:
            self.resume()
        self.manual_mode = False
        self.last_active = time.time()
        self.session_start_time = time.time()

        self.log_event(
            "vla",
            "EXECUTION STARTED",
            f"Execution started for instruction: \"{cmd}\"",
            f"Dispatched to Google VLA reasoning pipeline (Model: {self.vla_agent.model_name}). Synthesizing spatial plan and solving 5-DOF IK waypoints."
        )
        self.supervisor.state = "PLANNING"
        self.supervisor.post_user_message(cmd)
        self.supervisor.post_robot_message(f"Execution started for: \"{cmd}\". Synthesizing spatial plan...", status_tag="PLANNING")

        # Halt prior trajectory and refresh perception without resetting block positions
        self.handle_reset(reset_blocks=False)

        # Capture sensor observation through CameraInterface
        obs = self.camera.capture()
        robot_obs = self.robot_sensor.get_state()
        dets = self.localizer.localize(self.detector.detect(obs), obs)
        tracks = self.tracker.update(dets)
        self.mapper.update_from_perception(tracks, robot_obs)
        semantic_summary = self.mapper.get_semantic_summary()

        # 1. Computer Vision Perception Acquisition
        self.log_event(
            "perception",
            "CAMERA ACQUIRED",
            f"128×128 overhead RGB frame captured. Segmented {len(dets)} blocks, tracked {len(tracks)} active entities.",
            "Sensor: Orthographic top-down · Matrix: 128×128 RGB · Zero privileged ground-truth GPS cheating."
        )

        blobs = find_blobs(obs.rgb)
        detected_colors = [c for c, pt in blobs.items() if pt is not None and c not in ("tcp", "pad")]
        self.detected_blobs = {c: blobs[c] for c in detected_colors}

        # Synthesize manipulation, gesture, or feasibility validation through Google VLA using estimated perception
        result = self.vla_agent.parse_instruction(cmd, obs.rgb, None, semantic_summary, semantic_mapper=self.mapper)

        # If rejected as impossible:
        if not result.get("is_possible", True) or result.get("status") == "IMPOSSIBLE":
            self.controller.stop()
            self.active_target_block_id = None
            rej_reason = result.get('reason') or result.get('thought') or result.get('reasoning') or 'Action violates spatial reachability or semantic stability constraint.'
            self.log_event(
                "vla",
                "REJECT",
                f"Instruction rejected as infeasible: \"{cmd}\"",
                f"Reason: {rej_reason}"
            )
            self.supervisor.post_user_message(cmd)
            self.supervisor.post_robot_message(f"I cannot do this: {rej_reason}", status_tag="REJECTED")
            self.narrator_agent.log(
                category="anomaly",
                title="Command Infeasible",
                explanation=rej_reason,
                severity="error",
            )
            return result

        # 2. Segment target block & project 2D camera pixels to 3D metric coordinates
        color_map = {
            "cyan": "block_cyan",
            "blue": "block_cyan",
            "orange": "block_orange",
            "magenta": "block_magenta",
            "red": "block_magenta",
            "yellow": "block_yellow",
            "emerald": "block_emerald",
            "green": "block_emerald",
            "purple": "block_purple",
            "violet": "block_purple",
        }

        target_block = None
        if result.get("grounding") and len(result["grounding"]) > 0:
            target_block = result["grounding"][0].get("entity")
        elif result.get("stack_sequence") and len(result["stack_sequence"]) > 0:
            target_block = result["stack_sequence"][0]
        else:
            for c_name, b_id in color_map.items():
                if c_name in cmd.lower():
                    target_block = b_id
                    break

        if target_block:
            self.active_target_block_id = target_block
            matched_blob_color = None
            for c_name, b_id in color_map.items():
                if b_id == target_block and c_name in blobs and blobs[c_name]:
                    matched_blob_color = c_name
                    break

            if matched_blob_color and blobs.get(matched_blob_color):
                bu, bv = blobs[matched_blob_color]
                mx, mz = pixel_to_world(bu, bv)
                self.log_event(
                    "perception",
                    "PIXEL SEGMENTATION",
                    f"Target '{matched_blob_color}' cube segmented at pixel (u={int(round(bu))}, v={int(round(bv))}).",
                    f"HSV/RGB cluster centroid: ({bu:.1f}px, {bv:.1f}px) · Detected {len(detected_colors)} table cubes: {', '.join(detected_colors)}."
                )
                self.log_event(
                    "perception",
                    "3D PROJECTION",
                    f"Projected pixel (u={int(round(bu))}, v={int(round(bv))}) to metric coordinates [X={mx:.3f}m, Z={mz:.3f}m].",
                    f"Orthographic inverse projection matrix: X={mx:.3f}m, Z={mz:.3f}m · Altitude locked to table plane Y=0.758m."
                )
            else:
                self.log_event(
                    "perception",
                    "PIXEL SEGMENTATION",
                    f"Computer vision segmented {len(detected_colors)} cubes: {', '.join(detected_colors)}.",
                    f"Active target: {target_block} · Tracking reticle and vector lock engaged."
                )
        else:
            if detected_colors:
                self.log_event(
                    "perception",
                    "PIXEL SEGMENTATION",
                    f"Computer vision segmented {len(detected_colors)} cubes: {', '.join(detected_colors)}.",
                    f"Pixel centroids: " + "; ".join([f"{c}: [{int(round(blobs[c][0]))}, {int(round(blobs[c][1]))}]" for c in detected_colors if blobs[c]])
                )

        # Configure Task-Adaptive Scoring targets
        task_blocks = []
        task_targets = {}
        if result.get("action_type") == "COLORSEEK":
            task_blocks = ["block_cyan", "block_orange", "block_magenta"]
            for l_idx, b_id in enumerate(task_blocks):
                task_targets[b_id] = [STACK_ORIGIN[0], stack_slot_y(l_idx), STACK_ORIGIN[2]]
        elif result.get("grounding"):
            for l_idx, g in enumerate(result["grounding"]):
                b_id = g.get("entity")
                if b_id:
                    clean_id = b_id if b_id.startswith("block_") else f"block_{b_id}"
                    if clean_id in self.sim.blocks:
                        if clean_id not in task_blocks:
                            task_blocks.append(clean_id)
                        t_layer = g.get("target_layer", l_idx)
                        t_coords = g.get("target_coords") or [STACK_ORIGIN[0], stack_slot_y(t_layer), STACK_ORIGIN[2]]
                        task_targets[clean_id] = list(t_coords)
        elif result.get("stack_sequence"):
            for l_idx, b_id in enumerate(result["stack_sequence"]):
                clean_id = b_id if b_id.startswith("block_") else f"block_{b_id}"
                if clean_id in self.sim.blocks and clean_id not in task_blocks:
                    task_blocks.append(clean_id)
                    task_targets[clean_id] = [STACK_ORIGIN[0], stack_slot_y(l_idx), STACK_ORIGIN[2]]
        elif target_block:
            clean_id = target_block if target_block.startswith("block_") else f"block_{target_block}"
            if clean_id in self.sim.blocks:
                task_blocks = [clean_id]
                task_targets = {clean_id: [STACK_ORIGIN[0], stack_slot_y(0), STACK_ORIGIN[2]]}

        if task_blocks:
            self.sim.set_active_task(task_blocks, task_targets)

        # If ColorSeek closed-loop visual servoing was requested:
        if result.get("action_type") == "COLORSEEK":
            self.controller.stop()
            self.explorer.is_exploring = False
            self.color_seek.reset()
            self.color_seek_active = True
            self.vla_agent.execution_status = "EXECUTING"
            self.vla_agent.latest_thought = "ColorSeek: Closed-loop RGB visual servoing active (5 Hz policy stride)."
            self.log_event(
                "vla",
                "COLORSEEK",
                "Visual servoing stacking policy initialized",
                "RGB camera tracking active at 5 Hz control policy stride. Visual servoing closed-loop engaged."
            )
            return result

        # If active frontier exploration was requested:
        if result.get("action_type") == "EXPLORE" or (
            any(w in cmd.lower() for w in ("explore", "scan table", "survey", "frontier"))
            and not any(w in cmd.lower() for w in ("wave", "tower", "stack", "pick", "letter", "shape", "triangle", "cross", "sort"))
        ):
            self.explorer.start_exploration()
            self.controller.stop()
            self.orchestrator.stop_task()
            self.color_seek_active = False
            self.vla_agent.execution_status = "EXECUTING"
            self.vla_agent.latest_thought = f"Google VLA: Executing active frontier exploration for '{cmd}'."
            self.log_event(
                "motion",
                "EXPLORE",
                f"Frontier exploration started: \"{cmd}\"",
                "Autonomous workspace survey engaged. Updating 3D semantic voxel grid."
            )
            return {"status": "ok", "action": "EXPLORATION_STARTED", "thought": self.vla_agent.latest_thought}

        # Handle communicative gestures through direct kinematic controller
        if result.get("action_type") in ("GESTURE", "SWEEP_TRAJECTORY"):
            self.color_seek_active = False
            self.orchestrator.stop_task()
            plan = result.get("plan", [])
            if plan:
                self.controller.load_plan(plan)
                self.log_event(
                    "motion",
                    "IK TRAJECTORY",
                    f"Calculated 5-DOF IK trajectory with {len(plan)} waypoints.",
                    f"Action: {result.get('action_type', 'TASK')} · CoT reasoning: {result.get('reasoning', result.get('thought', 'Metric waypoint sequence dispatched.'))}"
                )
            return result

        # For all complex manipulation tasks: Dispatch through H-JEPA Closed-Loop Recovery Orchestrator
        self.color_seek_active = False
        self.controller.stop()
        orch_started = self.orchestrator.start_task(self.sim, cmd)
        if orch_started:
            self.total_subgoals_count = len(self.orchestrator.planner.subgoals)
            self.vla_agent.execution_status = "EXECUTING"
            self.vla_agent.latest_thought = (
                f"H-JEPA + VLA Recursive DAG: Dispatched {self.total_subgoals_count} subgoals for \"{cmd}\". "
                f"Real-time 60 Hz predictive monitoring active."
            )
            self.log_event(
                "vla",
                "H-JEPA DAG",
                f"Compiled {self.total_subgoals_count} subgoals for \"{cmd}\".",
                f"Action: {result.get('action_type', 'TASK')} · Level 1 & 2 H-JEPA energy validation engaged."
            )

            # Multi-agent breakdown and conversational proposal
            decomp = self.decomposer_agent.decompose(cmd, self.sim)
            self.supervisor.current_instruction = cmd
            self.supervisor.active_steps = decomp.get("easy_steps", [])
            self.supervisor.current_step_idx = 0
            self.supervisor.state = "EXECUTING"
            self.supervisor.post_user_message(cmd)
            proposal = decomp.get("proposal_message", f"Planned {self.total_subgoals_count} steps for \"{cmd}\". Starting now.")
            self.supervisor.post_robot_message(proposal, steps=self.supervisor.active_steps, status_tag="EXECUTING")
            self.narrator_agent.log(
                category="planning",
                title=f"Plan Initialized ({self.total_subgoals_count} Steps)",
                explanation=f"Starting autonomous execution of \"{cmd}\".",
                severity="info",
            )

            return {
                "status": "ok",
                "action_type": result.get("action_type", "TASK"),
                "subgoals_count": self.total_subgoals_count,
                "thought": self.vla_agent.latest_thought,
                "easy_steps": self.supervisor.active_steps,
            }

        # Fallback to legacy trajectory controller if orchestrator could not compile subgoals
        plan = result.get("plan", [])
        if plan:
            self.controller.load_plan(plan)
            self.log_event(
                "motion",
                "IK TRAJECTORY",
                f"Calculated 5-DOF IK trajectory with {len(plan)} waypoints.",
                f"Action: {result.get('action_type', 'TASK')} · CoT reasoning: {result.get('reasoning', result.get('thought', 'Metric waypoint sequence dispatched.'))}"
            )
            return result

        # Initial Task Failure Detection: check if initial step could not be formed
        subgoals_list = list(self.orchestrator.planner.subgoals.values()) if hasattr(self.orchestrator, "planner") else []
        init_fail = self.failure_detector.evaluate_initial_feasibility(cmd, subgoals_list, self.sim)
        if init_fail:
            self.handle_failure_detected(init_fail)
            return {"status": "FAILED", "reason": init_fail["reason"], "is_possible": False}

        return result

    def step(self) -> Optional[Dict[str, Any]]:
        if self.is_paused:
            return None

        # Failure Detector Agent: Continuous 60 Hz execution health & stall monitor
        if not self.manual_mode:
            fail_check = self.failure_detector.evaluate(
                sim=self.sim,
                controller=self.controller,
                orchestrator=self.orchestrator,
                is_color_seek=self.color_seek_active,
                current_instruction=self.supervisor.current_instruction
            )
            if fail_check and fail_check.get("failed"):
                self.handle_failure_detected(fail_check)
                return None

        fk_data = self.sim.get_snapshot()["tcp"]["position"]
        tcp_pos = (fk_data[0], fk_data[1], fk_data[2])

        # 0. H-JEPA Closed-Loop Recovery Orchestrator step if active
        if self.orchestrator.is_task_active and not self.manual_mode:
            orch_res = self.orchestrator.step_task(self.sim)
            self.latest_hjepa_residual = orch_res.get("residual", 0.0)
            if orch_res.get("is_anomaly"):
                self.hjepa_anomaly_count += 1
                self.log_event(
                    "vla",
                    "H-JEPA ANOMALY",
                    f"Latent energy anomaly spike (residual={self.latest_hjepa_residual:.2f}).",
                    "Dynamic recovery subgoal spliced into active DAG."
                )

            self.active_subgoal_desc = orch_res.get("subgoal_desc", "Active")
            cur_sg_id = orch_res.get("subgoal_id", "")
            prog = self.orchestrator.get_progress()
            self.vla_agent.execution_status = "EXECUTING"
            self.vla_agent.latest_thought = f"H-JEPA Subgoal {prog['step']}/{prog['total_steps']}: {self.active_subgoal_desc} (Latent Residual: {self.latest_hjepa_residual:.2f})"
            self.supervisor.current_step_idx = max(0, prog['step'] - 1)

            if cur_sg_id and cur_sg_id != getattr(self, "prev_subgoal_id", None):
                self.prev_subgoal_id = cur_sg_id
                ent = self.orchestrator.active_subgoal.entity if self.orchestrator.active_subgoal else None
                sk_type = self.orchestrator.active_subgoal.skill_type if self.orchestrator.active_subgoal else "STEP"
                if ent:
                    self.active_target_block_id = ent if ent.startswith("block_") else f"block_{ent}"
                    self.supervisor.active_entity = ent
                self.log_event(
                    "motion",
                    "SUBGOAL",
                    f"Executing Subgoal: {self.active_subgoal_desc}",
                    f"H-JEPA residual: {self.latest_hjepa_residual:.2f} · Target: {cur_sg_id}"
                )
                self.narrator_agent.explain_subgoal_transition(
                    subgoal_idx=prog['step'],
                    total_subgoals=prog['total_steps'],
                    skill_type=sk_type,
                    entity=ent,
                    slot_desc=self.active_subgoal_desc,
                )

            if not orch_res.get("is_active"):
                self.active_target_block_id = None
                self.active_subgoal_desc = "Idle"
                if orch_res.get("success"):
                    self.vla_agent.execution_status = "COMPLETED"
                    self.vla_agent.latest_thought = "H-JEPA: Task successfully completed and physically verified!"
                    self.supervisor.state = "COMPLETED"
                    self.supervisor.post_robot_message("All steps successfully executed and physically verified!", status_tag="COMPLETED")
                    self.narrator_agent.log(
                        category="success",
                        title="Task Completed",
                        explanation="All blocks placed and verified in target formation.",
                        severity="success",
                    )
                    self.log_event(
                        "system",
                        "COMPLETED",
                        "All subgoals completed and physically verified.",
                        f"Autonomous execution finished with {self.hjepa_anomaly_count} self-healing interventions."
                    )
                else:
                    self.vla_agent.execution_status = "FAILED"
                    self.vla_agent.latest_thought = f"H-JEPA: Task halted ({orch_res.get('message', 'Unresolved failure')})."
                    self.supervisor.state = "FAILED"
                    self.supervisor.post_robot_message(f"Task halted: {orch_res.get('message', 'Unresolved failure')}", status_tag="FAILED")

        # 1. Controller step if active plan
        elif self.controller.is_active and not self.manual_mode:
            target_cmd = self.controller.step(tcp_pos, self.sim.joints)
            if target_cmd:
                self.sim.set_joint_targets(target_cmd["joints"], target_cmd["gripper"])
                cur_idx = self.controller.step_idx
                if cur_idx != self.prev_step_idx and cur_idx < len(self.controller.current_plan):
                    self.prev_step_idx = cur_idx
                    step_data = self.controller.current_plan[cur_idx]
                    s_desc = step_data.get("desc", f"Waypoint {cur_idx + 1}")
                    s_tgt = step_data.get("target", (0, 0, 0))
                    s_grip = step_data.get("gripper", 0.0)
                    s_dwell = step_data.get("dwell_ticks", 10)
                    # Dynamically update active target block for camera HUD
                    for c_name, b_id in [
                        ("cyan", "block_cyan"), ("orange", "block_orange"), ("magenta", "block_magenta"),
                        ("yellow", "block_yellow"), ("emerald", "block_emerald"), ("green", "block_emerald"),
                        ("purple", "block_purple"), ("violet", "block_purple")
                    ]:
                        if c_name in s_desc.lower() or b_id in s_desc.lower():
                            self.active_target_block_id = b_id
                            break
                    is_contact = "GRIP" in s_desc.upper() or "RELEASE" in s_desc.upper()
                    s_cat = "contact" if is_contact else "motion"
                    s_badge = "GRIP" if "GRIP" in s_desc.upper() else ("RELEASE" if "RELEASE" in s_desc.upper() else "IK MOTION")
                    self.log_event(
                        s_cat,
                        s_badge,
                        f"Waypoint {cur_idx + 1}/{len(self.controller.current_plan)}: {s_desc}",
                        f"Target TCP: [{s_tgt[0]:.3f}, {s_tgt[1]:.3f}, {s_tgt[2]:.3f}] · Jaws: {'Closed (100%)' if s_grip > 0.5 else 'Open (100%)'} · Dwell: {s_dwell} ticks contact settle"
                    )

                if target_cmd.get("is_active") is False:
                    self.active_target_block_id = None
                    self.vla_agent.execution_status = "COMPLETED"
                    self.vla_agent.latest_thought = f"Completed action plan: {target_cmd.get('step_desc')}"
                    self.log_event(
                        "system",
                        "COMPLETED",
                        f"Action completed: {target_cmd.get('step_desc')}",
                        "All waypoints successfully executed to metric tolerances. Arm holding in standby pose."
                    )

        # 1b. ColorSeek closed-loop visual servoing step (5 Hz downsampled VLA stride)
        elif self.color_seek_active and not self.manual_mode:
            if self.sim.tick % VLA_STRIDE == 0:
                cam_frame = render_camera_frame(self.sim.get_snapshot())
                cs_cmd = self.color_seek.step(cam_frame, self.sim.joints, tcp_pos)
                if self.color_seek.target_idx < len(self.color_seek.order):
                    self.active_target_block_id = f"block_{self.color_seek.order[self.color_seek.target_idx]}"
                self.sim.set_joint_targets(cs_cmd["joints"], cs_cmd["gripper"])
                self.vla_agent.latest_thought = f"ColorSeek (5 Hz VLA): {cs_cmd.get('plan')}"
                if not cs_cmd.get("is_active", True):
                    self.active_target_block_id = None
                    self.color_seek_active = False
                    self.vla_agent.execution_status = "COMPLETED"
                    self.vla_agent.latest_thought = "ColorSeek: Closed-loop visual stacking sequence completed!"

        # 2. Frontier exploration step if active
        elif self.explorer.is_exploring and not self.manual_mode:
            wp = self.explorer.get_current_waypoint()
            target_pos = wp["target"]
            dx = target_pos[0] - tcp_pos[0]
            dy = target_pos[1] - tcp_pos[1]
            dz = target_pos[2] - tcp_pos[2]
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5

            ik_targets = inverse_kinematics(target_pos, gripper=0.0)
            self.sim.set_joint_targets(ik_targets, gripper=0.0)

            if dist <= 0.02:
                self.explorer.advance_waypoint()
                self.mapper.total_scanned_area_pct = self.explorer.exploration_pct
                if not self.explorer.is_exploring:
                    self.vla_agent.latest_thought = "Exploration complete. 3D Semantic Spatial Memory fully mapped."
                    self.vla_agent.execution_status = "IDLE"

        # 3. Step physics simulation
        snapshot = self.sim.step(FIXED_DT)

        # 4. Periodic Perception & Semantic Spatial Memory Update (12 Hz)
        if self.sim.tick % 5 == 0:
            obs = self.camera.capture()
            robot_obs = self.robot_sensor.get_state()
            dets = self.localizer.localize(self.detector.detect(obs), obs)
            tracks = self.tracker.update(dets)
            self.mapper.update_from_perception(tracks, robot_obs)

        # FPS calculations
        self.frame_count += 1
        if time.time() - self.fps_timer >= 1.0:
            self.current_fps = self.frame_count / (time.time() - self.fps_timer)
            self.fps_timer = time.time()
            self.frame_count = 0

        # Stability state change logging
        stab = snapshot.get("scores", {}).get("stability", {})
        is_ver = stab.get("is_verified", False)
        if is_ver and not self.prev_stability_verified:
            self.prev_stability_verified = True
            spat_acc = snapshot.get("scores", {}).get("spatial_accuracy", 0.0)
            self.log_event(
                "stability",
                "VERIFIED",
                "Post-release physical stability hold verified (18/18 ticks = 0.30s).",
                f"Target stack held without toppling. Linear velocity <= 0.02 m/s, angular velocity <= 0.05 rad/s. Spatial accuracy score: {spat_acc*100:.1f}%."
            )
        elif not is_ver:
            self.prev_stability_verified = False

        return snapshot

    def get_telemetry_payload(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "type": "telemetry",
            "session_id": self.session_id,
            "tick": snapshot["tick"],
            "fps": round(self.current_fps, 1),
            "joints": snapshot["joints"],
            "joint_targets": snapshot["joint_targets"],
            "arm": snapshot["arm"],
            "tcp": snapshot["tcp"],
            "blocks": snapshot["blocks"],
            "grasped_block_id": snapshot["grasped_block_id"],
            "actuators": snapshot.get("actuators", {}),
            "target_zone": snapshot["target_zone"],
            "scores": snapshot.get("scores", {}),
            "tcp_trail": (self.orchestrator.executor.controller.tcp_trail if self.orchestrator.is_task_active else self.controller.tcp_trail)[-60:],
            "planned_path": (self.orchestrator.executor.controller if self.orchestrator.is_task_active else self.controller).get_remaining_waypoints(snapshot["tcp"]["position"]),
            "planned_target": (self.orchestrator.executor.controller if self.orchestrator.is_task_active else self.controller).get_final_target(),
            "semantic_map": self.mapper.to_dict(),
            "perception": {
                "tracks": [t.to_dict() for t in self.tracker.tracks.values()],
                "count": len(self.tracker.tracks),
                "camera_model": "orthographic_128x128",
                "calibrated": True,
                "execution_phase": self.execution_monitor.current_phase,
                "mode": "perception_estimated",
                "target_block_id": self.active_target_block_id,
                "detected_count": len(self.detected_blobs),
                "camera_res": "128x128",
            },
            "frontier": self.explorer.to_dict(),
            "vla": {
                "model": "ColorSeek (Visual Servoing)" if self.color_seek_active else self.vla_agent.model_name,
                "status": self.vla_agent.execution_status,
                "thought": self.vla_agent.latest_thought,
                "progress": self.orchestrator.get_progress() if self.orchestrator.is_task_active else (
                    {"is_active": True, "step": self.color_seek.target_idx + 1, "total_steps": len(self.color_seek.order), "pct": int((self.color_seek.target_idx / max(1, len(self.color_seek.order))) * 100), "action": f"ColorSeek: {self.color_seek.order[min(self.color_seek.target_idx, len(self.color_seek.order)-1)]}"}
                    if self.color_seek_active else self.controller.get_progress()
                ),
                "grounding": self.vla_agent.latest_grounding,
                "has_api_key": bool(self.vla_agent.api_key),
                "mode": "hjepa_dag" if self.orchestrator.is_task_active else ("colorseek" if self.color_seek_active else getattr(self.vla_agent, "active_mode", "embedded")),
                "last_api_error": getattr(self.vla_agent, "last_api_error", None),
            },
            "cost": self.vla_agent.get_cost_summary(),
            "hjepa": {
                "active": True,
                "latest_residual": round(getattr(self, "latest_hjepa_residual", 0.0), 4),
                "anomaly_count": getattr(self, "hjepa_anomaly_count", 0),
                "active_subgoal": getattr(self, "active_subgoal_desc", "Idle"),
                "subgoal_count": getattr(self, "total_subgoals_count", 0),
            },
            "dialogue": self.supervisor.to_dict(),
            "narrative_logs": self.narrator_agent.flush_pending(),
            "is_paused": self.is_paused,
            "manual_mode": self.manual_mode,
        }

        # Send logs ONLY when new log events occur, rather than bundling recent_logs on every tick
        if self.pending_logs:
            payload["logs"] = list(self.pending_logs)
            self.pending_logs.clear()

        return payload

    async def broadcast_telemetry(self, payload: Dict[str, Any]):
        for ws in list(self.websockets):
            try:
                await ws.send_json(payload)
            except Exception:
                self.websockets.discard(ws)


class SessionManager:
    """Registry managing active and idle user sessions with a maximum capacity of 5."""
    MAX_SESSIONS: int = 5

    def __init__(self, max_sessions: int = 5):
        self.MAX_SESSIONS = max_sessions
        self.sessions: Dict[str, UserSession] = {}

    def get_candidate_to_evict(self, exclude_sid: Optional[str] = None) -> Optional[UserSession]:
        """
        Finds the session to evict when capacity exceeds MAX_SESSIONS:
        1. Prioritize idle sessions with 0 websockets and no active tasks (oldest created_at).
        2. Otherwise, select the session whose connection was opened earliest (first connected).
        """
        candidates = [s for s in self.sessions.values() if s.session_id != exclude_sid]
        if not candidates:
            return None

        # 1. Look for idle sessions with no open websockets and no active tasks
        idle_sessions = [
            s for s in candidates
            if len(s.websockets) == 0 and not s.controller.is_active and not s.orchestrator.is_task_active
        ]
        if idle_sessions:
            return min(idle_sessions, key=lambda s: s.created_at)

        # 2. All sessions have connections or tasks: evict the one whose connection was opened earliest
        return min(
            candidates,
            key=lambda s: s.first_connected_at if s.first_connected_at is not None else s.created_at
        )

    async def ensure_capacity(self, incoming_session_id: Optional[str] = None):
        """Asynchronously evicts the earliest opened session if at or above MAX_SESSIONS."""
        sid = (incoming_session_id or "default").strip()
        if sid in self.sessions:
            return

        while len(self.sessions) >= self.MAX_SESSIONS:
            victim = self.get_candidate_to_evict(exclude_sid=sid)
            if not victim:
                break
            victim_id = victim.session_id
            print(f"[*] Max sessions ({self.MAX_SESSIONS}) exceeded. Closing earliest opened session: {victim_id}")
            await victim.close_connections(
                f"Maximum of {self.MAX_SESSIONS} concurrent sessions reached. Earliest connection ({victim_id}) was closed."
            )
            if victim_id in self.sessions:
                del self.sessions[victim_id]

    def ensure_capacity_sync(self, incoming_session_id: Optional[str] = None):
        """Synchronously evicts the earliest opened session if at or above MAX_SESSIONS."""
        sid = (incoming_session_id or "default").strip()
        if sid in self.sessions:
            return

        while len(self.sessions) >= self.MAX_SESSIONS:
            victim = self.get_candidate_to_evict(exclude_sid=sid)
            if not victim:
                break
            victim_id = victim.session_id
            print(f"[*] Max sessions ({self.MAX_SESSIONS}) exceeded. Closing earliest opened session: {victim_id}")
            victim.close_connections_sync(
                f"Maximum of {self.MAX_SESSIONS} concurrent sessions reached. Earliest connection ({victim_id}) was closed."
            )
            if victim_id in self.sessions:
                del self.sessions[victim_id]

    def get_or_create(self, session_id: Optional[str]) -> UserSession:
        sid = (session_id or "default").strip()
        if not sid:
            sid = "default"
        if sid not in self.sessions:
            self.ensure_capacity_sync(sid)
            self.sessions[sid] = UserSession(sid)
        sess = self.sessions[sid]
        sess.last_active = time.time()
        return sess

    def get_active_sessions(self) -> list[UserSession]:
        active = []
        for sid, sess in list(self.sessions.items()):
            if sid == "default" or sess.is_active():
                active.append(sess)
        return active

    def cleanup_inactive(self, max_idle_seconds: int = 600):
        now = time.time()
        for sid, sess in list(self.sessions.items()):
            if sid != "default" and not sess.websockets and not sess.controller.is_active:
                if (now - sess.last_active) > max_idle_seconds:
                    print(f"[*] Reclaiming idle session: {sid}")
                    del self.sessions[sid]


session_manager = SessionManager()

# Global backwards-compatible singleton references
default_session = session_manager.get_or_create("default")
sim = default_session.sim
mapper = default_session.mapper
explorer = default_session.explorer
vla_agent = default_session.vla_agent
controller = default_session.controller
color_seek = default_session.color_seek
recent_logs = default_session.recent_logs

def log_event(category: str, badge: str, what: str, how: str):
    return session_manager.get_or_create("default").log_event(category, badge, what, how)

def handle_reset():
    return session_manager.get_or_create("default").handle_reset()

async def handle_vla_command(cmd: str):
    return await session_manager.get_or_create("default").handle_vla_command(cmd)

def extract_session_id(request: Request, body_session_id: Optional[str] = None) -> str:
    if body_session_id and body_session_id.strip():
        return body_session_id.strip()
    header_sid = request.headers.get("X-Session-ID") or request.headers.get("x-session-id")
    if header_sid and header_sid.strip():
        return header_sid.strip()
    query_sid = request.query_params.get("session_id")
    if query_sid and query_sid.strip():
        return query_sid.strip()
    return "default"


# Background Simulation Loop
async def simulation_loop():
    last_time = time.time()
    cleanup_timer = time.time()

    while True:
        now = time.time()
        elapsed = now - last_time

        if elapsed >= FIXED_DT:
            last_time = now

            for sid, sess in list(session_manager.sessions.items()):
                try:
                    # 1. Automatically suspend session after 25 minutes of inactivity
                    # Never suspend while arm is actively executing a task/trajectory!
                    if (
                        not sess.is_suspended
                        and not sess.controller.is_active
                        and not sess.orchestrator.is_task_active
                        and not sess.color_seek_active
                        and not sess.explorer.is_exploring
                        and (now - sess.last_active) >= SESSION_MAX_DURATION
                    ):
                        await sess.suspend()
                        continue

                    # 2. Skip simulation step and broadcast if suspended
                    if sess.is_suspended:
                        continue

                    # 2b. If paused, skip physics integration but broadcast static telemetry at 6 Hz
                    if sess.is_paused:
                        if sess.sim.tick % 10 == 0 and sess.websockets:
                            payload = sess.get_telemetry_payload(sess.sim.get_snapshot())
                            await sess.broadcast_telemetry(payload)
                        continue

                    # 3. For sessions with no connections and no active task, skip
                    if sid != "default" and not sess.is_active():
                        continue

                    snapshot = sess.step()
                    if snapshot and sess.websockets:
                        payload = sess.get_telemetry_payload(snapshot)
                        await sess.broadcast_telemetry(payload)
                except Exception as e:
                    print(f"[Simulation Error in session {sess.session_id}]: {e}")

            if now - cleanup_timer >= 60.0:
                cleanup_timer = now
                session_manager.cleanup_inactive(max_idle_seconds=600)

        rem = FIXED_DT - (time.time() - last_time)
        await asyncio.sleep(max(0.002, rem if rem > 0 else 0.002))


@asynccontextmanager
async def lifespan(app: FastAPI):
    sim_task = asyncio.create_task(simulation_loop())
    yield
    sim_task.cancel()


app = FastAPI(title="AERO: Agentic Execution for Robotic Operations", lifespan=lifespan)

# WebSocket Endpoint with Session ID Support
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = (websocket.query_params.get("session_id") or "default").strip()
    resume_flag = websocket.query_params.get("resume")
    await session_manager.ensure_capacity(session_id)
    sess = session_manager.get_or_create(session_id)
    if sess.first_connected_at is None:
        sess.first_connected_at = time.time()
    if resume_flag in ("1", "true", "yes"):
        sess.resume()
    await websocket.accept()
    sess.websockets.add(websocket)
    sess.last_active = time.time()

    # If currently suspended, notify the client immediately
    if sess.is_suspended:
        try:
            await websocket.send_json({
                "type": "session_suspended",
                "is_suspended": True,
                "session_id": sess.session_id,
                "reason": sess.suspension_reason or "Session auto-suspended after 25 minutes to conserve cloud bandwidth.",
                "duration_minutes": 25,
            })
        except Exception:
            pass
    else:
        # Immediately push initial world telemetry frame so 3D blocks & joints mount synchronously in frame 0
        try:
            init_snapshot = sess.sim.get_snapshot()
            init_payload = sess.get_telemetry_payload(init_snapshot)
            await websocket.send_json(init_payload)
        except Exception:
            pass

        if sess.recent_logs:
            # Sync initial logs once on connect
            try:
                await websocket.send_json({
                    "type": "logs_sync",
                    "logs": sess.recent_logs[-40:],
                })
            except Exception:
                pass

    try:
        while True:
            data = await websocket.receive_json()
            sess.last_active = time.time()
            msg_type = data.get("type")
            if msg_type == "resume":
                sess.resume()
            elif msg_type == "command":
                if sess.is_suspended:
                    sess.resume()
                cmd = data.get("command", "")
                await sess.handle_vla_command(cmd)
            elif msg_type == "teleop":
                if sess.is_suspended:
                    sess.resume()
                joints = data.get("joints", {})
                grip = data.get("gripper")
                sess.manual_mode = True
                sess.controller.stop()
                sess.explorer.is_exploring = False
                if joints:
                    sess.sim.set_joint_targets(joints, grip)
            elif msg_type == "reset":
                if sess.is_suspended:
                    sess.resume()
                sess.handle_reset()
            elif msg_type == "toggle_pause":
                if sess.is_suspended:
                    sess.resume()
                sess.is_paused = not sess.is_paused
    except WebSocketDisconnect:
        sess.websockets.discard(websocket)


@app.post("/api/vla/test_key")
async def api_test_key(req: TestKeyRequest, request: Request):
    sid = extract_session_id(request, req.session_id)
    sess = session_manager.get_or_create(sid)
    if req.model:
        sess.vla_agent.model_name = req.model
    valid, message = sess.vla_agent.test_api_key(req.api_key)
    if valid:
        sess.vla_agent.set_api_key(req.api_key)
    return {"valid": valid, "message": message}


@app.post("/api/vla/command")
async def api_command(req: CommandRequest, request: Request):
    sid = extract_session_id(request, req.session_id)
    sess = session_manager.get_or_create(sid)
    if sess.is_suspended:
        sess.resume()
    if req.api_key:
        sess.vla_agent.set_api_key(req.api_key)
    if req.model:
        sess.vla_agent.model_name = req.model

    res = await sess.handle_vla_command(req.command)
    return res


@app.post("/api/vla/pause")
async def api_pause(req: PauseRequest, request: Request):
    sid = extract_session_id(request, req.session_id)
    sess = session_manager.get_or_create(sid)
    if sess.is_suspended:
        sess.resume()
    sess.is_paused = True
    pause_res = sess.supervisor.handle_pause_request(requested_by_user=True, reason=req.reason)
    return {"status": "ok", "paused": True, "dialogue": pause_res}


@app.post("/api/vla/guidance")
async def api_guidance(req: GuidanceRequest, request: Request):
    sid = extract_session_id(request, req.session_id)
    sess = session_manager.get_or_create(sid)
    if sess.is_suspended:
        sess.resume()
    guide_res = sess.supervisor.process_guidance(req.guidance, sess.orchestrator, sess.sim)
    if guide_res.get("resumed"):
        sess.is_paused = False
    return {"status": "ok", **guide_res}


@app.post("/api/vla/chat")
async def api_chat(req: ChatRequest, request: Request):
    sid = extract_session_id(request, req.session_id)
    sess = session_manager.get_or_create(sid)
    if sess.is_suspended:
        sess.resume()
    msg = req.message.strip().lower()
    if any(w in msg for w in ("pause", "wait", "hold", "stop")) and not sess.is_paused and sess.orchestrator.is_task_active:
        sess.is_paused = True
        pause_res = sess.supervisor.handle_pause_request(requested_by_user=True, reason="Operator paused via chat")
        return {"status": "ok", "action": "PAUSED", "dialogue": pause_res}
    elif sess.supervisor.state in ("PAUSED_BY_USER", "PAUSED_BY_ANOMALY"):
        guide_res = sess.supervisor.process_guidance(req.message, sess.orchestrator, sess.sim)
        if guide_res.get("resumed"):
            sess.is_paused = False
        return {"status": "ok", "action": "GUIDANCE_PROCESSED", **guide_res}
    else:
        return await sess.handle_vla_command(req.message)


@app.post("/api/vla/chat/reset")
async def api_chat_reset(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    if sess.is_suspended:
        sess.resume()
    sess.supervisor.reset()
    return {
        "status": "ok",
        "message": "Chat section and dialogue state reset.",
        "dialogue": sess.supervisor.chat_history,
    }


@app.post("/api/vla/explore")
async def api_explore(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    if sess.is_suspended:
        sess.resume()
    sess.manual_mode = False
    sess.handle_reset()
    sess.explorer.start_exploration()
    sess.vla_agent.execution_status = "EXECUTING"
    sess.vla_agent.latest_thought = "Active frontier exploration initiated. Scanning table quadrants."
    sess.log_event(
        "motion",
        "EXPLORE",
        "Active frontier exploration triggered",
        "Workspace survey engaged. Updating 3D semantic voxel occupancy grid."
    )
    return {"status": "ok", "message": "Exploration initiated"}


@app.post("/api/sim/reset")
async def api_reset(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    if sess.is_suspended:
        sess.resume()
    sess.handle_reset(randomize=False, reset_blocks=True)
    return {"status": "ok", "message": "Arm returned to rest pose and workspace reset."}


@app.post("/api/sim/randomize")
async def api_randomize(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    if sess.is_suspended:
        sess.resume()
    sess.handle_reset(randomize=True, reset_blocks=True)
    return {"status": "ok", "message": "Blocks spawned at randomized collision-free tabletop coordinates."}


@app.post("/api/sim/nudge")
async def api_nudge_block(request: Request, block_id: Optional[str] = None):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    if sess.is_suspended:
        sess.resume()
    result = sess.sim.nudge_random_block(block_id=block_id)
    if not result:
        return {"status": "noop", "message": "No available tabletop block to nudge cleanly."}

    # Refresh camera perception & semantic map immediately after external disturbance
    obs = sess.camera.capture()
    robot_obs = sess.robot_sensor.get_state()
    dets = sess.localizer.localize(sess.detector.detect(obs), obs)
    tracks = sess.tracker.update(dets)
    sess.mapper.update_from_perception(tracks, robot_obs)

    sess.log_event(
        "perception",
        "DISTURBANCE",
        f"Physical disturbance applied: {result['name']} nudged on tabletop.",
        f"Displaced from [{result['old_position'][0]:.3f}, {result['old_position'][2]:.3f}] to "
        f"[{result['new_position'][0]:.3f}, {result['new_position'][2]:.3f}]. Semantic map re-localized."
    )
    return {"status": "ok", **result}


@app.post("/api/session/resume")
async def api_resume_session(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    sess.resume()
    return {"status": "ok", "message": "Session resumed", "session_id": sid}



@app.post("/api/sim/rest")
async def api_rest(request: Request):
    return await api_reset(request)


@app.get("/api/logs")
async def api_get_logs(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    return {"logs": sess.recent_logs}


@app.post("/api/logs/clear")
async def api_clear_logs(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    sess.recent_logs.clear()
    sess.log_event("system", "CLEARED", "Execution logs cleared by user.", "Log buffer flushed.")
    return {"status": "ok", "logs": sess.recent_logs}


@app.post("/api/sim/toggle_pause")
async def api_toggle_pause(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    sess.is_paused = not sess.is_paused
    return {"is_paused": sess.is_paused}


@app.post("/api/sim/teleop")
async def api_teleop(req: TeleopRequest, request: Request):
    sid = extract_session_id(request, req.session_id)
    sess = session_manager.get_or_create(sid)
    sess.manual_mode = True
    sess.controller.stop()
    sess.explorer.is_exploring = False
    if req.joints:
        sess.sim.set_joint_targets(req.joints, req.gripper)
    return {"status": "ok"}


@app.get("/api/vla/camera_image")
async def api_camera_image(request: Request, annotated: bool = True):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    snapshot = sess.sim.get_snapshot()
    img = render_camera_frame(snapshot, annotated=annotated, target_block_id=sess.active_target_block_id)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/vla/depth_image")
async def api_depth_image(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    snapshot = sess.sim.get_snapshot()
    img = render_depth_visual(snapshot, size=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/vla/wrist_image")
async def api_wrist_image(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    snapshot = sess.sim.get_snapshot()
    img = render_wrist_camera_frame(snapshot, size=128, target_block_id=sess.active_target_block_id)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/vla/map_image")
async def api_map_image(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    snapshot = sess.sim.get_snapshot()
    tcp_pos = tuple(snapshot.get("tcp", {}).get("position", [0.08, 0.85, 0.0]))
    img = render_occupancy_map_visual(
        sess.explorer.to_dict(),
        list(sess.tracker.tracks.values()),
        tcp_pos=tcp_pos,
        size=128,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/semantic_map")
async def api_semantic_map(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    return sess.mapper.to_dict()


@app.get("/api/vla/cost")
async def api_get_cost(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    return sess.vla_agent.get_cost_summary()


@app.post("/api/vla/cost/reset")
async def api_reset_cost(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    sess.vla_agent.reset_cost()
    return {"status": "ok", "cost": sess.vla_agent.get_cost_summary()}


@app.get("/api/telemetry")
async def api_telemetry(request: Request, session_id: Optional[str] = None):
    sid = extract_session_id(request, session_id)
    sess = session_manager.get_or_create(sid)
    snapshot = sess.sim.get_snapshot()
    return sess.get_telemetry_payload(snapshot)


@app.get("/api/session/info")
async def api_session_info(request: Request):
    sid = extract_session_id(request)
    sess = session_manager.get_or_create(sid)
    elapsed = time.time() - sess.session_start_time
    remaining = max(0.0, SESSION_MAX_DURATION - elapsed)
    return {
        "session_id": sid,
        "active_sessions_count": len(session_manager.sessions),
        "max_sessions": session_manager.MAX_SESSIONS,
        "websockets_count": len(sess.websockets),
        "has_api_key": bool(sess.vla_agent.api_key),
        "is_suspended": sess.is_suspended,
        "elapsed_seconds": round(elapsed, 1),
        "remaining_seconds": round(remaining, 1),
        "max_duration_seconds": SESSION_MAX_DURATION,
    }


# Static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    home_file = os.path.join(static_dir, "home.html")
    if os.path.exists(home_file):
        with open(home_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    index_file = os.path.join(static_dir, "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/home", response_class=HTMLResponse)
async def serve_home_alias():
    return await serve_home()


@app.get("/studio", response_class=HTMLResponse)
async def serve_studio():
    studio_file = os.path.join(static_dir, "index.html")
    if os.path.exists(studio_file):
        with open(studio_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Studio file not found</h1>", status_code=404)


@app.get("/exploded", response_class=HTMLResponse)
@app.get("/anatomy", response_class=HTMLResponse)
async def serve_exploded():
    exploded_file = os.path.join(static_dir, "exploded.html")
    if os.path.exists(exploded_file):
        with open(exploded_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Exploded View file not found</h1>", status_code=404)


@app.get("/logs", response_class=HTMLResponse)
async def serve_logs():
    logs_file = os.path.join(static_dir, "logs.html")
    if os.path.exists(logs_file):
        with open(logs_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Execution Logs file not found</h1>", status_code=404)


if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=False), name="static")
