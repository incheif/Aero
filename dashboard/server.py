"""
FastAPI Server for Robot Arm VLA Simulation & Three.js Studio.
Streams real-time 60 FPS physics telemetry over WebSockets and handles
Google VLA natural-language commands, 3D semantic mapping, and frontier exploration.
"""

import os
import sys
import io
import asyncio
import time
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
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
)
from simulation.physics_engine import RobotArmSimulation
from simulation.camera import render_camera_frame, encode_image_base64
from vla.semantic_mapper import SemanticSpatialMapper
from vla.frontier_explorer import TabletopFrontierExplorer
from vla.google_vla_agent import GoogleVLAAgent
from vla.trajectory_controller import TrajectoryController
from vla.color_seek_policy import ColorSeekPolicy

# Core singletons
sim = RobotArmSimulation()
mapper = SemanticSpatialMapper()
explorer = TabletopFrontierExplorer()
vla_agent = GoogleVLAAgent()
controller = TrajectoryController()
color_seek = ColorSeekPolicy()

is_paused: bool = False
manual_mode: bool = False
color_seek_active: bool = False

VLA_STRIDE = 12  # 5 Hz policy updates over 60 Hz physics (vsarena standard)

class CommandRequest(BaseModel):
    command: str
    api_key: Optional[str] = None
    model: Optional[str] = "gemini-2.5-flash"

class TestKeyRequest(BaseModel):
    api_key: str
    model: Optional[str] = "gemini-2.5-flash"

class TeleopRequest(BaseModel):
    joints: Optional[Dict[str, float]] = None
    gripper: Optional[float] = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

# System execution logs buffer (maintains what is happening and how)
recent_logs: list[Dict[str, Any]] = []

def log_event(category: str, badge: str, what: str, how: str):
    """
    Records an execution or cognitive event.
    category: 'vla' | 'motion' | 'contact' | 'stability' | 'system'
    badge: e.g. 'COMMAND', 'PLAN', 'MOTION', 'GRIP', 'RELEASE', 'STABILITY', 'RESET', 'REJECT'
    what: high-level description of what is happening
    how: technical parameters, metric targets, clearance, dwell ticks, velocities, etc.
    """
    entry = {
        "id": f"log_{int(time.time()*1000)}_{len(recent_logs)}",
        "timestamp": time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 100):02d}",
        "category": category,
        "badge": badge,
        "what": what,
        "how": how,
    }
    recent_logs.append(entry)
    if len(recent_logs) > 250:
        recent_logs.pop(0)
    return entry

# Initial boot log
log_event("system", "STARTUP", "Simulation engine and Google VLA studio initialized.", "Rapier3D physics running at 60 FPS (dt=16.6ms). 5 blocks spawned at canonical table coordinates.")

# Background Simulation Loop
async def simulation_loop():
    global is_paused, manual_mode, color_seek_active
    last_time = time.time()
    fps_timer = time.time()
    frame_count = 0
    current_fps = 60.0
    prev_step_idx = -1
    prev_stability_verified = False

    while True:
        now = time.time()
        elapsed = now - last_time

        if elapsed >= FIXED_DT:
            last_time = now

            if not is_paused:
                # 1. Controller step if active plan
                fk_data = sim.get_snapshot()["tcp"]["position"]
                tcp_pos = (fk_data[0], fk_data[1], fk_data[2])

                if controller.is_active and not manual_mode:
                    target_cmd = controller.step(tcp_pos, sim.joints)
                    if target_cmd:
                        sim.set_joint_targets(target_cmd["joints"], target_cmd["gripper"])
                        cur_idx = controller.step_idx
                        if cur_idx != prev_step_idx and cur_idx < len(controller.current_plan):
                            prev_step_idx = cur_idx
                            step_data = controller.current_plan[cur_idx]
                            s_desc = step_data.get("desc", f"Waypoint {cur_idx + 1}")
                            s_tgt = step_data.get("target", (0, 0, 0))
                            s_grip = step_data.get("gripper", 0.0)
                            s_dwell = step_data.get("dwell_ticks", 10)
                            is_contact = "GRIP" in s_desc.upper() or "RELEASE" in s_desc.upper()
                            s_cat = "contact" if is_contact else "motion"
                            s_badge = "GRIP" if "GRIP" in s_desc.upper() else ("RELEASE" if "RELEASE" in s_desc.upper() else "IK MOTION")
                            log_event(
                                s_cat,
                                s_badge,
                                f"Waypoint {cur_idx + 1}/{len(controller.current_plan)}: {s_desc}",
                                f"Target TCP: [{s_tgt[0]:.3f}, {s_tgt[1]:.3f}, {s_tgt[2]:.3f}] · Jaws: {'Closed (100%)' if s_grip > 0.5 else 'Open (100%)'} · Dwell: {s_dwell} ticks contact settle"
                            )

                        if target_cmd.get("is_active") is False:
                            vla_agent.execution_status = "COMPLETED"
                            vla_agent.latest_thought = f"Completed action plan: {target_cmd.get('step_desc')}"
                            log_event(
                                "system",
                                "COMPLETED",
                                f"Action completed: {target_cmd.get('step_desc')}",
                                "All waypoints successfully executed to metric tolerances. Arm holding in standby pose."
                            )

                # 1b. ColorSeek closed-loop visual servoing step (5 Hz downsampled VLA stride)
                elif color_seek_active and not manual_mode:
                    if sim.tick % VLA_STRIDE == 0:
                        cam_frame = render_camera_frame(sim.get_snapshot())
                        cs_cmd = color_seek.step(cam_frame, sim.joints, tcp_pos)
                        sim.set_joint_targets(cs_cmd["joints"], cs_cmd["gripper"])
                        vla_agent.latest_thought = f"ColorSeek (5 Hz VLA): {cs_cmd.get('plan')}"
                        if not cs_cmd.get("is_active", True):
                            color_seek_active = False
                            vla_agent.execution_status = "COMPLETED"
                            vla_agent.latest_thought = "ColorSeek: Closed-loop visual stacking sequence completed!"

                # 2. Frontier exploration step if active
                elif explorer.is_exploring and not manual_mode:
                    wp = explorer.get_current_waypoint()
                    target_pos = wp["target"]
                    dx = target_pos[0] - tcp_pos[0]
                    dy = target_pos[1] - tcp_pos[1]
                    dz = target_pos[2] - tcp_pos[2]
                    dist = (dx*dx + dy*dy + dz*dz) ** 0.5

                    from simulation.kinematics import inverse_kinematics
                    ik_targets = inverse_kinematics(target_pos, gripper=0.0)
                    sim.set_joint_targets(ik_targets, gripper=0.0)

                    if dist <= 0.02:
                        explorer.advance_waypoint()
                        mapper.total_scanned_area_pct = explorer.exploration_pct
                        if not explorer.is_exploring:
                            vla_agent.latest_thought = "Exploration complete. 3D Semantic Spatial Memory fully mapped."
                            vla_agent.execution_status = "IDLE"

                # 3. Step physics simulation
                snapshot = sim.step(FIXED_DT)

                # 4. Update 3D Semantic Spatial Mapper
                mapper.update_from_simulation(snapshot)

                # FPS calculations
                frame_count += 1
                if time.time() - fps_timer >= 1.0:
                    current_fps = frame_count / (time.time() - fps_timer)
                    fps_timer = time.time()
                    frame_count = 0

                # Stability state change logging
                stab = snapshot.get("scores", {}).get("stability", {})
                is_ver = stab.get("is_verified", False)
                if is_ver and not prev_stability_verified:
                    prev_stability_verified = True
                    spat_acc = snapshot.get("scores", {}).get("spatial_accuracy", 0.0)
                    log_event(
                        "stability",
                        "VERIFIED",
                        "Post-release physical stability hold verified (18/18 ticks = 0.30s).",
                        f"Target stack held without toppling. Linear velocity <= 0.02 m/s, angular velocity <= 0.05 rad/s. Spatial accuracy score: {spat_acc*100:.1f}%."
                    )
                elif not is_ver:
                    prev_stability_verified = False

                # 5. Broadcast live telemetry state
                payload = {
                    "type": "telemetry",
                    "tick": snapshot["tick"],
                    "fps": round(current_fps, 1),
                    "joints": snapshot["joints"],
                    "joint_targets": snapshot["joint_targets"],
                    "arm": snapshot["arm"],
                    "tcp": snapshot["tcp"],
                    "blocks": snapshot["blocks"],
                    "grasped_block_id": snapshot["grasped_block_id"],
                    "target_zone": snapshot["target_zone"],
                    "scores": snapshot.get("scores", {}),
                    "tcp_trail": controller.tcp_trail[-60:],
                    "semantic_map": mapper.to_dict(),
                    "frontier": explorer.to_dict(),
                    "vla": {
                        "model": "ColorSeek (Visual Servoing)" if color_seek_active else vla_agent.model_name,
                        "status": vla_agent.execution_status,
                        "thought": vla_agent.latest_thought,
                        "progress": controller.get_progress() if not color_seek_active else {"current_step": color_seek.target_idx + 1, "total_steps": len(color_seek.order), "pct": int((color_seek.target_idx / max(1, len(color_seek.order))) * 100)},
                        "grounding": vla_agent.latest_grounding,
                        "has_api_key": bool(vla_agent.api_key),
                        "mode": "colorseek" if color_seek_active else getattr(vla_agent, "active_mode", "embedded"),
                        "last_api_error": getattr(vla_agent, "last_api_error", None),
                    },
                    "cost": vla_agent.get_cost_summary(),
                    "is_paused": is_paused,
                    "manual_mode": manual_mode,
                    "logs": recent_logs[-40:],
                }
                await manager.broadcast(payload)

        # Yield to event loop adaptively based on time until next 60 Hz tick
        rem = FIXED_DT - (time.time() - last_time)
        await asyncio.sleep(max(0.002, rem if rem > 0 else 0.002))

@asynccontextmanager
async def lifespan(app: FastAPI):
    sim_task = asyncio.create_task(simulation_loop())
    yield
    sim_task.cancel()

app = FastAPI(title="Robot Arm Google VLA Simulation", lifespan=lifespan)

# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            if msg_type == "command":
                cmd = data.get("command", "")
                await handle_vla_command(cmd)
            elif msg_type == "teleop":
                joints = data.get("joints", {})
                grip = data.get("gripper")
                sim.set_joint_targets(joints, grip)
            elif msg_type == "reset":
                handle_reset()
            elif msg_type == "toggle_pause":
                global is_paused
                is_paused = not is_paused
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/vla/test_key")
async def api_test_key(req: TestKeyRequest):
    if req.model:
        vla_agent.model_name = req.model
    valid, message = vla_agent.test_api_key(req.api_key)
    if valid:
        vla_agent.set_api_key(req.api_key)
    return {"valid": valid, "message": message}

@app.post("/api/vla/command")
async def api_command(req: CommandRequest):
    if req.api_key:
        vla_agent.set_api_key(req.api_key)
    if req.model:
        vla_agent.model_name = req.model

    res = await handle_vla_command(req.command)
    return res

async def handle_vla_command(cmd: str) -> Dict[str, Any]:
    global manual_mode
    manual_mode = False

    log_event(
        "vla",
        "INSTRUCTION",
        f"VLA instruction received: \"{cmd}\"",
        f"Dispatched to Google VLA reasoning pipeline (Model: {vla_agent.model_name}). Synthesizing spatial chain-of-thought."
    )

    # Automatically reset the system before every new action
    handle_reset()

    snapshot = sim.get_snapshot()
    img = render_camera_frame(snapshot)
    semantic_summary = mapper.get_semantic_summary()

    # Synthesize manipulation, gesture, or feasibility validation through Google VLA
    result = vla_agent.parse_instruction(cmd, img, snapshot, semantic_summary)

    # If rejected as impossible:
    if not result.get("is_possible", True) or result.get("status") == "IMPOSSIBLE":
        controller.stop()
        log_event(
            "vla",
            "REJECT",
            f"Instruction rejected as infeasible: \"{cmd}\"",
            f"Reason: {result.get('reasoning', 'Action violates spatial reachability or semantic stability constraint.')}"
        )
        return result

    # If ColorSeek closed-loop visual servoing was requested:
    if result.get("action_type") == "COLORSEEK":
        global color_seek_active
        controller.stop()
        explorer.is_exploring = False
        color_seek.reset()
        color_seek_active = True
        vla_agent.execution_status = "EXECUTING"
        vla_agent.latest_thought = "ColorSeek: Closed-loop RGB visual servoing active (5 Hz policy stride)."
        log_event(
            "vla",
            "COLORSEEK",
            "Visual servoing stacking policy initialized",
            "RGB camera tracking active at 5 Hz control policy stride. Visual servoing closed-loop engaged."
        )
        return result

    # If active frontier exploration was requested:
    if result.get("action_type") == "EXPLORE" or (
        any(w in cmd.lower() for w in ("explore", "scan table", "survey", "frontier"))
        and not any(w in cmd.lower() for w in ("wave", "tower", "stack", "pick"))
    ):
        explorer.start_exploration()
        controller.stop()
        color_seek_active = False
        vla_agent.execution_status = "EXECUTING"
        vla_agent.latest_thought = f"Google VLA: Executing active frontier exploration for '{cmd}'."
        log_event(
            "motion",
            "EXPLORE",
            f"Frontier exploration started: \"{cmd}\"",
            "Autonomous workspace survey engaged. Updating 3D semantic voxel grid."
        )
        return {"status": "ok", "action": "EXPLORATION_STARTED", "thought": vla_agent.latest_thought}

    # Load and execute trajectory plan (gestures, stacking, pick-and-place)
    color_seek_active = False
    plan = result.get("plan", [])
    if plan:
        controller.load_plan(plan)
        log_event(
            "vla",
            "PLAN READY",
            f"Synthesized trajectory plan with {len(plan)} waypoints.",
            f"Action: {result.get('action_type', 'TASK')} · CoT reasoning: {result.get('reasoning', result.get('thought', 'Metric waypoint sequence dispatched.'))}"
        )
    return result

@app.post("/api/vla/explore")
async def api_explore():
    global manual_mode, color_seek_active
    manual_mode = False
    handle_reset()
    explorer.start_exploration()
    vla_agent.execution_status = "EXECUTING"
    vla_agent.latest_thought = "Active frontier exploration initiated. Scanning table quadrants."
    log_event(
        "motion",
        "EXPLORE",
        "Active frontier exploration triggered",
        "Workspace survey engaged. Updating 3D semantic voxel occupancy grid."
    )
    return {"status": "ok", "message": "Exploration initiated"}

@app.post("/api/sim/reset")
async def api_reset():
    handle_reset()
    return {"status": "ok", "message": "Arm returned to rest pose and workspace reset."}

@app.post("/api/sim/rest")
async def api_rest():
    handle_reset()
    return {"status": "ok", "message": "Arm returned to rest pose and workspace reset."}

@app.get("/api/logs")
async def api_get_logs():
    return {"logs": recent_logs}

@app.post("/api/logs/clear")
async def api_clear_logs():
    recent_logs.clear()
    log_event("system", "CLEARED", "Execution logs cleared by user.", "Log buffer flushed.")
    return {"status": "ok", "logs": recent_logs}

def handle_reset():
    global color_seek_active
    color_seek_active = False
    color_seek.reset()
    controller.stop()
    explorer.is_exploring = False
    explorer.exploration_pct = 25.0
    for k in explorer.explored_cells:
        explorer.explored_cells[k] = 0.25
    sim.reset()
    mapper.update_from_simulation(sim.get_snapshot())
    vla_agent.execution_status = "RESTING"
    vla_agent.latest_thought = "Arm returned to rest configuration. Ready for new command."
    log_event(
        "system",
        "RESET",
        "Workspace and arm reset to canonical state.",
        "Joint targets zeroed/rest pose. Blocks re-spawned at baseline coordinates. Trajectory queue stopped."
    )

@app.post("/api/sim/toggle_pause")
async def api_toggle_pause():
    global is_paused
    is_paused = not is_paused
    return {"is_paused": is_paused}

@app.post("/api/sim/teleop")
async def api_teleop(req: TeleopRequest):
    global manual_mode
    manual_mode = True
    controller.stop()
    explorer.is_exploring = False
    if req.joints:
        sim.set_joint_targets(req.joints, req.gripper)
    return {"status": "ok"}

@app.get("/api/vla/camera_image")
async def api_camera_image():
    snapshot = sim.get_snapshot()
    img = render_camera_frame(snapshot)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/api/semantic_map")
async def api_semantic_map():
    return mapper.to_dict()

@app.get("/api/vla/cost")
async def api_get_cost():
    return vla_agent.get_cost_summary()

@app.post("/api/vla/cost/reset")
async def api_reset_cost():
    vla_agent.reset_cost()
    return {"status": "ok", "cost": vla_agent.get_cost_summary()}

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

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=False), name="static")
