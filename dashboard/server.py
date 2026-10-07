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

# Core singletons
sim = RobotArmSimulation()
mapper = SemanticSpatialMapper()
explorer = TabletopFrontierExplorer()
vla_agent = GoogleVLAAgent()
controller = TrajectoryController()

is_paused: bool = False
manual_mode: bool = False

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

# Background Simulation Loop
async def simulation_loop():
    global is_paused, manual_mode
    last_time = time.time()
    fps_timer = time.time()
    frame_count = 0
    current_fps = 60.0

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
                        if target_cmd.get("is_active") is False:
                            vla_agent.execution_status = "COMPLETED"
                            vla_agent.latest_thought = f"Completed action plan: {target_cmd.get('step_desc')}"

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
                    "tcp_trail": controller.tcp_trail[-60:],
                    "semantic_map": mapper.to_dict(),
                    "frontier": explorer.to_dict(),
                    "vla": {
                        "model": vla_agent.model_name,
                        "status": vla_agent.execution_status,
                        "thought": vla_agent.latest_thought,
                        "progress": controller.get_progress(),
                        "grounding": vla_agent.latest_grounding,
                        "has_api_key": bool(vla_agent.api_key),
                        "mode": getattr(vla_agent, "active_mode", "embedded"),
                        "last_api_error": getattr(vla_agent, "last_api_error", None),
                    },
                    "cost": vla_agent.get_cost_summary(),
                    "is_paused": is_paused,
                    "manual_mode": manual_mode,
                }
                await manager.broadcast(payload)

        # Yield to event loop
        await asyncio.sleep(0.002)

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
    snapshot = sim.get_snapshot()
    img = render_camera_frame(snapshot)
    semantic_summary = mapper.get_semantic_summary()

    # Synthesize manipulation, gesture, or feasibility validation through Google VLA
    result = vla_agent.parse_instruction(cmd, img, snapshot, semantic_summary)

    # If rejected as impossible:
    if not result.get("is_possible", True) or result.get("status") == "IMPOSSIBLE":
        controller.stop()
        return result

    # If active frontier exploration was requested:
    if result.get("action_type") == "EXPLORE" or (
        any(w in cmd.lower() for w in ("explore", "scan table", "survey", "frontier"))
        and not any(w in cmd.lower() for w in ("wave", "tower", "stack", "pick"))
    ):
        explorer.start_exploration()
        controller.stop()
        vla_agent.execution_status = "EXECUTING"
        vla_agent.latest_thought = f"Google VLA: Executing active frontier exploration for '{cmd}'."
        return {"status": "ok", "action": "EXPLORATION_STARTED", "thought": vla_agent.latest_thought}

    # Load and execute trajectory plan (gestures, stacking, pick-and-place)
    plan = result.get("plan", [])
    if plan:
        controller.load_plan(plan)
    return result

@app.post("/api/vla/explore")
async def api_explore():
    global manual_mode
    manual_mode = False
    controller.stop()
    explorer.start_exploration()
    vla_agent.execution_status = "EXECUTING"
    vla_agent.latest_thought = "Active frontier exploration initiated. Scanning table quadrants."
    return {"status": "ok", "message": "Exploration initiated"}

@app.post("/api/sim/reset")
async def api_reset():
    handle_reset()
    return {"status": "ok", "message": "Arm returned to rest pose and workspace reset."}

@app.post("/api/sim/rest")
async def api_rest():
    handle_reset()
    return {"status": "ok", "message": "Arm returned to rest pose and workspace reset."}

def handle_reset():
    controller.stop()
    explorer.is_exploring = False
    explorer.exploration_pct = 25.0
    for k in explorer.explored_cells:
        explorer.explored_cells[k] = 0.25
    sim.reset()
    mapper.update_from_simulation(sim.get_snapshot())
    vla_agent.execution_status = "RESTING"
    vla_agent.latest_thought = "Arm returned to rest configuration. Ready for new command."

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
