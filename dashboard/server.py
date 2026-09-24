"""
server.py

FastAPI Web Server and Real-Time WebSocket Broadcaster for AERO Cognitive Dashboard.
Integrates local Gemma Cognitive Brain, Semantic Spatial Memory, and real-time
multi-room simulation telemetry.
"""

import asyncio
import json
import math
import os
import sys
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.sim_engine import LiveSimulationSession
from ros2_ws.src.robot_controller.robot_controller.gemma_brain_node import GemmaCognitiveReasoner

app = FastAPI(title="AERO Cognitive Embodied Dashboard")

# Connected WebSocket clients
active_connections: List[WebSocket] = []

# Global session & cognitive brain state
current_session = LiveSimulationSession()
gemma_brain = GemmaCognitiveReasoner()
is_nav_running = False
sim_task: Optional[asyncio.Task] = None
last_gemma_thought = "System initialized. Standing by for natural language navigation commands."


class ChatRequest(BaseModel):
    message: str


class RunRequest(BaseModel):
    problem_statement: str = "Navigate to the Red Sofa"
    mode: str = "navigate"  # "navigate" or "explore"


async def broadcast(message: Dict[str, Any]) -> None:
    """Broadcast JSON message to all connected WebSocket clients."""
    if not active_connections:
        return
    payload = json.dumps(message)
    dead_connections = []
    for ws in active_connections:
        try:
            await ws.send_text(payload)
        except Exception:
            dead_connections.append(ws)
    for ws in dead_connections:
        if ws in active_connections:
            active_connections.remove(ws)


def compute_navigation_step(sim: LiveSimulationSession) -> tuple[float, float]:
    """
    Computes smooth differential drive navigation commands toward active target
    with real-time obstacle avoidance.
    """
    dx = sim.arena.target_x - sim.robot.x
    dy = sim.arena.target_y - sim.robot.y
    dist = math.hypot(dx, dy)
    desired_yaw = math.atan2(dy, dx)
    yaw_err = (desired_yaw - sim.robot.yaw + math.pi) % (2.0 * math.pi) - math.pi

    ranges, min_dist = sim.robot.compute_lidar(sim.arena)
    n = len(ranges)
    front = min(ranges[:int(n * 0.12)] + ranges[-int(n * 0.12):])
    left = min(ranges[int(n * 0.12):int(n * 0.38)])
    right = min(ranges[int(n * 0.62):int(n * 0.88)])

    # Obstacle avoidance override
    if front < 0.55:
        linear_cmd = 0.04
        angular_cmd = 1.1 if left > right else -1.1
    else:
        # P-controller heading alignment
        if abs(yaw_err) > 0.55:  # ~30 degrees
            linear_cmd = 0.06
            angular_cmd = max(-1.4, min(1.4, 1.6 * yaw_err))
        else:
            linear_cmd = min(0.24, 0.22 * dist)
            angular_cmd = max(-1.2, min(1.2, 1.3 * yaw_err))

    return linear_cmd, angular_cmd


async def run_navigation_loop() -> None:
    """Runs high-frequency 20 Hz simulation ticks while streaming frames to frontend."""
    global is_nav_running, current_session

    is_nav_running = True
    dt = 0.05

    while current_session.is_active and is_nav_running:
        lin, ang = compute_navigation_step(current_session)
        frame = current_session.tick(lin, ang, dt=dt)
        frame["gemma_thought"] = last_gemma_thought
        await broadcast(frame)
        await asyncio.sleep(dt)

    is_nav_running = False
    await broadcast({
        "type": "status_update",
        "status": current_session.status,
        "reason": current_session.reason
    })
    await broadcast({
        "type": "log",
        "level": "success" if current_session.status == "PASSED" else "error",
        "message": f"Navigation finished: [{current_session.status}] {current_session.reason} (Remaining: {current_session.get_state()['goal_distance']:.2f}m)"
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        # Initial greeting payload
        state = current_session.get_state()
        state["gemma_thought"] = last_gemma_thought
        await websocket.send_text(json.dumps(state))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.post("/api/chat")
async def handle_natural_language_chat(req: ChatRequest):
    """
    Cognitive Chat Endpoint:
    Sends user natural language command to Gemma reasoner, decides plan,
    updates simulation target, and dispatches navigation.
    """
    global last_gemma_thought, sim_task, is_nav_running

    user_text = req.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message.")

    # Get known landmarks from simulation arena
    landmarks_dict = {
        obj["id"]: {
            "name": obj["name"],
            "category": obj["category"],
            "room": obj["room"],
            "x": obj["x"],
            "y": obj["y"]
        }
        for obj in current_session.arena.semantic_objects
    }

    # Reason with Gemma
    decision = gemma_brain.parse_instruction(user_text, landmarks_dict)
    last_gemma_thought = decision["thought"]

    await broadcast({
        "type": "log",
        "level": "info",
        "message": f"🗣️ User: \"{user_text}\""
    })
    await broadcast({
        "type": "log",
        "level": "warning",
        "message": f"🧠 Gemma Thought: {decision['thought']}"
    })
    await broadcast({
        "type": "log",
        "level": "success",
        "message": f"🤖 Gemma Reply: {decision['reply']}"
    })

    # Dispatch action
    if decision["action"] in ["NAVIGATE_TO_OBJECT", "DOCK"] and decision["coordinates"]:
        coords = decision["coordinates"]
        current_session.set_target(coords[0], coords[1], name=decision["target"])
        if not is_nav_running:
            sim_task = asyncio.create_task(run_navigation_loop())
    elif decision["action"] == "EXPLORE":
        # Target uncharted space
        current_session.set_target(-2.2, 2.5, name="Kitchen (Uncharted)")
        if not is_nav_running:
            sim_task = asyncio.create_task(run_navigation_loop())

    return decision


@app.get("/api/status")
async def get_status():
    state = current_session.get_state()
    state["gemma_thought"] = last_gemma_thought
    state["is_running"] = is_nav_running
    return state


@app.post("/api/run")
async def start_run(req: RunRequest):
    global sim_task
    if is_nav_running:
        raise HTTPException(status_code=400, detail="Navigation already running.")
    current_session.reset()
    sim_task = asyncio.create_task(run_navigation_loop())
    return {"status": "started"}


@app.post("/api/stop")
async def stop_run():
    global is_nav_running
    is_nav_running = False
    current_session.is_active = False
    await broadcast({
        "type": "log",
        "level": "warning",
        "message": "🛑 User paused navigation."
    })
    return {"status": "stopped"}


# Mount static assets
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "AERO Cognitive Dashboard API running."})


if __name__ == "__main__":
    import uvicorn
    print("Starting AERO Cognitive Embodied Dashboard on http://127.0.0.1:8000 ...")
    uvicorn.run("dashboard.server:app", host="127.0.0.1", port=8000, reload=True)
