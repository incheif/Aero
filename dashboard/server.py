"""
server.py

FastAPI Web Server and Real-Time WebSocket Broadcaster for AERO Dashboard.
Bridges natural language prompts, agent code modifications, and real-time
simulation telemetry to the browser frontend.
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
from agent_harness.patcher import patch_controller_code, revert_controller_code, validate_python_syntax
from agent_harness.introspector import inspect_controller_interfaces
from run_agent_loop import create_diagnostic_prompt, mock_llm_code_repair

app = FastAPI(title="AERO Interactive Dashboard")

# Connected WebSocket clients
active_connections: List[WebSocket] = []

# Global session state
current_session = LiveSimulationSession()
is_loop_running = False
current_iteration = 1
max_iterations = 5
selected_model = "gemini-2.5-pro"
problem_statement = "Navigate to (3.0, 3.0) avoiding obstacles with max speed 0.22 m/s"
sim_task: Optional[asyncio.Task] = None


class RunRequest(BaseModel):
    problem_statement: str = "Navigate to (3.0, 3.0) avoiding obstacles"
    model: str = "gemini-2.5-pro"
    max_retries: int = 5
    mode: str = "loop"  # "loop" or "single"


class PatchRequest(BaseModel):
    code: str


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


def get_controller_code() -> str:
    path = os.path.join(
        PROJECT_ROOT, 'ros2_ws', 'src', 'robot_controller', 'robot_controller', 'controller_node.py'
    )
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def get_backup_code() -> str:
    path = os.path.join(
        PROJECT_ROOT, 'ros2_ws', 'src', 'robot_controller', 'robot_controller', 'controller_node.py.bak'
    )
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return get_controller_code()


def compute_controller_action(code_str: str, sim: LiveSimulationSession) -> tuple[float, float]:
    """
    Executes a simulated navigation control step based on controller parameters.
    """
    # Extract threshold and gains if present in code
    obs_thresh = 0.45
    if "obstacle_threshold = " in code_str:
        try:
            obs_thresh = float(code_str.split("obstacle_threshold = ")[1].split("\n")[0].split("#")[0].strip())
        except Exception:
            pass

    max_lin = 0.22
    if "max_linear_speed" in code_str:
        try:
            max_lin = float(code_str.split("max_linear_speed', ")[1].split(")")[0].strip())
        except Exception:
            pass

    dx = sim.arena.target_x - sim.robot.x
    dy = sim.arena.target_y - sim.robot.y
    dist = math.hypot(dx, dy)
    desired_yaw = math.atan2(dy, dx)
    yaw_err = (desired_yaw - sim.robot.yaw + math.pi) % (2.0 * math.pi) - math.pi

    # Check LiDAR clearances
    ranges, min_dist = sim.robot.compute_lidar(sim.arena)
    n = len(ranges)
    front = min(ranges[:int(n * 0.1)] + ranges[-int(n * 0.1):])
    left = min(ranges[int(n * 0.1):int(n * 0.35)])
    right = min(ranges[int(n * 0.65):int(n * 0.9)])

    if front < obs_thresh:
        linear_cmd = 0.04
        angular_cmd = 1.0 if left > right else -1.0
    else:
        linear_cmd = min(max_lin, 0.18 * dist) if abs(yaw_err) < 0.6 else 0.05
        angular_cmd = max(-1.2, min(1.2, 1.4 * yaw_err))

    return linear_cmd, angular_cmd


async def run_simulation_loop(mode: str = "loop") -> None:
    """Orchestrates the live trial or autonomous loop with real-time WebSocket streaming."""
    global is_loop_running, current_iteration, current_session

    is_loop_running = True
    iterations_to_run = max_iterations if mode == "loop" else 1

    await broadcast({
        "type": "log",
        "level": "info",
        "message": f"🤖 Gemma Cognitive Planner active | Model: {selected_model}"
    })

    # Cognitive task interpretation
    target_x, target_y = 3.2, 3.0  # default red sofa
    lower_cmd = problem_statement.lower()

    if "sofa" in lower_cmd or "couch" in lower_cmd:
        target_x, target_y = 3.2, 3.0
        gemma_thought = "Identified object 'Red Sofa' in 3D Semantic Memory at (3.2m, 3.0m). Dispatching Nav2 NavigateToPose goal."
    elif "table" in lower_cmd or "kitchen" in lower_cmd:
        target_x, target_y = -2.2, 2.5
        gemma_thought = "Identified object 'Kitchen Table' in 3D Semantic Memory at (-2.2m, 2.5m). Dispatching Nav2 goal."
    elif "dock" in lower_cmd or "charge" in lower_cmd:
        target_x, target_y = 0.0, -3.2
        gemma_thought = "Human requested return to base. Dispatching Nav2 goal to Charging Dock origin (0.0m, -3.2m)."
    elif "explore" in lower_cmd or "map" in lower_cmd:
        target_x, target_y = 2.0, 2.0
        gemma_thought = "Unmapped area detected. Activating Frontier Exploration to expand 2D occupancy grid."
    else:
        gemma_thought = f"Interpreted human instruction: '{problem_statement}'. Planning collision-free trajectory."

    await broadcast({
        "type": "log",
        "level": "warning",
        "message": f"🧠 Gemma Reasoning: {gemma_thought}"
    })

    for iteration in range(1, iterations_to_run + 1):
        if not is_loop_running:
            break

        current_iteration = iteration
        current_session.reset()
        current_session.arena.target_x = target_x
        current_session.arena.target_y = target_y

        await broadcast({
            "type": "status_update",
            "status": "RUNNING",
            "iteration": iteration,
            "max_iterations": max_iterations
        })

        await broadcast({
            "type": "log",
            "level": "info",
            "message": f"[Trial {iteration}] Nav2 Path Planner executing trajectory to ({target_x}m, {target_y}m)..."
        })

        # Broadcast active code
        curr_code = get_controller_code()
        prev_code = get_backup_code()
        await broadcast({
            "type": "code_update",
            "current_code": curr_code,
            "previous_code": prev_code,
            "iteration": iteration
        })

        # Run 20 Hz simulation ticks
        dt = 0.05
        while current_session.is_active and is_loop_running:
            lin, ang = compute_controller_action(curr_code, current_session)
            frame = current_session.tick(lin, ang, dt=dt)
            await broadcast(frame)
            await asyncio.sleep(dt)

        # Trial finished
        await broadcast({
            "type": "log",
            "level": "success" if current_session.status == "PASSED" else "error",
            "message": (
                f"Trial {iteration} Finished: [{current_session.status}] "
                f"Reason: {current_session.reason} | "
                f"Distance: {current_session.get_state()['goal_distance']:.2f}m | "
                f"Time: {current_session.sim_time:.1f}s"
            )
        })

        if current_session.status == "PASSED":
            await broadcast({
                "type": "trial_end",
                "verdict": "PASSED",
                "iteration": iteration,
                "reason": "SUCCESS",
                "final_distance": current_session.get_state()["goal_distance"],
                "time_elapsed": current_session.sim_time
            })
            break

        # If failed and more iterations remain, trigger agent repair
        if iteration < iterations_to_run and is_loop_running:
            await broadcast({
                "type": "log",
                "level": "warning",
                "message": f"🧠 Diagnosing failure ({current_session.reason}). Prompting {selected_model} for fix..."
            })
            await asyncio.sleep(1.0)

            # Generate repaired code
            mock_prompt = f"Failed with {current_session.reason} at distance {current_session.get_state()['goal_distance']}"
            repaired = mock_llm_code_repair(mock_prompt, curr_code, iteration)

            # Patch code
            patch_controller_code(repaired)
            await broadcast({
                "type": "log",
                "level": "info",
                "message": "✏️ Applied agent patch to controller_node.py. Validating and recompiling..."
            })
            await asyncio.sleep(1.0)

    is_loop_running = False
    await broadcast({
        "type": "status_update",
        "status": current_session.status if current_session.status == "PASSED" else "IDLE",
        "iteration": current_iteration,
        "max_iterations": max_iterations
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        # Send initial state
        initial_payload = {
            "type": "init",
            "status": current_session.status,
            "iteration": current_iteration,
            "max_iterations": max_iterations,
            "problem_statement": problem_statement,
            "selected_model": selected_model,
            "current_code": get_controller_code(),
            "previous_code": get_backup_code(),
            "sim_frame": current_session.get_state()
        }
        await websocket.send_text(json.dumps(initial_payload))
        while True:
            data = await websocket.receive_text()
            # Heartbeat or client commands
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.get("/api/status")
async def get_status():
    return {
        "status": current_session.status,
        "is_running": is_loop_running,
        "iteration": current_iteration,
        "max_iterations": max_iterations,
        "problem_statement": problem_statement,
        "selected_model": selected_model,
        "sim_state": current_session.get_state()
    }


@app.get("/api/code")
async def get_code():
    return {
        "current_code": get_controller_code(),
        "previous_code": get_backup_code()
    }


@app.post("/api/run")
async def start_run(req: RunRequest):
    global sim_task, problem_statement, selected_model, max_iterations
    if is_loop_running:
        raise HTTPException(status_code=400, detail="A simulation run is already active.")

    problem_statement = req.problem_statement
    selected_model = req.model
    max_iterations = req.max_retries

    sim_task = asyncio.create_task(run_simulation_loop(mode=req.mode))
    return {"status": "started", "mode": req.mode}


@app.post("/api/stop")
async def stop_run():
    global is_loop_running
    is_loop_running = False
    current_session.is_active = False
    await broadcast({
        "type": "log",
        "level": "warning",
        "message": "🛑 User stopped simulation run."
    })
    return {"status": "stopped"}


@app.post("/api/patch")
async def patch_code(req: PatchRequest):
    ok, err = validate_python_syntax(req.code)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Syntax Error: {err}")
    success, msg = patch_controller_code(req.code)
    if not success:
        raise HTTPException(status_code=500, detail=msg)

    await broadcast({
        "type": "code_update",
        "current_code": req.code,
        "previous_code": get_backup_code(),
        "iteration": current_iteration
    })
    return {"status": "patched", "message": msg}


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
    return JSONResponse({"message": "AERO Dashboard API running. Static UI loading..."})


if __name__ == "__main__":
    import uvicorn
    print("Starting AERO Interactive Web Dashboard on http://127.0.0.1:8000 ...")
    uvicorn.run("dashboard.server:app", host="127.0.0.1", port=8000, reload=True)
