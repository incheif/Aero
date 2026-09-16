"""
runner.py

Execution runner and trial lifecycle supervisor for AERO.
Starts the headless Gazebo simulation and Ground Truth Oracle, supervises execution,
ensures clean process teardown, and parses the resulting test_results.json.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .process_cleaner import cleanup_ros_and_gazebo


@dataclass
class EvaluationResult:
    """Parsed outcome of a navigation simulation benchmark trial."""
    status: str  # "PASSED" | "FAILED"
    reason: str  # "SUCCESS" | "COLLISION" | "TIMEOUT" | "TOPIC_STARVATION" | "CRASH"
    final_distance: float
    time_elapsed_sec: float
    min_obstacle_distance: float
    error_log_snippet: str
    raw_json: Optional[Dict[str, Any]] = None

    def is_passed(self) -> bool:
        return self.status == "PASSED"

    def get_summary(self) -> str:
        return (
            f"Trial Result: [{self.status}] (Reason: {self.reason})\n"
            f"  - Final Distance to Target: {self.final_distance:.3f} m\n"
            f"  - Simulation Time Elapsed: {self.time_elapsed_sec:.2f} s\n"
            f"  - Min Obstacle Distance:   {self.min_obstacle_distance:.3f} m\n"
            + (f"  - Error Trace: {self.error_log_snippet}\n" if self.error_log_snippet else "")
        )


def run_evaluation(
    timeout_sec: float = 35.0,
    target_x: float = 3.0,
    target_y: float = 3.0,
    output_json_path: Optional[str] = None,
    use_docker: bool = False,
    docker_image: str = 'aero-ros2'
) -> EvaluationResult:
    """
    Executes a single headless simulation trial.

    Args:
        timeout_sec: Maximum wall-clock seconds before terminating the trial.
        target_x: Target coordinate X.
        target_y: Target coordinate Y.
        output_json_path: Path to write test_results.json. Defaults to workspace root.
        use_docker: If True, executes the trial inside the AERO Docker container.
        docker_image: Docker image name when use_docker is True.

    Returns:
        EvaluationResult dataclass.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(current_dir)

    if output_json_path is None:
        output_json_path = os.path.join(workspace_root, 'test_results.json')

    # 1. Clean previous run artifacts and lingering processes
    if os.path.exists(output_json_path):
        try:
            os.remove(output_json_path)
        except OSError:
            pass

    cleanup_ros_and_gazebo()

    # 2. Build execution command
    if use_docker:
        cmd = [
            'docker', 'run', '--rm',
            '--ipc=host', '--net=host',
            '-e', 'DISPLAY=',
            '-e', 'LIBGL_ALWAYS_SOFTWARE=1',
            '-e', 'QT_QPA_PLATFORM=offscreen',
            '-v', f"{workspace_root}:/workspace",
            '-w', '/workspace',
            docker_image,
            'bash', '-c',
            f"source /opt/ros/humble/setup.bash && "
            f"source /workspace/ros2_ws/install/setup.bash && "
            f"ros2 launch agent_evaluator eval_headless.launch.py "
            f"target_x:={target_x} target_y:={target_y} "
            f"timeout_sec:=30.0 output_json_path:=/workspace/test_results.json"
        ]
    else:
        # Native / WSL command
        cmd = [
            'ros2', 'launch', 'agent_evaluator', 'eval_headless.launch.py',
            f'target_x:={target_x}',
            f'target_y:={target_y}',
            'timeout_sec:=30.0',
            f'output_json_path:={output_json_path}'
        ]

    stdout_log = []
    stderr_log = []
    start_time = time.time()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=workspace_root
        )

        # Monitor execution up to timeout
        try:
            stdout_data, stderr_data = proc.communicate(timeout=timeout_sec)
            stdout_log.append(stdout_data)
            stderr_log.append(stderr_data)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_data, stderr_data = proc.communicate()
            stdout_log.append(stdout_data)
            stderr_log.append(stderr_data)

    except Exception as e:
        return EvaluationResult(
            status="FAILED",
            reason="CRASH",
            final_distance=999.0,
            time_elapsed_sec=round(time.time() - start_time, 2),
            min_obstacle_distance=0.0,
            error_log_snippet=f"Failed to launch evaluation process: {str(e)}"
        )
    finally:
        # Always guarantee clean process teardown
        cleanup_ros_and_gazebo()

    # 3. Parse generated test_results.json
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return EvaluationResult(
                status=data.get('status', 'FAILED'),
                reason=data.get('reason', 'UNKNOWN'),
                final_distance=float(data.get('final_distance', 999.0)),
                time_elapsed_sec=float(data.get('time_elapsed_sec', 0.0)),
                min_obstacle_distance=float(data.get('min_obstacle_distance', 0.0)),
                error_log_snippet=data.get('error_log_snippet', ''),
                raw_json=data
            )
        except (json.JSONDecodeError, OSError) as err:
            return EvaluationResult(
                status="FAILED",
                reason="CRASH",
                final_distance=999.0,
                time_elapsed_sec=round(time.time() - start_time, 2),
                min_obstacle_distance=0.0,
                error_log_snippet=f"Error reading test_results.json: {err}"
            )

    # 4. Fallback if test_results.json was not created (early crash)
    combined_err = "".join(stderr_log) + "\n" + "".join(stdout_log)
    snippet = "\n".join(combined_err.strip().splitlines()[-20:]) if combined_err.strip() else "Simulation exited with no output."

    return EvaluationResult(
        status="FAILED",
        reason="CRASH",
        final_distance=999.0,
        time_elapsed_sec=round(time.time() - start_time, 2),
        min_obstacle_distance=0.0,
        error_log_snippet=f"test_results.json was not produced by Oracle.\nLog snippet:\n{snippet}"
    )


if __name__ == '__main__':
    res = run_evaluation(timeout_sec=10)
    print(res.get_summary())
