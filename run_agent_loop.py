#!/usr/bin/env python3
"""
run_agent_loop.py

Main closed-loop iterative driver for AERO.
Coordinates:
  1. Static interface inspection (introspector.py)
  2. Colcon workspace build (builder.py)
  3. Headless Gazebo evaluation (runner.py + oracle_node.py)
  4. Diagnostic prompt construction on failure
  5. Code synthesis and patching (patcher.py)
  6. Retry budgeting (default N=5)
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional

from agent_harness.builder import build_workspace, BuildResult
from agent_harness.patcher import patch_controller_code, revert_controller_code
from agent_harness.introspector import inspect_controller_interfaces, GraphInspectionResult
from agent_harness.runner import run_evaluation, EvaluationResult
from agent_harness.process_cleaner import cleanup_ros_and_gazebo


def create_diagnostic_prompt(
    iteration: int,
    current_code: str,
    inspection_result: GraphInspectionResult,
    build_result: Optional[BuildResult],
    eval_result: Optional[EvaluationResult]
) -> str:
    """Builds a structured, high-context prompt for the LLM to diagnose and fix the controller."""
    prompt = [
        "### AERO Autonomous Agent Diagnostic Report",
        f"**Trial Iteration:** {iteration}",
        "",
        "**Benchmark Goal:**",
        "Navigate differential drive robot from (0.0, 0.0) to target (3.0, 3.0) within 0.1m tolerance.",
        "Constraint: 30 simulation seconds budget, zero collisions with obstacles or boundary walls.",
        "",
    ]

    # Pre-flight inspection feedback
    if not inspection_result.valid:
        prompt.append("❌ **Interface Introspection Errors:**")
        for err in inspection_result.missing_interfaces:
            prompt.append(f"  - Missing required interface: {err}")
        prompt.append("")

    # Build failure feedback
    if build_result and not build_result.success:
        prompt.append("❌ **Colcon Build Failure:**")
        prompt.append("```text")
        prompt.append(build_result.isolated_error)
        prompt.append("```")
        prompt.append("")

    # Simulation / Oracle evaluation failure feedback
    if eval_result and not eval_result.is_passed():
        prompt.append("❌ **Simulation Benchmark Failure:**")
        prompt.append(f"- **Verdict:** {eval_result.status}")
        prompt.append(f"- **Failure Reason:** {eval_result.reason}")
        prompt.append(f"- **Final Distance to Goal:** {eval_result.final_distance:.3f} m (Goal: <= 0.10 m)")
        prompt.append(f"- **Simulation Time Elapsed:** {eval_result.time_elapsed_sec:.2f} s (Budget: 30.0 s)")
        prompt.append(f"- **Minimum Obstacle Clearance:** {eval_result.min_obstacle_distance:.3f} m")
        if eval_result.error_log_snippet:
            prompt.append(f"- **Log Diagnostic:**\n```text\n{eval_result.error_log_snippet}\n```")
        prompt.append("")

    prompt.append("### Current Controller Source Code:")
    prompt.append("```python")
    prompt.append(current_code)
    prompt.append("```")
    prompt.append("")
    prompt.append("### Instructions for Agent:")
    prompt.append("1. Analyze the exact failure cause (compiler error, collision, starvation, or timeout).")
    prompt.append("2. Output ONLY the corrected, complete Python code for `controller_node.py` wrapped in ```python ... ```.")
    prompt.append("3. Ensure it defines a valid ROS 2 node that subscribes to `/odom` and `/scan`, and publishes to `/cmd_vel`.")

    return "\n".join(prompt)


def mock_llm_code_repair(prompt: str, current_code: str, iteration: int) -> str:
    """
    Mock agent repair engine used for testing and deterministic validation
    when no live LLM API key is supplied.
    """
    # Simply ensure the controller has sound parameters
    repaired_code = current_code.replace(
        "obstacle_threshold = 0.45",
        "obstacle_threshold = 0.55"
    )
    return repaired_code


def extract_python_code(response_text: str) -> str:
    """Extracts Python code block from LLM markdown response."""
    if "```python" in response_text:
        parts = response_text.split("```python")
        code_part = parts[1].split("```")[0]
        return code_part.strip()
    elif "```" in response_text:
        parts = response_text.split("```")
        return parts[1].strip()
    return response_text.strip()


def run_agent_loop(
    max_retries: int = 5,
    use_docker: bool = False,
    api_provider: Optional[str] = None
) -> bool:
    """
    Executes the autonomous agentic closed loop.
    
    Returns:
        True if benchmark PASSED, False if retry budget exhausted.
    """
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    controller_file = os.path.join(
        workspace_root, 'ros2_ws', 'src', 'robot_controller', 'robot_controller', 'controller_node.py'
    )
    history_file = os.path.join(workspace_root, 'run_history.json')

    print("==================================================================")
    print("⚡ AERO: Closed-Loop ROS 2 Autonomous Navigation Driver")
    print(f"Max Retry Budget: {max_retries} | Containerized: {use_docker}")
    print("==================================================================")

    history: List[Dict[str, Any]] = []
    trial_passed = False

    for iteration in range(1, max_retries + 1):
        print(f"\n[Iteration {iteration}/{max_retries}] Starting evaluation cycle...")

        # 1. Read current code
        with open(controller_file, 'r', encoding='utf-8') as f:
            current_code = f.read()

        # 2. Static ROS interface inspection
        print("  🔍 Running static ROS graph introspection...")
        inspection = inspect_controller_interfaces(controller_file)
        if not inspection.valid:
            print(f"  ⚠️  Static interface check failed: {inspection.missing_interfaces}")

        # 3. Colcon workspace build
        print("  🔨 Building ROS 2 workspace (robot_controller)...")
        build_result = build_workspace()
        if not build_result.success:
            print(f"  ❌ Build failed: {build_result.isolated_error[:120]}...")
            eval_result = None
        else:
            print("  ✅ Build successful.")

            # 4. Headless simulation benchmark trial
            print("  🚀 Launching headless Gazebo trial & Ground Truth Oracle (30s sim)...")
            eval_result = run_evaluation(timeout_sec=35.0, use_docker=use_docker)
            print(f"  📊 Status: [{eval_result.status}] | Reason: [{eval_result.reason}] | Distance: {eval_result.final_distance:.3f}m")

            if eval_result.is_passed():
                print(f"\n🎉 BENCHMARK PASSED at iteration {iteration}! Target reached within {eval_result.final_distance:.3f}m.")
                trial_passed = True
                history.append({
                    "iteration": iteration,
                    "status": "PASSED",
                    "reason": eval_result.reason,
                    "final_distance": eval_result.final_distance,
                    "time_elapsed_sec": eval_result.time_elapsed_sec,
                    "timestamp": time.time()
                })
                break

        # Record iteration history
        history.append({
            "iteration": iteration,
            "build_success": build_result.success,
            "status": eval_result.status if eval_result else "BUILD_FAILED",
            "reason": eval_result.reason if eval_result else "COMPILER_ERROR",
            "final_distance": eval_result.final_distance if eval_result else 999.0,
            "error_snippet": (eval_result.error_log_snippet if eval_result else build_result.isolated_error),
            "timestamp": time.time()
        })

        if iteration == max_retries:
            print(f"\n❌ Retry budget exhausted ({max_retries} iterations). Benchmark failed.")
            break

        # 5. Diagnostic Prompt Construction
        print("  🧠 Generating diagnostic prompt for agent...")
        prompt = create_diagnostic_prompt(
            iteration=iteration,
            current_code=current_code,
            inspection_result=inspection,
            build_result=build_result,
            eval_result=eval_result
        )

        # 6. LLM Patch Generation
        print("  💡 Requesting code patch from agent...")
        if api_provider == 'gemini' and 'GEMINI_API_KEY' in os.environ:
            from google import genai
            client = genai.Client()
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=prompt
            )
            new_code = extract_python_code(response.text)
        else:
            print("  ℹ️  Using deterministic mock patch generator (set GEMINI_API_KEY to use live LLM).")
            new_code = mock_llm_code_repair(prompt, current_code, iteration)

        # 7. Apply AST-validated patch
        patch_ok, msg = patch_controller_code(new_code, controller_file)
        if patch_ok:
            print(f"  ✏️  {msg}")
        else:
            print(f"  ❌ Patch rejected: {msg}")

    # Save run history artifact
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved trial timeline to '{history_file}'")

    cleanup_ros_and_gazebo()
    return trial_passed


def main():
    parser = argparse.ArgumentParser(description="AERO Closed-Loop ROS 2 Driver")
    parser.add_argument('--max-retries', type=int, default=5, help="Maximum agent self-correction retries")
    parser.add_argument('--docker', action='store_true', help="Run simulation trials inside Docker")
    parser.add_argument('--provider', type=str, default=None, choices=['gemini', 'openai', 'mock'], help="LLM provider")
    args = parser.parse_args()

    success = run_agent_loop(
        max_retries=args.max_retries,
        use_docker=args.docker,
        api_provider=args.provider
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
