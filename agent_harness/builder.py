"""
builder.py

Colcon build automation and compiler error isolation tool for AERO.
Runs isolated package builds and extracts actionable compiler/linter error traces.
"""

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class BuildResult:
    """Structured outcome of a colcon workspace build."""
    success: bool
    returncode: int
    package_name: str
    stdout: str
    stderr: str
    isolated_error: str

    def get_summary(self) -> str:
        if self.success:
            return f"Build PASSED for package '{self.package_name}'."
        return (
            f"Build FAILED for package '{self.package_name}' (exit {self.returncode}):\n"
            f"{self.isolated_error or self.stderr or self.stdout}"
        )


def extract_key_errors(raw_output: str) -> str:
    """Filter out noisy colcon progress logs and extract relevant Python/C++ compiler errors."""
    lines = raw_output.splitlines()
    error_lines = []
    capture = False

    for line in lines:
        lower = line.lower()
        if any(keyword in lower for keyword in ['syntaxerror:', 'traceback', 'importerror:', 'error:', 'failed <<<']):
            capture = True
        if capture:
            error_lines.append(line)
            # Stop capturing after 30 lines to keep diagnostic prompt focused
            if len(error_lines) >= 30:
                break

    if error_lines:
        return "\n".join(error_lines)

    # Fallback to last 15 lines if no explicit keywords matched
    return "\n".join(lines[-15:]) if lines else "Unknown build failure."


def build_workspace(
    workspace_dir: Optional[str] = None,
    package_name: str = 'robot_controller',
    timeout_sec: float = 60.0
) -> BuildResult:
    """
    Executes `colcon build --packages-select <package_name>`.

    Args:
        workspace_dir: Path to ros2_ws directory. If None, auto-detected.
        package_name: Name of target ROS 2 package to build.
        timeout_sec: Build timeout budget.

    Returns:
        BuildResult with status and isolated error message.
    """
    if workspace_dir is None:
        # Default to ../ros2_ws relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.join(os.path.dirname(current_dir), 'ros2_ws')

    if not os.path.exists(workspace_dir):
        return BuildResult(
            success=False,
            returncode=-1,
            package_name=package_name,
            stdout="",
            stderr="",
            isolated_error=f"Workspace directory does not exist: {workspace_dir}"
        )

    cmd = ['colcon', 'build', '--packages-select', package_name]

    try:
        proc = subprocess.run(
            cmd,
            cwd=workspace_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec
        )
        combined_output = proc.stdout + "\n" + proc.stderr
        is_success = (proc.returncode == 0)
        isolated = "" if is_success else extract_key_errors(combined_output)

        return BuildResult(
            success=is_success,
            returncode=proc.returncode,
            package_name=package_name,
            stdout=proc.stdout,
            stderr=proc.stderr,
            isolated_error=isolated
        )
    except FileNotFoundError:
        return BuildResult(
            success=False,
            returncode=-2,
            package_name=package_name,
            stdout="",
            stderr="",
            isolated_error="'colcon' command not found. Ensure ROS 2 environment is sourced."
        )
    except subprocess.TimeoutExpired:
        return BuildResult(
            success=False,
            returncode=-3,
            package_name=package_name,
            stdout="",
            stderr="",
            isolated_error=f"colcon build timed out after {timeout_sec}s."
        )


if __name__ == '__main__':
    result = build_workspace()
    print(result.get_summary())
