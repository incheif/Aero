"""
process_cleaner.py

Robust process tree hygiene utility for AERO.
Ensures lingering Gazebo and ROS 2 processes (gzserver, gzclient, daemon, nodes)
are cleanly terminated between trial iterations to prevent hanging or port conflicts.
"""

import os
import platform
import subprocess
import time
from typing import List

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

TARGET_PROCESS_NAMES = [
    'gzserver',
    'gzclient',
    'ign gazebo',
    'gz sim',
    'oracle_node',
    'controller_node',
    'robot_state_publisher',
    'spawn_entity.py',
    '_ros2_daemon',
]


def cleanup_ros_and_gazebo(timeout_sec: float = 3.0) -> int:
    """
    Terminates all running simulation and ROS 2 trial processes.
    
    Args:
        timeout_sec: Maximum seconds to wait for graceful SIGTERM before SIGKILL.
        
    Returns:
        Number of terminated processes.
    """
    terminated_count = 0
    is_windows = platform.system() == 'Windows'

    # Step 1: Terminate using psutil if available
    if HAS_PSUTIL:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or []).lower()
                pname = (proc.info['name'] or '').lower()

                matches = any(
                    target.lower() in pname or target.lower() in cmdline
                    for target in TARGET_PROCESS_NAMES
                )
                if matches and proc.pid != os.getpid():
                    proc.terminate()
                    terminated_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        # Wait for termination, escalate to kill if stubborn
        gone, alive = psutil.wait_procs(
            [p for p in psutil.process_iter() if any(
                t.lower() in (p.name() or '').lower() for t in TARGET_PROCESS_NAMES
            )],
            timeout=timeout_sec
        )
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    # Step 2: Fallback OS commands
    if not is_windows:
        for target in ['gzserver', 'gzclient']:
            subprocess.run(
                ['pkill', '-9', '-f', target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
    else:
        for target in ['gzserver.exe', 'gzclient.exe']:
            subprocess.run(
                ['taskkill', '/F', '/IM', target, '/T'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

    # Step 3: Flush ROS 2 daemon discovery cache
    try:
        subprocess.run(
            ['ros2', 'daemon', 'stop'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0
        )
    except Exception:
        pass

    # Small pause to allow socket and file lock release
    time.sleep(0.3)
    return terminated_count


if __name__ == '__main__':
    killed = cleanup_ros_and_gazebo()
    print(f"Cleaned up {killed} lingering simulation/ROS processes.")
