"""
tests/test_failure_detector_agent.py

Comprehensive Unit Tests for Autonomous Failure Detector Agent:
1. Stall / Hang detection (motion commanded but arm stationary).
2. Waypoint timeout (stopped in between waypoints).
3. Subgoal deadlock (not able to move to next task).
4. Initial task failure (not able to do first task / unreachable target).
5. Immediate safety stop and logging integration into Session.
"""

import pytest
from vla.agents.failure_detector import FailureDetectorAgent
from dashboard.server import session_manager


class DummySim:
    def __init__(self, tcp_pos=(0.3, 0.75, 0.0), blocks=None):
        self.tcp_pos = list(tcp_pos)
        self.blocks = blocks or [
            {"id": "block_cyan", "position": [0.35, 0.74, 0.0]},
            {"id": "block_orange", "position": [0.45, 0.74, 0.0]}
        ]

    def get_snapshot(self):
        return {
            "tcp": {"position": list(self.tcp_pos)},
            "joints": {"base": 0.0, "shoulder": 0.0, "elbow": 0.0, "wrist": 0.0, "gripper": 0.0},
            "blocks": list(self.blocks),
        }


class DummyController:
    def __init__(self, is_active=True, target_pos=(0.45, 0.85, 0.0), step_idx=0):
        self.is_active = is_active
        self.target_pos = target_pos
        self.step_idx = step_idx
        self.current_plan = [
            {"desc": "Transit to target", "target": target_pos, "dwell_ticks": 10}
        ]
        self.stopped = False

    def stop(self):
        self.is_active = False
        self.stopped = True


class DummySubgoal:
    def __init__(self, sg_id="sg_1", desc="Move block", block_id="block_cyan", target_slot=(0.45, 0.85, 0.0)):
        self.id = sg_id
        self.description = desc
        self.parameters = {"block_id": block_id, "target_slot": target_slot}


class DummyOrchestrator:
    def __init__(self, is_task_active=True, active_subgoal=None):
        self.is_task_active = is_task_active
        self.active_subgoal = active_subgoal or DummySubgoal()
        self.stopped = False

    def stop_task(self):
        self.is_task_active = False
        self.stopped = True


def test_failure_detector_initial_task_failure():
    detector = FailureDetectorAgent()
    sim = DummySim()

    # Case 1: Empty subgoals list
    fail1 = detector.evaluate_initial_feasibility("invalid command", [], sim)
    assert fail1 is not None
    assert fail1["failed"] is True
    assert fail1["type"] == "INITIAL_TASK_FAILURE"

    # Case 2: Target block missing from simulation
    missing_sg = DummySubgoal(block_id="block_nonexistent")
    fail2 = detector.evaluate_initial_feasibility("pick nonexistent", [missing_sg], sim)
    assert fail2 is not None
    assert fail2["failed"] is True
    assert fail2["type"] == "TARGET_UNAVAILABLE"

    # Case 3: Target block valid
    valid_sg = DummySubgoal(block_id="block_cyan")
    fail3 = detector.evaluate_initial_feasibility("pick cyan", [valid_sg], sim)
    assert fail3 is None


def test_failure_detector_stall_detection():
    # Set a low threshold (5 ticks) for fast unit testing
    detector = FailureDetectorAgent(stall_tick_threshold=5)
    detector.start_monitoring("test command")

    sim = DummySim(tcp_pos=(0.2, 0.75, 0.0))
    controller = DummyController(is_active=True, target_pos=(0.5, 0.85, 0.0))
    orchestrator = DummyOrchestrator(is_task_active=True)

    # Arm stationary across ticks
    fail = None
    for _ in range(6):
        fail = detector.evaluate(sim, controller, orchestrator)
        if fail:
            break

    assert fail is not None
    assert fail["failed"] is True
    assert fail["type"] == "MOTION_STALLED"
    assert "hanged" in fail["reason"].lower() or "stopped" in fail["reason"].lower()


def test_failure_detector_waypoint_timeout():
    detector = FailureDetectorAgent()
    detector.start_monitoring("test waypoint timeout")

    sim = DummySim(tcp_pos=(0.4, 0.80, 0.0))
    controller = DummyController(is_active=True, target_pos=(0.4, 0.80, 0.0), step_idx=0)
    orchestrator = DummyOrchestrator(is_task_active=False)

    # Simulate waypoint stuck for > 320 ticks
    detector.waypoint_ticks = 315
    detector.current_waypoint_idx = 0

    fail = None
    for _ in range(10):
        fail = detector.evaluate(sim, controller, orchestrator)
        if fail:
            break

    assert fail is not None
    assert fail["failed"] is True
    assert fail["type"] == "WAYPOINT_TIMEOUT"


def test_failure_detector_subgoal_deadlock():
    # Set low deadlock threshold (5 ticks)
    detector = FailureDetectorAgent(deadlock_tick_threshold=5)
    detector.start_monitoring("test deadlock")

    sim = DummySim(tcp_pos=(0.3, 0.75, 0.0))
    controller = DummyController(is_active=False)
    sg = DummySubgoal(sg_id="deadlocked_subgoal", desc="Unachievable goal")
    orchestrator = DummyOrchestrator(is_task_active=True, active_subgoal=sg)

    fail = None
    for _ in range(6):
        fail = detector.evaluate(sim, controller, orchestrator)
        if fail:
            break

    assert fail is not None
    assert fail["failed"] is True
    assert fail["type"] == "SUBGOAL_DEADLOCK"
    assert "deadlock" in fail["reason"].lower() or "advance" in fail["reason"].lower()


def test_session_handle_failure_detected_integration():
    sess = session_manager.get_or_create("test_failure_session")
    sess.controller.is_active = True
    sess.orchestrator.is_task_active = True

    failure_data = {
        "failed": True,
        "type": "MOTION_STALLED",
        "reason": "Robot arm has hanged",
        "explanation": "Actuator stalled against obstacle at X=0.32, Y=0.75.",
        "timestamp": "12:00:00"
    }

    sess.handle_failure_detected(failure_data)

    # Verify execution stopped
    assert sess.controller.is_active is False
    assert sess.orchestrator.is_task_active is False
    assert sess.supervisor.state == "FAILED"
    assert sess.vla_agent.execution_status == "FAILED"

    # Verify log recorded
    failure_logs = [l for l in sess.recent_logs if l.get("badge") == "FAILURE"]
    assert len(failure_logs) > 0
    assert "MOTION_STALLED" in failure_logs[-1]["what"]

    # Verify robot chat message posted
    chat_msgs = sess.supervisor.chat_history
    assert len(chat_msgs) > 0
    assert chat_msgs[-1]["sender"] == "robot"
    assert chat_msgs[-1]["status_tag"] == "FAILED"
    assert "Robot arm has hanged" in chat_msgs[-1]["text"]
