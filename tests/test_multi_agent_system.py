"""
tests/test_multi_agent_system.py

Unit and integration tests for AERO's multi-agent collaborative framework:
  1. EpisodicMemoryAgent (learning from mistakes and human corrections)
  2. NarrativeLoggerAgent (explainable plain-English telemetry logging)
  3. InstructionDecomposerAgent (step decomposition and plain-English proposals)
  4. HumanDialogueSupervisorAgent (conversational pauses, guidance ingestion, and resumption)
"""

import os
import pytest
from simulation.physics_engine import RobotArmSimulation
from vla.agents.episodic_memory import EpisodicMemoryAgent
from vla.agents.narrator import NarrativeLoggerAgent
from vla.agents.decomposer import InstructionDecomposerAgent
from vla.agents.dialogue_supervisor import HumanDialogueSupervisorAgent
from vla.closed_loop_recovery import ClosedLoopRecoveryOrchestrator


def test_episodic_memory_agent(tmp_path):
    storage = str(tmp_path / "test_memory.json")
    agent = EpisodicMemoryAgent(storage_path=storage)

    # Initially empty
    assert len(agent.episodes) == 0

    # Record a failure resolved by human guidance
    agent.record_episode(
        task_instruction="make a letter D with blocks",
        subgoal_id="sg_0_clear",
        failure_type="PAUSED_BY_USER",
        entity="block_orange",
        human_guidance="lift 4cm higher to clear the pad",
        adaptation={"lift_offset_y": 0.04},
        outcome="SUCCESS",
    )

    assert len(agent.episodes) == 1

    # Query experience for similar task & entity
    exp = agent.query_experience(
        task_instruction="make a letter D with blocks",
        entity="block_orange",
    )
    assert exp is not None
    assert exp["learned_from_guidance"] == "lift 4cm higher to clear the pad"
    assert exp["recommended_adaptation"]["lift_offset_y"] == 0.04

    # Verify disk persistence
    agent2 = EpisodicMemoryAgent(storage_path=storage)
    assert len(agent2.episodes) == 1
    assert agent2.episodes[0]["entity"] == "block_orange"


def test_narrative_logger_agent():
    narrator = NarrativeLoggerAgent(max_history=50)

    # Log an approach step
    entry1 = narrator.explain_subgoal_transition(
        subgoal_idx=1,
        total_subgoals=6,
        skill_type="DEPOSIT",
        entity="block_cyan",
        slot_desc="bottom-left spine",
    )
    assert "Placing Cyan Cube" in entry1["title"]
    assert entry1["category"] == "place"

    # Log an anomaly
    entry2 = narrator.explain_anomaly(
        subgoal_desc="Lifting block",
        residual=5.2,
        entity="block_yellow",
    )
    assert "Yellow" in entry2["explanation"]
    assert entry2["severity"] == "warning"

    # Verify flush
    pending = narrator.flush_pending()
    assert len(pending) == 2
    assert len(narrator.pending_narratives) == 0
    assert len(narrator.narrative_history) == 2


def test_instruction_decomposer_agent():
    sim = RobotArmSimulation()
    decomposer = InstructionDecomposerAgent()

    # Decompose letter D
    res = decomposer.decompose("make a letter D with blocks", sim)
    assert res["is_possible"] is True
    assert len(res["subgoals"]) >= 6
    assert len(res["easy_steps"]) >= 6
    assert "Step 1" in res["proposal_message"]
    assert res["easy_steps"][0]["step_number"] == 1

    # Test impossible command rejection
    rej = decomposer.decompose("build a tower of 20 blocks", sim)
    assert rej["is_possible"] is False
    assert rej["status"] == "IMPOSSIBLE"


def test_human_dialogue_supervisor_flow(tmp_path):
    sim = RobotArmSimulation()
    orchestrator = ClosedLoopRecoveryOrchestrator()
    orchestrator.start_task(sim, "make a letter D with blocks")

    mem = EpisodicMemoryAgent(storage_path=str(tmp_path / "supervisor_memory.json"))
    supervisor = HumanDialogueSupervisorAgent(memory_agent=mem)
    supervisor.current_instruction = "make a letter D with blocks"
    supervisor.active_entity = "block_cyan"
    supervisor.current_step_idx = 0

    # User triggers pause
    pause_res = supervisor.handle_pause_request(requested_by_user=True, reason="User clicked pause")
    assert pause_res["state"] == "PAUSED_BY_USER"
    assert len(pause_res["options"]) > 0
    assert supervisor.state == "PAUSED_BY_USER"

    # User gives guidance: "use purple instead"
    guide_res = supervisor.process_guidance("use purple instead of cyan", orchestrator, sim)
    assert guide_res["resumed"] is True
    assert guide_res["action"] == "SWAP_TARGET_ENTITY"
    assert supervisor.state == "EXECUTING"
    assert orchestrator.active_subgoal.entity == "block_purple"

    # Check that episodic memory captured the interaction
    assert len(supervisor.memory.episodes) == 1
    assert supervisor.memory.episodes[0]["adaptation"]["new_entity"] == "block_purple"
