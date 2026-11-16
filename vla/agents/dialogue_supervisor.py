"""
vla/agents/dialogue_supervisor.py

Human Dialogue Supervisor Agent for Collaborative Human-in-the-Loop Robotics.
Coordinates conversational turns, mid-motion human pauses, anomaly queries,
and corrective guidance integration.
"""

import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from vla.agents.episodic_memory import EpisodicMemoryAgent
from vla.agents.narrator import NarrativeLoggerAgent
from vla.agents.decomposer import InstructionDecomposerAgent


class HumanDialogueSupervisorAgent:
    """
    Supervises the conversational state machine and coordinates human-robot interaction.
    Allows the operator to pause, query, and guide the arm at any point during execution.
    """

    def __init__(
        self,
        memory_agent: Optional[EpisodicMemoryAgent] = None,
        narrator_agent: Optional[NarrativeLoggerAgent] = None,
        decomposer_agent: Optional[InstructionDecomposerAgent] = None,
    ):
        self.memory = memory_agent or EpisodicMemoryAgent()
        self.narrator = narrator_agent or NarrativeLoggerAgent()
        self.decomposer = decomposer_agent or InstructionDecomposerAgent()

        self.chat_history: List[Dict[str, Any]] = []
        self.state: str = "IDLE"  # IDLE, PLANNING, EXECUTING, PAUSED_BY_USER, PAUSED_BY_ANOMALY, AWAITING_GUIDANCE, COMPLETED, FAILED
        self.pause_reason: Optional[str] = None
        self.current_instruction: str = ""
        self.active_steps: List[Dict[str, Any]] = []
        self.current_step_idx: int = 0
        self.active_entity: Optional[str] = None
        self.suggested_actions: List[Dict[str, str]] = []

    def post_user_message(self, text: str) -> Dict[str, Any]:
        """Records an incoming user chat message."""
        msg = {
            "id": f"msg_user_{int(time.time() * 1000)}",
            "sender": "user",
            "text": text,
            "timestamp": time.strftime("%H:%M:%S"),
        }
        self.chat_history.append(msg)
        return msg

    def post_robot_message(
        self,
        text: str,
        steps: Optional[List[Dict[str, Any]]] = None,
        options: Optional[List[Dict[str, str]]] = None,
        status_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generates and records an explainable robot response."""
        msg = {
            "id": f"msg_robot_{int(time.time() * 1000)}",
            "sender": "robot",
            "text": text,
            "timestamp": time.strftime("%H:%M:%S"),
            "steps": steps or [],
            "options": options or [],
            "status_tag": status_tag or self.state,
        }
        self.chat_history.append(msg)
        if options:
            self.suggested_actions = options
        return msg

    def handle_pause_request(self, requested_by_user: bool = True, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Pauses active execution immediately and enters guidance mode.
        """
        if requested_by_user:
            self.state = "PAUSED_BY_USER"
            self.pause_reason = reason or "Operator requested pause."
            self.narrator.log(
                category="pause",
                title="Paused by Operator",
                explanation="Arm motion halted in safe hover. Awaiting your guidance.",
                severity="warning",
            )
            options = [
                {"id": "retry", "label": "🔄 Retry Current Step", "desc": "Re-attempt this step"},
                {"id": "skip", "label": "⏭️ Skip to Next Step", "desc": "Skip this block"},
                {"id": "lift_higher", "label": "⬆️ Lift Higher (+4cm)", "desc": "Increase transit clearance"},
                {"id": "resume", "label": "▶️ Resume As-Is", "desc": "Continue execution"},
                {"id": "home", "label": "🏠 Return to Home", "desc": "Safe abort"},
            ]
            msg_text = (
                f"⏸️ **I've paused at Step {self.current_step_idx + 1}** ({self.active_entity or 'active target'}). "
                "Where should I make an adjustment, or what would you like me to do?"
            )
        else:
            self.state = "PAUSED_BY_ANOMALY"
            self.pause_reason = reason or "H-JEPA latent anomaly detected."
            options = [
                {"id": "retry", "label": "🔄 Re-grasp from Table", "desc": "Re-align and grip"},
                {"id": "use_alt", "label": "🔀 Use Alternate Block", "desc": "Switch to next available cube"},
                {"id": "lift_higher", "label": "⬆️ Add Clearance", "desc": "Transit higher to clear obstacles"},
                {"id": "home", "label": "🏠 Abort to Home", "desc": "Safe abort"},
            ]
            msg_text = (
                f"⚠️ **I encountered an issue on Step {self.current_step_idx + 1}**: {reason}\n"
                "I've stopped safely in place to prevent dropping blocks. How should I proceed?"
            )

        robot_msg = self.post_robot_message(msg_text, options=options, status_tag=self.state)
        return {
            "state": self.state,
            "message": robot_msg,
            "pause_reason": self.pause_reason,
            "options": options,
        }

    def process_guidance(self, text: str, orchestrator: Any, sim: Any) -> Dict[str, Any]:
        """
        Interprets human guidance given in the chat while paused, applies adaptations
        to the active orchestrator/DAG, records the experience, and resumes.
        """
        if not text or not str(text).strip():
            return {"action": "NO_OP", "resumed": False, "reply": "Please specify what you would like me to do."}

        guidance = str(text).strip().lower()
        self.post_user_message(text)

        if orchestrator is None:
            reply_text = "No active task to guide."
            self.post_robot_message(reply_text, status_tag="IDLE")
            return {"action": "NO_TASK", "resumed": False, "reply": reply_text}

        adaptation: Dict[str, Any] = {}
        action_taken = ""

        # Check common guidance intents
        if any(w in guidance for w in ("retry", "try again", "re-grasp", "re-align", "attempt again")):
            action_taken = "RETRY_STEP"
            adaptation["action"] = "retry_subgoal"
            if orchestrator.active_subgoal:
                orchestrator.active_subgoal.status = "PENDING"
                orchestrator.active_subgoal.plan_steps = []
            reply_text = f"Got it. Re-attempting Step {self.current_step_idx + 1} from a fresh alignment."

        elif any(w in guidance for w in ("higher", "lift", "altitude", "clearance", "too low")):
            action_taken = "INCREASE_CLEARANCE"
            adaptation["lift_offset_y"] = 0.04
            # Patch active planner or controller hover
            reply_text = "Understood. Increasing transit altitude by +4cm to ensure clearance."

        elif any(w in guidance for w in ("skip", "next", "ignore")):
            action_taken = "SKIP_SUBGOAL"
            adaptation["action"] = "skip"
            if orchestrator.active_subgoal:
                orchestrator.planner.mark_completed(orchestrator.active_subgoal.id)
                orchestrator.active_subgoal = None
            reply_text = f"Skipping Step {self.current_step_idx + 1}. Moving to the next step."

        elif any(w in guidance for w in ("resume", "continue", "go ahead", "proceed", "carry on")):
            action_taken = "RESUME"
            reply_text = "Resuming execution..."

        elif any(w in guidance for w in ("home", "stop", "abort", "cancel", "reset")):
            action_taken = "ABORT"
            orchestrator.stop_task()
            self.state = "IDLE"
            self.suggested_actions = []
            reply_text = "Aborted task. Returning arm to safe home standby pose."
            self.post_robot_message(reply_text, status_tag="IDLE")
            return {"action": action_taken, "resumed": False, "reply": reply_text}

        else:
            # Check for block swapping (e.g. "use purple instead", "grab the blue one")
            color_map = {
                "purple": "block_purple",
                "violet": "block_purple",
                "yellow": "block_yellow",
                "cyan": "block_cyan",
                "blue": "block_cyan",
                "orange": "block_orange",
                "magenta": "block_magenta",
                "red": "block_magenta",
                "emerald": "block_emerald",
                "green": "block_emerald",
            }
            new_target = None
            for c_name, b_id in color_map.items():
                if c_name in guidance and b_id in sim.blocks:
                    new_target = b_id
                    break

            active_sg = orchestrator.active_subgoal or (orchestrator.planner.get_ready_subgoals()[0] if orchestrator.planner.get_ready_subgoals() else None)
            if new_target and active_sg:
                old_ent = active_sg.entity
                active_sg.entity = new_target
                active_sg.status = "PENDING"
                if hasattr(active_sg, "plan_steps"):
                    active_sg.plan_steps = []
                orchestrator.active_subgoal = active_sg
                action_taken = "SWAP_TARGET_ENTITY"
                adaptation["old_entity"] = old_ent
                adaptation["new_entity"] = new_target
                reply_text = f"Understood! Switching target from {old_ent.replace('block_', '')} to {new_target.replace('block_', '')}."
            else:
                action_taken = "GENERIC_RESUME"
                reply_text = f"Applying feedback: \"{text}\". Resuming execution."

        # Record this episode into Episodic Memory so the arm learns from this interaction
        if self.pause_reason:
            self.memory.record_episode(
                task_instruction=self.current_instruction,
                subgoal_id=orchestrator.active_subgoal.id if orchestrator.active_subgoal else "subgoal",
                failure_type=self.state,
                entity=self.active_entity,
                human_guidance=text,
                adaptation=adaptation,
                outcome="SUCCESS",
            )
            self.narrator.log(
                category="learned",
                title="Learned From Human Guidance",
                explanation=f"Stored feedback in episodic memory: \"{text}\" ({action_taken}).",
                severity="success",
            )

        # Transition back to EXECUTING
        self.state = "EXECUTING"
        self.pause_reason = None
        self.suggested_actions = []
        robot_msg = self.post_robot_message(reply_text, status_tag="EXECUTING")

        return {
            "action": action_taken,
            "resumed": True,
            "adaptation": adaptation,
            "reply": reply_text,
            "robot_message": robot_msg,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes dialogue state for WebSocket streaming."""
        return {
            "state": self.state,
            "pause_reason": self.pause_reason,
            "current_step_idx": self.current_step_idx,
            "total_steps": len(self.active_steps),
            "active_steps": self.active_steps,
            "suggested_actions": self.suggested_actions,
            "chat_history": self.chat_history[-20:], # last 20 messages for UI
            "memory_summary": self.memory.get_summary(),
        }
