"""
vla/agents/narrator.py

Narrative Logger Agent for Explainable Robotic Telemetry.
Translates low-level kinematics, contact forces, and H-JEPA world model residuals
into human-understandable, plain-English narrative logs.
"""

import time
from typing import Dict, List, Any, Optional


class NarrativeLoggerAgent:
    """
    Translates raw mechanical, sensory, and cognitive events into plain-English
    narratives so the operator can clearly understand the robot's thought process
    and physical status.
    """

    def __init__(self, max_history: int = 150):
        self.max_history = max_history
        self.narrative_history: List[Dict[str, Any]] = []
        self.pending_narratives: List[Dict[str, Any]] = []

    def log(
        self,
        category: str,
        title: str,
        explanation: str,
        icon: Optional[str] = None,
        severity: str = "info", # "info", "warning", "error", "success"
    ) -> Dict[str, Any]:
        """
        Creates an explainable narrative entry.
        """
        icon_map = {
            "planning": "🧠",
            "approach": "🎯",
            "grasp": "🤏",
            "transit": "✈️",
            "place": "📦",
            "release": "✨",
            "pause": "⏸️",
            "anomaly": "⚠️",
            "recovery": "🔧",
            "success": "🎉",
            "dialogue": "💬",
            "learned": "📚",
        }
        chosen_icon = icon or icon_map.get(category, "🤖")

        entry = {
            "id": f"nar_{int(time.time() * 1000)}_{len(self.narrative_history)}",
            "timestamp": time.strftime("%H:%M:%S"),
            "category": category,
            "severity": severity,
            "icon": chosen_icon,
            "title": title,
            "explanation": explanation,
        }

        self.narrative_history.append(entry)
        if len(self.narrative_history) > self.max_history:
            self.narrative_history.pop(0)

        self.pending_narratives.append(entry)
        if len(self.pending_narratives) > 40:
            self.pending_narratives.pop(0)

        return entry

    def flush_pending(self) -> List[Dict[str, Any]]:
        """Returns newly queued narratives since last flush and clears pending."""
        items = list(self.pending_narratives)
        self.pending_narratives.clear()
        return items

    def explain_anomaly(
        self,
        subgoal_desc: str,
        residual: float,
        entity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Explains an H-JEPA world model anomaly in simple terms."""
        ent_name = (entity or "target block").replace("block_", "").title()
        if residual > 6.0:
            msg = f"The {ent_name} cube experienced a severe physical disturbance or dropped from the gripper."
        elif residual > 4.5:
            msg = f"The {ent_name} cube slipped slightly out of alignment while moving."
        else:
            msg = f"Minor motion variance detected on {ent_name} (within recoverable limits)."

        return self.log(
            category="anomaly",
            title=f"Unexpected Physical Variance ({subgoal_desc})",
            explanation=msg,
            severity="warning" if residual <= 6.0 else "error",
        )

    def explain_subgoal_transition(
        self,
        subgoal_idx: int,
        total_subgoals: int,
        skill_type: str,
        entity: Optional[str],
        slot_desc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generates an intuitive narrative for each step in a task."""
        ent_name = (entity or "block").replace("block_", "").title()
        
        if skill_type == "CLEAR_OBSTACLE":
            title = f"Step {subgoal_idx}/{total_subgoals}: Clearing Obstacle"
            expl = f"Moving {ent_name} cube to the safe staging zone to clear space."
            cat = "approach"
        elif skill_type == "DEPOSIT":
            title = f"Step {subgoal_idx}/{total_subgoals}: Placing {ent_name} Cube"
            expl = f"Moving {ent_name} to its target location {f'({slot_desc})' if slot_desc else ''}."
            cat = "place"
        elif skill_type == "APPROACH":
            title = f"Step {subgoal_idx}/{total_subgoals}: Approaching {ent_name}"
            expl = f"Centering gripper jaws directly above {ent_name} cube."
            cat = "approach"
        elif skill_type == "GRASP":
            title = f"Step {subgoal_idx}/{total_subgoals}: Gripping {ent_name}"
            expl = f"Closing jaws with gentle force to hold {ent_name} securely."
            cat = "grasp"
        else:
            title = f"Step {subgoal_idx}/{total_subgoals}: {skill_type.replace('_', ' ').title()}"
            expl = f"Performing {skill_type.lower()} action on {ent_name}."
            cat = "transit"

        return self.log(
            category=cat,
            title=title,
            explanation=expl,
            severity="info",
        )

