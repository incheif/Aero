"""
vla/agents/decomposer.py

Instruction Decomposer Agent for Collaborative Robotics.
Decomposes complex high-level natural language instructions into:
  1. An executable Subgoal Dependency DAG for the motion planner.
  2. A plain-English, easy-to-follow step-by-step proposal for the human operator.
"""

from typing import Dict, List, Any, Optional
from vla.recursive_dag_planner import RecursiveDAGPlanner, Subgoal
from vla.instruction_interpreter import InstructionInterpreter


class InstructionDecomposerAgent:
    """
    Translates open-ended commands into structured subgoals and human-readable step explanations.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.interpreter = InstructionInterpreter(api_key=api_key, model_name=model_name)
        self.planner = RecursiveDAGPlanner()

    def set_api_key(self, api_key: str):
        self.interpreter.set_api_key(api_key)

    def decompose(self, instruction: str, sim: Any) -> Dict[str, Any]:
        """
        Decomposes an instruction into executable subgoals and plain-English step summaries.
        """
        if not instruction or not str(instruction).strip():
            return {
                "is_possible": False,
                "status": "EMPTY_INSTRUCTION",
                "reason": "Please provide a valid robotic instruction.",
                "subgoals": [],
                "easy_steps": [],
                "proposal_message": "Instruction was empty. Please provide a command.",
            }

        # Step 1: Pre-process with cognitive interpreter
        interp = self.interpreter.interpret(instruction)
        if not interp.get("is_possible", True) or interp.get("status") == "IMPOSSIBLE":
            return {
                "is_possible": False,
                "status": "IMPOSSIBLE",
                "reason": interp.get("reason", "This task violates physical workcell constraints."),
                "subgoals": [],
                "easy_steps": [],
                "proposal_message": f"I cannot do this: {interp.get('reason')}",
            }

        # Step 2: Compile Subgoal DAG
        if hasattr(sim, "get_snapshot"):
            blocks = sim.get_snapshot()["blocks"]
        elif hasattr(sim, "blocks"):
            blocks = list(sim.blocks.values())
        else:
            blocks = sim

        subgoals = self.planner.decompose(instruction, blocks)
        if not subgoals:
            return {
                "is_possible": False,
                "status": "FAILED_TO_DECOMPOSE",
                "reason": "Could not formulate safe physical subgoals for this workspace configuration.",
                "subgoals": [],
                "easy_steps": [],
                "proposal_message": "I couldn't find a valid collision-free way to arrange the blocks for this command.",
            }

        # Step 3: Format plain-English steps
        easy_steps: List[Dict[str, Any]] = []
        for idx, sg in enumerate(subgoals, start=1):
            ent_name = (sg.entity or "block").replace("block_", "").title()
            if sg.skill_type == "CLEAR_OBSTACLE":
                step_title = "Clear Workspace"
                step_desc = f"Move {ent_name} cube to temporary staging area to avoid collision."
            elif sg.skill_type == "DEPOSIT":
                coords_str = f"[{sg.target_pose[0]:.2f}, {sg.target_pose[2]:.2f}]" if sg.target_pose else "target slot"
                step_title = f"Place {ent_name}"
                step_desc = f"Pick up {ent_name} cube and place it at {coords_str}."
            else:
                step_title = sg.skill_type.replace("_", " ").title()
                step_desc = sg.description

            easy_steps.append({
                "step_number": idx,
                "subgoal_id": sg.id,
                "title": step_title,
                "description": step_desc,
                "entity": sg.entity,
                "skill_type": sg.skill_type,
            })

        # Step 4: Conversational summary message
        lines = [f"I understood: **\"{instruction}\"**. Here is my {len(easy_steps)}-step plan:"]
        for s in easy_steps:
            lines.append(f"**Step {s['step_number']}**: {s['description']}")
        lines.append("Starting step 1 now...")
        proposal_message = "\n".join(lines)

        return {
            "is_possible": True,
            "status": "OK",
            "action_type": interp.get("action_type", "TASK"),
            "subgoals": subgoals,
            "easy_steps": easy_steps,
            "proposal_message": proposal_message,
            "interpretation": interp,
        }
