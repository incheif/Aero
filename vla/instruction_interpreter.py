"""
vla/instruction_interpreter.py

Cognitive Instruction Interpreter & Feasibility Validator for Robot Arm VLA.
Pre-processes natural-language commands prior to robotic trajectory planning:
  1. Validates physical feasibility against workcell constraints (block count, rigid body dynamics, non-destructive gripper).
  2. Rejects impossible commands (e.g. tower >5 blocks, break/crush blocks, throw objects, fly).
  3. Recognizes communicative gestures (e.g. wave high, nod, celebrate, point).
  4. Disambiguates tower layer counts (1 to 5) and maps informal colors (e.g. blue->cyan, red->magenta, green->emerald).
  5. Supports optional LLM (Google Gemini) pre-call with robust deterministic offline fallback.
"""

import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple

# Physical constants for validation
MAX_PHYSICAL_BLOCKS = 5
MAX_LAYER_HEIGHT = 5
AVAILABLE_BLOCKS = {
    "block_cyan": {"name": "Cyan Cube", "color": "#00AEEF", "aliases": ["cyan", "blue", "light blue", "sky blue", "teal"]},
    "block_orange": {"name": "Orange Cube", "color": "#F7941E", "aliases": ["orange", "amber", "tangerine"]},
    "block_magenta": {"name": "Magenta Cube", "color": "#E11D8F", "aliases": ["magenta", "red", "pink", "crimson", "purple", "rose"]},
    "block_yellow": {"name": "Yellow Cube", "color": "#FACC15", "aliases": ["yellow", "gold", "lemon"]},
    "block_emerald": {"name": "Emerald Cube", "color": "#10B981", "aliases": ["emerald", "green", "dark green", "mint"]},
}

COLOR_ALIAS_MAP = {}
for b_id, info in AVAILABLE_BLOCKS.items():
    for alias in info["aliases"]:
        COLOR_ALIAS_MAP[alias] = b_id

# Common words for non-existent items
NON_EXISTENT_OBJECTS = [
    "cup", "coffee", "mug", "apple", "banana", "fruit", "glass", "bottle",
    "pen", "pencil", "knife", "fork", "spoon", "plate", "bowl", "book",
    "laptop", "phone", "sponge", "box", "can", "chair", "wrench", "tool",
    "hammer", "scissors", "paper", "cardboard"
]

class InstructionInterpreter:
    """Pre-processes, validates, and disambiguates natural-language instructions."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        self.model_name = model_name

    def set_api_key(self, api_key: str):
        self.api_key = (api_key or "").strip()

    def interpret(self, instruction: str) -> Dict[str, Any]:
        """
        Interprets the user instruction.
        Attempts LLM pre-call if API key is active; falls back to deterministic rule engine.
        Always returns a structured interpretation dictionary.
        """
        raw_cmd = instruction.strip()
        cmd = raw_cmd.lower()

        # If Google Gemini API is available and key is set, try LLM classification
        if self.api_key:
            try:
                llm_result = self._call_llm_interpreter(raw_cmd)
                if llm_result:
                    return llm_result
            except Exception as e:
                print(f"[InstructionInterpreter] LLM pre-call error: {e}. Falling back to rule engine.")

        # Fallback / Embedded Deterministic Rule Engine
        return self._rule_based_interpretation(raw_cmd)

    def _rule_based_interpretation(self, raw_cmd: str) -> Dict[str, Any]:
        """High-precision deterministic parser and validator."""
        cmd = raw_cmd.lower().strip()

        # -------------------------------------------------------------
        # 1. IMPOSSIBILITY CHECK: Destructive Actions
        # -------------------------------------------------------------
        destructive_words = [
            "break", "smash", "crush", "destroy", "cut", "slice", "melt",
            "burn", "tear", "shred", "split", "fracture", "dent", "squash",
            "flatten", "explode", "vaporize", "damage"
        ]
        for w in destructive_words:
            if re.search(rf"\b{w}\b", cmd):
                return {
                    "is_possible": False,
                    "status": "IMPOSSIBLE",
                    "action_type": "REJECTED",
                    "reason": (
                        f"❌ Not Possible: The user requested to '{w}' an object. "
                        "All tabletop blocks are rigid physical bodies. The robot arm is equipped with "
                        "a parallel-jaw positioning gripper intended for non-destructive pick-and-place manipulation, "
                        "and cannot break, crush, cut, or deform materials."
                    ),
                    "thought": f"Command rejected: Destructive action '{w}' is physically infeasible in this workcell."
                }

        # -------------------------------------------------------------
        # 2. IMPOSSIBILITY CHECK: Throwing & Ballistics
        # -------------------------------------------------------------
        throw_words = ["throw", "toss", "fling", "yeet", "launch", "catapult", "kick", "shoot"]
        for w in throw_words:
            if re.search(rf"\b{w}\b", cmd):
                return {
                    "is_possible": False,
                    "status": "IMPOSSIBLE",
                    "action_type": "REJECTED",
                    "reason": (
                        f"❌ Not Possible: Dynamic throwing ('{w}') is disabled for safety and kinematic stability. "
                        "The robot arm executes controlled, collision-free pick-and-place trajectories."
                    ),
                    "thought": f"Command rejected: Dynamic throwing ('{w}') violates workcell safety limits."
                }

        # -------------------------------------------------------------
        # 3. IMPOSSIBILITY CHECK: Locomotion & Flying
        # -------------------------------------------------------------
        locomotion_words = ["fly", "fly away", "airplane", "drive", "walk", "roll", "leave table", "leave room", "take off"]
        for w in locomotion_words:
            if re.search(rf"\b{w}\b", cmd):
                return {
                    "is_possible": False,
                    "status": "IMPOSSIBLE",
                    "action_type": "REJECTED",
                    "reason": (
                        f"❌ Not Possible: The robot arm is rigidly mounted to the workcell table pedestal. "
                        "It is an articulated stationary manipulator and cannot fly, drive, walk, or leave the table."
                    ),
                    "thought": f"Command rejected: Robot is stationary and cannot perform '{w}'."
                }

        # -------------------------------------------------------------
        # 4. IMPOSSIBILITY CHECK: Tower Height > 5 Blocks
        # -------------------------------------------------------------
        # Check explicit digit counts > 5
        number_matches = re.findall(r"\b(\d+)\s*(?:blocks?|layers?|cubes?|tiers?|high|levels?)\b", cmd)
        number_matches += re.findall(r"\b(?:tower|stack)\s*(?:of\s*)?(\d+)\b", cmd)
        word_number_map = {
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "twenty": 20, "hundred": 100
        }
        requested_count = None
        for nm in number_matches:
            try:
                requested_count = int(nm)
                break
            except ValueError:
                pass

        if requested_count is None:
            for word_num, val in word_number_map.items():
                if re.search(rf"\b{word_num}\s*(?:blocks?|layers?|cubes?|tiers?|high|levels?)\b", cmd) or re.search(rf"\b(?:tower|stack)\s*(?:of\s*)?{word_num}\b", cmd):
                    requested_count = val
                    break

        if requested_count is not None:
            if requested_count > MAX_PHYSICAL_BLOCKS:
                return {
                    "is_possible": False,
                    "status": "IMPOSSIBLE",
                    "action_type": "REJECTED",
                    "reason": (
                        f"❌ Not Possible: Requested {requested_count} blocks/layers, but the workcell only contains "
                        f"{MAX_PHYSICAL_BLOCKS} physical blocks (Cyan, Orange, Magenta, Yellow, Emerald). "
                        f"A tower of {requested_count} layers exceeds the physical blocks present and the robot's "
                        f"vertical reach limit (Y ≤ 1.12m)."
                    ),
                    "thought": f"Command rejected: {requested_count}-block tower exceeds available blocks ({MAX_PHYSICAL_BLOCKS}) and vertical kinematic boundary."
                }
            if requested_count <= 0:
                return {
                    "is_possible": False,
                    "status": "IMPOSSIBLE",
                    "action_type": "REJECTED",
                    "reason": "❌ Not Possible: Tower must consist of at least 1 layer.",
                    "thought": "Command rejected: Non-positive layer count requested."
                }

        # -------------------------------------------------------------
        # 5. IMPOSSIBILITY CHECK: Non-Existent Entities
        # -------------------------------------------------------------
        for bad_obj in NON_EXISTENT_OBJECTS:
            if re.search(rf"\b{bad_obj}\b", cmd) and not any(k in cmd for k in ("block", "cube", "pad", "table", "arm")):
                return {
                    "is_possible": False,
                    "status": "IMPOSSIBLE",
                    "action_type": "REJECTED",
                    "reason": (
                        f"❌ Not Possible: '{bad_obj}' is not present in the workcell. "
                        "The tabletop environment contains: Cyan Cube, Orange Cube, Magenta Cube, "
                        "Yellow Cube, Emerald Cube, and the Green Target Pad."
                    ),
                    "thought": f"Command rejected: Entity '{bad_obj}' not found in 3D perception memory."
                }

        # -------------------------------------------------------------
        # 6. GESTURES: Wave High, Wave, Nod, Point, Celebrate
        # -------------------------------------------------------------
        if any(p in cmd for p in ("wave high", "wave high up", "high wave", "reach up and wave")):
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "GESTURE",
                "gesture_name": "wave_high",
                "thought": "Executing 'Wave High' gesture: Raising end-effector to elevated altitude (Y=1.02m) and sweeping base yaw with synchronized gripper wiggles.",
            }

        if any(p in cmd for p in ("wave", "say hi", "greet", "hello", "hi there")):
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "GESTURE",
                "gesture_name": "wave_high",
                "thought": "Executing 'Wave Hello' gesture: Raising arm into view and oscillating yaw to greet the user.",
            }

        if any(p in cmd for p in ("nod", "agree", "say yes")):
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "GESTURE",
                "gesture_name": "nod",
                "thought": "Executing 'Nod' gesture: Articulating wrist pitch up and down rhythmically.",
            }

        if any(p in cmd for p in ("celebrate", "victory", "dance", "cheer")):
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "GESTURE",
                "gesture_name": "celebrate",
                "thought": "Executing 'Victory Celebration' gesture: Elevating arm and cycling gripper grasping.",
            }

        if any(p in cmd for p in ("point at pad", "point to pad", "point target", "point to green pad")):
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "GESTURE",
                "gesture_name": "point_pad",
                "thought": "Executing 'Point at Pad' gesture: Aiming wrist and TCP directly at target landing zone [0.48, 0.72, 0.22].",
            }

        if any(p in cmd for p in ("shake head", "say no", "disagree")):
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "GESTURE",
                "gesture_name": "shake_head",
                "thought": "Executing 'Shake Head' gesture: Articulating base yaw side to side.",
            }

        # -------------------------------------------------------------
        # 7. EXPLORATION
        # -------------------------------------------------------------
        if any(w in cmd for w in ("explore", "scan table", "survey", "search", "map table", "frontier")):
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "EXPLORE",
                "thought": "High-level workspace exploration requested. Initiating active multi-quadrant frontier scanning.",
            }

        # -------------------------------------------------------------
        # 8. CLEAR / RESET / REST
        # -------------------------------------------------------------
        if any(w in cmd for w in ("clear", "empty pad", "clear pad", "displace pad")):
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "CLEAR",
                "thought": "Clearing target pad. Transferring resting blocks back to table perimeter.",
            }

        if any(w in cmd for w in ("rest", "standby", "home pose", "park arm")):
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "REST",
                "thought": "Returning robot arm to home standby configuration.",
            }

        # -------------------------------------------------------------
        # 9. TOWER STACKING (Normalized Layer Count & Color Mapping)
        # -------------------------------------------------------------
        # Typo tolerance: "twer", "towr", "stack", "build", "layer"
        is_tower_request = any(w in cmd for w in ("tower", "twer", "towr", "stack", "build", "layer", "tier", "pad"))
        if is_tower_request or any(c in cmd for c in ("cyan", "orange", "magenta", "yellow", "emerald", "blue", "red", "green")):
            # Parse requested layer count
            layer_count = requested_count or 5

            # Word regex checks for 1 to 5
            if re.search(r"\b(1|one|single)\s*(layer|block|tier|level|cube)\b", cmd) or "1 layer" in cmd or "1-layer" in cmd or "one layer" in cmd:
                layer_count = 1
            elif re.search(r"\b(2|two|double)\s*(layer|block|tier|level|cube)\b", cmd) or "2 layer" in cmd or "2-layer" in cmd or "two layer" in cmd or "stack 2" in cmd:
                layer_count = 2
            elif re.search(r"\b(3|three|triple)\s*(layer|block|tier|level|cube)\b", cmd) or "3 layer" in cmd or "3-layer" in cmd or "three layer" in cmd or "stack 3" in cmd:
                layer_count = 3
            elif re.search(r"\b(4|four|quad)\s*(layer|block|tier|level|cube)\b", cmd) or "4 layer" in cmd or "4-layer" in cmd or "four layer" in cmd or "stack 4" in cmd:
                layer_count = 4
            elif re.search(r"\b(5|five|all)\s*(layer|block|tier|level|cube)\b", cmd) or "5 layer" in cmd or "5-layer" in cmd or "all 5" in cmd or "all blocks" in cmd:
                layer_count = 5

            # If user explicitly asked for single pick without stacking words
            if layer_count == 5 and not any(w in cmd for w in ("tower", "twer", "towr", "stack", "build", "all", "5")) and any(c in cmd for c in ("pick", "move", "place", "grab", "transfer")):
                layer_count = 1

            # Extract color sequence mentioned by user
            color_mentions = []
            notes = []
            
            # Words to search for in order of appearance
            known_colors = [
                ("cyan", "block_cyan", "Cyan Cube"),
                ("blue", "block_cyan", "Cyan Cube (mapped from 'blue')"),
                ("light blue", "block_cyan", "Cyan Cube"),
                ("orange", "block_orange", "Orange Cube"),
                ("magenta", "block_magenta", "Magenta Cube"),
                ("red", "block_magenta", "Magenta Cube (mapped from 'red' - nearest available color)"),
                ("pink", "block_magenta", "Magenta Cube"),
                ("purple", "block_magenta", "Magenta Cube"),
                ("yellow", "block_yellow", "Yellow Cube"),
                ("emerald", "block_emerald", "Emerald Cube"),
                ("green", "block_emerald", "Emerald Cube (mapped from 'green')"),
            ]

            found_items = []
            for alias, b_id, label in known_colors:
                # Avoid matching "green pad" as the green block
                if alias == "green" and any(p in cmd for p in ("green pad", "green target", "green zone", "green circle")):
                    continue
                # Find all occurrences of this alias as a whole word
                pattern = rf"\b{re.escape(alias)}\b"
                for match in re.finditer(pattern, cmd):
                    found_items.append((match.start(), b_id, alias, label))

            # Sort by order of appearance in instruction
            found_items.sort(key=lambda x: x[0])
            
            target_blocks = []
            for _, b_id, alias, label in found_items:
                if b_id not in target_blocks:
                    target_blocks.append(b_id)
                    if "mapped from" in label:
                        notes.append(label)

            # Fill in remaining blocks up to layer_count
            default_priority = ["block_cyan", "block_orange", "block_magenta", "block_yellow", "block_emerald"]
            for def_id in default_priority:
                if len(target_blocks) >= layer_count:
                    break
                if def_id not in target_blocks:
                    target_blocks.append(def_id)

            # Strict truncate to requested layer_count!
            target_blocks = target_blocks[:layer_count]

            notes_str = f" [Notes: {'; '.join(notes)}]" if notes else ""
            block_names = [AVAILABLE_BLOCKS[b]["name"] for b in target_blocks]
            thought = (
                f"Interpreted request for a {layer_count}-layer tower assembly on the target pad. "
                f"Sequence (base to top): {' -> '.join(block_names)}.{notes_str} "
                f"Remaining {MAX_PHYSICAL_BLOCKS - layer_count} block(s) will remain resting on the tabletop."
            )

            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "STACK_TOWER" if layer_count > 1 else "PICK_PLACE",
                "layer_count": layer_count,
                "target_blocks": target_blocks,
                "clarification_notes": notes,
                "thought": thought,
            }

        # Default fallback: General task
        return {
            "is_possible": True,
            "status": "OK",
            "action_type": "STACK_TOWER",
            "layer_count": 5,
            "target_blocks": ["block_cyan", "block_orange", "block_magenta", "block_yellow", "block_emerald"],
            "thought": f"Interpreted general manipulation instruction: '{raw_cmd}'. Planning full 5-block assembly.",
        }

    def _call_llm_interpreter(self, raw_cmd: str) -> Optional[Dict[str, Any]]:
        """Invokes Google Gemini with structured JSON prompt for cognitive classification."""
        try:
            import google.genai as genai
            client = genai.Client(api_key=self.api_key)
        except Exception:
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                client = legacy_genai.GenerativeModel(self.model_name)
            except Exception:
                return None

        prompt = f"""You are the cognitive natural-language instruction pre-processor for an autonomous 5-DOF table-mounted robot arm workcell.
Workcell Constraints:
- Available Blocks: Exactly 5 rigid cubes:
  1. Cyan Cube (color #00AEEF; aliases: cyan, blue)
  2. Orange Cube (color #F7941E; aliases: orange)
  3. Magenta Cube (color #E11D8F; aliases: magenta, red, pink, purple; note: there is NO pure red cube, magenta is the closest)
  4. Yellow Cube (color #FACC15; aliases: yellow)
  5. Emerald Cube (color #10B981; aliases: emerald, green)
- Available Target: 1 Green Target Pad at [0.48, 0.72, 0.22]
- Robot Physical Limits:
  - Stationary, rigidly table-mounted (CANNOT fly, drive, roll, leave the table)
  - Non-destructive parallel-jaw gripper (CANNOT break, crush, cut, melt, destroy blocks)
  - Controlled speed (CANNOT throw or launch objects)
  - Maximum 5 blocks can ever be stacked (towers > 5 blocks are physically IMPOSSIBLE)
- Gestures supported: "wave_high", "wave_hello", "nod", "celebrate", "point_pad", "shake_head".

User Instruction: "{raw_cmd}"

Determine:
1. Is this command physically possible and safe?
   - If user asks for tower > 5 blocks, break/destroy blocks, throw objects, fly, or items not on table -> "is_possible": false with a clear explanation in "reason".
   - If user asks for gestures ("wave high", "nod", "say hi", etc.) -> "is_possible": true, "action_type": "GESTURE", "gesture_name": "wave_high" (or appropriate gesture).
   - If user asks to build a tower with N layers (e.g. 3 layers with red, blue, yellow) -> "is_possible": true, "action_type": "STACK_TOWER", "layer_count": 3, "target_blocks": ["block_magenta", "block_cyan", "block_yellow"]. Map "blue"->block_cyan, "red"->block_magenta, "yellow"->block_yellow.

Respond with ONLY a JSON object with this format:
{{
  "is_possible": true or false,
  "status": "OK" or "IMPOSSIBLE",
  "reason": "Clear explanation if impossible",
  "action_type": "STACK_TOWER" or "GESTURE" or "EXPLORE" or "CLEAR" or "REST" or "REJECTED",
  "gesture_name": "wave_high" or null,
  "layer_count": 3 or null,
  "target_blocks": ["block_magenta", "block_cyan", "block_yellow"] or [],
  "thought": "Your cognitive reasoning explanation"
}}
"""
        response_text = ""
        try:
            if hasattr(client, "models"):
                resp = client.models.generate_content(model=self.model_name, contents=prompt)
                response_text = resp.text or ""
            else:
                resp = client.generate_content(prompt)
                response_text = resp.text or ""
        except Exception as e:
            print(f"[InstructionInterpreter] Gemini API error: {e}")
            return None

        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                # Validate critical fields
                if "is_possible" in parsed:
                    # Enforce strict layer limit even if LLM hallucinated
                    if parsed.get("layer_count", 0) and parsed["layer_count"] > MAX_PHYSICAL_BLOCKS:
                        parsed["is_possible"] = False
                        parsed["status"] = "IMPOSSIBLE"
                        parsed["reason"] = f"❌ Not Possible: Tower of {parsed['layer_count']} exceeds the 5 available physical blocks."
                    return parsed
            except Exception:
                pass
        return None
