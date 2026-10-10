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
MAX_DISTINCT_TOWERS = 5

AVAILABLE_BLOCKS = {
    "block_magenta": {"name": "Red Cube", "color": "#FF1E27", "aliases": ["red", "crimson", "ruby", "scarlet", "magenta"]},
    "block_cyan": {"name": "Blue Cube", "color": "#1E60FF", "aliases": ["blue", "cyan", "sky blue", "cobalt", "navy", "teal"]},
    "block_yellow": {"name": "Yellow Cube", "color": "#FFC300", "aliases": ["yellow", "gold", "amber", "lemon"]},
    "block_emerald": {"name": "Green Cube", "color": "#00B341", "aliases": ["green", "emerald", "lime", "mint"]},
    "block_orange": {"name": "Orange Cube", "color": "#FF5500", "aliases": ["orange", "amber", "tangerine"]},
}

COLOR_ALIAS_MAP = {}
for b_id, info in AVAILABLE_BLOCKS.items():
    for alias in info["aliases"]:
        COLOR_ALIAS_MAP[alias] = b_id

# Word-to-number dictionary for parsing natural language numbers
WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "hundred": 100,
    "single": 1, "double": 2, "triple": 3, "quad": 4, "pair": 2, "dual": 2,
}

def parse_num_token(val: Any) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s.isdigit():
        return int(s)
    if s in WORD_TO_NUM:
        return WORD_TO_NUM[s]
    return None

def parse_tower_distribution(cmd: str, num_towers: int = 2) -> List[int]:
    """
    Parses tower block counts for multi-tower instructions (e.g. '2 of 2 blocks each and one of 1 block').
    Returns a list of integer layer counts for each tower.
    """
    cmd = cmd.lower()
    # Check for composite 'X of Y' patterns (e.g. '2 of 2 blocks each and one of 1 block')
    comp = re.findall(r"(\d+|one|two|three|four|five)\s+of\s+(\d+|one|two|three|four|five)\s*(?:blocks?|cubes?|layers?)?", cmd)
    if comp:
        dist = []
        for count_t, size_t in comp:
            c = parse_num_token(count_t) or 1
            s = parse_num_token(size_t) or 1
            dist.extend([s] * c)
        if dist:
            return dist

    # Check for 'N blocks each'
    m_each = re.search(r"(\d+|one|two|three|four|five)\s*(?:blocks?|layers?|cubes?)?\s*(?:each|per tower)", cmd)
    if m_each:
        per = parse_num_token(m_each.group(1)) or 2
        return [per] * num_towers

    # Default distributions for N towers across 5 blocks
    if num_towers == 2:
        return [2, 2]
    elif num_towers == 3:
        return [2, 2, 1]
    elif num_towers == 4:
        return [2, 1, 1, 1]
    elif num_towers == 5:
        return [1, 1, 1, 1, 1]
    return [5]

# Common words for non-existent items
NON_EXISTENT_OBJECTS = [
    "cup", "coffee", "mug", "apple", "banana", "fruit", "glass", "bottle",
    "pen", "pencil", "knife", "fork", "spoon", "plate", "bowl", "book",
    "laptop", "phone", "sponge", "box", "can", "chair", "wrench", "tool",
    "hammer", "scissors", "paper", "cardboard", "ball", "sphere", "cylinder",
    "coin", "shoe", "cloth", "car", "keys", "wallet", "mirror"
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
        Interprets the user instruction with cognitive physical feasibility validation.
        1. Runs exhaustive physical affordance validation first. If the request violates workcell
           limits (e.g. >2 distinct towers, >5 layers, duplicate colors, destructive/ballistic actions),
           it immediately rejects the command and returns a clear, specific explanation.
        2. If physically possible and an API key is set, invokes Google Gemini for multimodal reasoning.
        3. Enforces feasibility guardrails on LLM output, falling back to the deterministic rule engine.
        """
        raw_cmd = instruction.strip()

        # Step 1: Pre-validation of physical feasibility and workcell affordances
        feasibility_rejection = self._check_physical_feasibility(raw_cmd)
        if feasibility_rejection is not None:
            return feasibility_rejection

        # Step 2: LLM Classification if configured
        if self.api_key:
            try:
                llm_result = self._call_llm_interpreter(raw_cmd)
                if llm_result:
                    # Enforce feasibility guardrails on LLM output
                    if llm_result.get("is_possible", True):
                        override = self._check_physical_feasibility(raw_cmd)
                        if override is not None:
                            return override
                    return llm_result
            except Exception as e:
                print(f"[InstructionInterpreter] LLM pre-call error: {e}. Falling back to rule engine.")

        # Step 3: Deterministic Rule Engine for valid manipulation tasks
        return self._rule_based_interpretation(raw_cmd)

    def _check_physical_feasibility(self, raw_cmd: str) -> Optional[Dict[str, Any]]:
        """
        Comprehensive cognitive affordance and physical feasibility validator.
        Evaluates the requested command against rigid-body workcell constraints:
          1. Multi-tower landing zone capacity (maximum 2 distinct towers).
          2. Tower height / layer count limits (1 to 5 layers, vertical reach Y <= 1.12m).
          3. Total block inventory capacity (maximum 5 blocks).
          4. Per-color uniqueness inventory (maximum 1 cube per color).
          5. Pyramid geometry constraints (3-block stepped or 5-block square only).
          6. Non-existent colors and objects.
          7. Destructive, ballistic, fluid, chemical, and locomotion actions.
          8. Out-of-workspace boundary violations.
        Returns a rejection dict if infeasible, or None if physically plausible.
        """
        cmd = raw_cmd.lower().strip()

        # -------------------------------------------------------------
        # A. MULTI-TOWER FEASIBILITY: Up to 5 Distinct Towers (Total Blocks <= 5)
        # -------------------------------------------------------------
        tower_qty_matches = re.findall(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|hundred)\s+"
            r"(?:distinct\s+|separate\s+|different\s+|independent\s+|individual\s+)?towers?\b",
            cmd
        )
        tower_qty_matches += re.findall(
            r"\b(?:build|make|create|assemble|stack|erect)\s+"
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"(?:distinct\s+|separate\s+|different\s+|independent\s+|individual\s+)?towers?\b",
            cmd
        )
        tower_qty_matches += re.findall(
            r"\btowers?:\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
            cmd
        )

        for match_val in tower_qty_matches:
            tower_count = parse_num_token(match_val)
            if tower_count is not None:
                if tower_count > MAX_DISTINCT_TOWERS:
                    return {
                        "is_possible": False,
                        "status": "IMPOSSIBLE",
                        "action_type": "REJECTED",
                        "reason": (
                            f"❌ Not Possible: Requested {tower_count} distinct towers, but the workcell only contains "
                            f"{MAX_PHYSICAL_BLOCKS} physical cubes in total. The robot arm can assemble at most {MAX_DISTINCT_TOWERS} distinct towers "
                            f"(each tower requiring at least 1 cube)."
                        ),
                        "thought": f"Command rejected: Requested {tower_count} distinct towers exceeds total physical block inventory ({MAX_PHYSICAL_BLOCKS})."
                    }
                elif tower_count <= 0:
                    return {
                        "is_possible": False,
                        "status": "IMPOSSIBLE",
                        "action_type": "REJECTED",
                        "reason": "❌ Not Possible: Tower count must be a positive number.",
                        "thought": "Command rejected: Non-positive tower count requested."
                    }
                else:
                    distribution = parse_tower_distribution(cmd, tower_count)
                    total_req_blocks = sum(distribution)
                    if total_req_blocks > MAX_PHYSICAL_BLOCKS:
                        return {
                            "is_possible": False,
                            "status": "IMPOSSIBLE",
                            "action_type": "REJECTED",
                            "reason": (
                                f"❌ Not Possible: Building {tower_count} distinct towers with requested breakdown ({' + '.join(str(d) for d in distribution)}) "
                                f"requires {total_req_blocks} blocks, but only {MAX_PHYSICAL_BLOCKS} physical blocks are available in the workcell."
                            ),
                            "thought": f"Command rejected: Multi-tower request requires {total_req_blocks} blocks, exceeding inventory ({MAX_PHYSICAL_BLOCKS})."
                        }

        # -------------------------------------------------------------
        # B. TOWER HEIGHT / LAYER COUNT FEASIBILITY: Maximum 5 Layers
        # -------------------------------------------------------------
        if not any(w in cmd for w in ("pyramid", "pyramind", "distinct tower", "distinct towers", "separate tower", "separate towers", "different tower", "different towers")):
            layer_matches = re.findall(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|hundred)\s*(?:blocks?|layers?|cubes?|tiers?|levels?)\s*(?:high|tall)?\b", cmd)
            layer_matches += re.findall(r"\b(?:tower|stack)\s*(?:of\s*)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", cmd)
            layer_matches += re.findall(r"\b(?:height|tall|high)\s*(?:of\s*)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", cmd)
            layer_matches += re.findall(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)[-\s](?:layer|tier|block|cube)\b", cmd)

            for lm in layer_matches:
                val = parse_num_token(lm)
                if val is not None:
                    if "tower" in lm and "layer" not in lm and "block" not in lm and "cube" not in lm and "tier" not in lm:
                        continue
                    if val > MAX_PHYSICAL_BLOCKS:
                        return {
                            "is_possible": False,
                            "status": "IMPOSSIBLE",
                            "action_type": "REJECTED",
                            "reason": (
                                f"❌ Not Possible: Requested a {val}-layer tower, but the workcell only contains "
                                f"{MAX_PHYSICAL_BLOCKS} physical blocks (Red, Blue, Yellow, Green, Orange). "
                                f"A tower of {val} layers also exceeds the robot's vertical reach limit (Y ≤ 1.12m)."
                            ),
                            "thought": f"Command rejected: Requested {val}-layer tower exceeds inventory ({MAX_PHYSICAL_BLOCKS}) and vertical kinematic limit."
                        }
                    elif val <= 0:
                        return {
                            "is_possible": False,
                            "status": "IMPOSSIBLE",
                            "action_type": "REJECTED",
                            "reason": "❌ Not Possible: Tower must consist of at least 1 layer.",
                            "thought": "Command rejected: Non-positive layer count requested."
                        }

        # -------------------------------------------------------------
        # C. TOTAL BLOCKS INVENTORY: Maximum 5 Blocks
        # -------------------------------------------------------------
        total_block_matches = re.findall(r"\b(?:use|stack|pick|place|move|with)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(?:cubes?|blocks?)\b", cmd)
        for tbm in total_block_matches:
            val = parse_num_token(tbm)
            if val is not None and val > MAX_PHYSICAL_BLOCKS:
                return {
                    "is_possible": False,
                    "status": "IMPOSSIBLE",
                    "action_type": "REJECTED",
                    "reason": (
                        f"❌ Not Possible: Requested {val} blocks, but the workcell only contains "
                        f"{MAX_PHYSICAL_BLOCKS} physical blocks in total (Red, Blue, Yellow, Green, Orange)."
                    ),
                    "thought": f"Command rejected: Requested {val} blocks exceeds workcell inventory ({MAX_PHYSICAL_BLOCKS})."
                }

        # -------------------------------------------------------------
        # D. DUPLICATE COLOR INVENTORY: Exactly 1 Cube per Color
        # -------------------------------------------------------------
        color_qty_pattern = r"\b(\d+|two|three|four|five|six)\s+(red|crimson|ruby|magenta|blue|cyan|yellow|green|emerald|orange)s?\s*(?:cubes?|blocks?)?\b"
        for match in re.finditer(color_qty_pattern, cmd):
            qty_str, color_str = match.groups()
            qty = parse_num_token(qty_str)
            if qty is not None and qty > 1:
                color_name = AVAILABLE_BLOCKS.get(COLOR_ALIAS_MAP.get(color_str, ""), {}).get("name", color_str.capitalize())
                return {
                    "is_possible": False,
                    "status": "IMPOSSIBLE",
                    "action_type": "REJECTED",
                    "reason": (
                        f"❌ Not Possible: Requested {qty} {color_name}s, but there is only ONE {color_name} "
                        f"in the workcell (1 Red, 1 Blue, 1 Yellow, 1 Green, 1 Orange)."
                    ),
                    "thought": f"Command rejected: Requested {qty} {color_name}s exceeds single-instance inventory."
                }

        # -------------------------------------------------------------
        # E. NON-EXISTENT COLORS
        # -------------------------------------------------------------
        invalid_colors = ["black", "white", "purple", "violet", "brown", "gray", "grey", "silver", "beige", "transparent", "clear"]
        for ic in invalid_colors:
            if re.search(rf"\b{ic}\b", cmd) and any(k in cmd for k in ("block", "cube", "tower", "stack", "pick", "place", "move", "grab")):
                return {
                    "is_possible": False,
                    "status": "IMPOSSIBLE",
                    "action_type": "REJECTED",
                    "reason": (
                        f"❌ Not Possible: Color '{ic}' does not exist in the workcell. "
                        f"The table contains only 5 cubes: Red (#FF1E27), Blue (#1E60FF), Yellow (#FFC300), Green (#00B341), and Orange (#FF5500)."
                    ),
                    "thought": f"Command rejected: Color '{ic}' not found in tabletop perception."
                }

        # -------------------------------------------------------------
        # F. PYRAMID FEASIBILITY: 3-Block Stepped or 5-Block Square Only
        # -------------------------------------------------------------
        if any(w in cmd for w in ("pyramid", "pyramind")):
            pyr_num_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)[-\s]*(?:blocks?|cubes?|base|layers?)", cmd)
            if pyr_num_match:
                pyr_val = parse_num_token(pyr_num_match.group(1))
                if pyr_val is not None and pyr_val not in (3, 5):
                    return {
                        "is_possible": False,
                        "status": "IMPOSSIBLE",
                        "action_type": "REJECTED",
                        "reason": (
                            f"❌ Not Possible: A pyramid can only be constructed as either a 3-block stepped pyramid "
                            f"(2 base cubes + 1 apex) or a 5-block square pyramid (4 base cubes in 2x2 + 1 apex). "
                            f"A {pyr_val}-block pyramid is not geometrically feasible with the available 5 cubes."
                        ),
                        "thought": f"Command rejected: {pyr_val}-block pyramid is geometrically infeasible."
                    }

        # -------------------------------------------------------------
        # G. INFEASIBLE VERBS (Destructive, Ballistic, Locomotion, Fluid, Tools, Surface)
        # -------------------------------------------------------------
        destructive_words = [
            "break", "smash", "crush", "destroy", "cut", "slice", "melt",
            "burn", "tear", "shred", "split", "fracture", "dent", "squash",
            "flatten", "explode", "vaporize", "damage", "bend", "twist", "deform", "compress"
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

        throw_words = ["throw", "toss", "fling", "yeet", "launch", "catapult", "kick", "shoot", "hurl", "lob"]
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

        locomotion_words = [
            "fly", "fly away", "airplane", "drive", "walk", "roll", "leave table",
            "leave room", "take off", "swim", "run", "relocate", "move base", "move table"
        ]
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

        fluid_words = ["pour", "spill", "fill", "drink", "eat", "cook", "bake", "boil", "freeze", "microwave", "feed", "sip"]
        for w in fluid_words:
            if re.search(rf"\b{w}\b", cmd):
                return {
                    "is_possible": False,
                    "status": "IMPOSSIBLE",
                    "action_type": "REJECTED",
                    "reason": (
                        f"❌ Not Possible: The action '{w}' involves fluids, chemistry, or consumption. "
                        "The robot arm is designed strictly for solid tabletop cube pick-and-place manipulation."
                    ),
                    "thought": f"Command rejected: Action '{w}' is outside physical workcell capabilities."
                }

        adhesion_words = ["glue", "weld", "tape", "paste", "stick together", "screw", "nail", "solder", "fasten"]
        for w in adhesion_words:
            if re.search(rf"\b{w}\b", cmd):
                return {
                    "is_possible": False,
                    "status": "IMPOSSIBLE",
                    "action_type": "REJECTED",
                    "reason": (
                        f"❌ Not Possible: The action '{w}' requires bonding agents or assembly tools. "
                        "Manipulated objects are passive rigid cubes held by gravitational friction."
                    ),
                    "thought": f"Command rejected: Adhesion / bonding action '{w}' is unsupported."
                }

        surface_words = ["paint", "draw", "write", "color", "sketch", "erase", "scrape", "carve"]
        for w in surface_words:
            if re.search(rf"\b{w}\b", cmd) and not any(k in cmd for k in ("colorseek", "color seek")):
                return {
                    "is_possible": False,
                    "status": "IMPOSSIBLE",
                    "action_type": "REJECTED",
                    "reason": (
                        f"❌ Not Possible: Surface alteration ('{w}') requires marking implements. "
                        "The robot arm is equipped only with a parallel-jaw positioning gripper."
                    ),
                    "thought": f"Command rejected: Surface alteration action '{w}' is unsupported."
                }

        # -------------------------------------------------------------
        # H. NON-EXISTENT OBJECTS
        # -------------------------------------------------------------
        for bad_obj in NON_EXISTENT_OBJECTS:
            if re.search(rf"\b{bad_obj}\b", cmd) and not any(k in cmd for k in ("block", "cube", "pad", "table", "arm")):
                return {
                    "is_possible": False,
                    "status": "IMPOSSIBLE",
                    "action_type": "REJECTED",
                    "reason": (
                        f"❌ Not Possible: '{bad_obj}' is not present in the workcell. "
                        "The tabletop environment contains: Red Cube, Blue Cube, Yellow Cube, "
                        "Green Cube, Orange Cube, and the landing pads."
                    ),
                    "thought": f"Command rejected: Entity '{bad_obj}' not found in 3D perception memory."
                }

        # -------------------------------------------------------------
        # I. OUT-OF-WORKSPACE BOUNDARIES
        # -------------------------------------------------------------
        oob_words = ["floor", "drop on floor", "throw off", "outside table", "off table", "off the table", "ceiling", "wall"]
        for w in oob_words:
            if re.search(rf"\b{w}\b", cmd):
                return {
                    "is_possible": False,
                    "status": "IMPOSSIBLE",
                    "action_type": "REJECTED",
                    "reason": (
                        f"❌ Not Possible: Target placement '{w}' is outside the reachable tabletop manipulation envelope. "
                        "All manipulations must remain within the table boundary [X: -0.55m..0.75m, Z: -0.50m..0.50m]."
                    ),
                    "thought": f"Command rejected: Target '{w}' is outside physical workspace boundary."
                }

        # Feasibility checks passed
        return None

    def _rule_based_interpretation(self, raw_cmd: str) -> Dict[str, Any]:
        """High-precision deterministic parser for validated feasible instructions."""
        cmd = raw_cmd.lower().strip()

        # -------------------------------------------------------------
        # 1. GESTURES: Wave High, Wave, Nod, Point, Celebrate
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

        if any(p in cmd for p in ("nod", "agree", "say yes", "annuis")):
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "GESTURE",
                "gesture_name": "nod",
                "thought": "Executing 'Nod' gesture: Articulating wrist pitch up and down rhythmically.",
            }

        if any(p in cmd for p in ("snap", "pinch", "clap", "click gripper", "open and close", "apri e chiudi")):
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "GESTURE",
                "gesture_name": "snap",
                "thought": "Executing 'Snap' gesture: Rapid rhythmic gripper snapping and pinching.",
            }

        if any(p in cmd for p in ("celebrate", "victory", "dance", "cheer")):
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "GESTURE",
                "gesture_name": "celebrate",
                "thought": "Executing 'Victory Celebration' gesture: Elevating arm and cycling gripper grasping.",
            }

        # Dynamic pointing gesture
        point_match = re.search(r"(?:point|aim|punta)\s+(?:at|to|towards|il|al)?\s+(?:the\s+)?(cyan|orange|magenta|yellow|emerald|blue|red|green|pad|target)\b", cmd)
        if point_match:
            target_term = point_match.group(1).lower()
            if target_term in ("pad", "target"):
                return {
                    "is_possible": True,
                    "status": "OK",
                    "action_type": "GESTURE",
                    "gesture_name": "point_pad",
                    "thought": "Executing 'Point at Pad' gesture: Aiming wrist and TCP directly at green landing pad.",
                }
            mapped_color = COLOR_ALIAS_MAP.get(target_term, target_term)
            clean_name = AVAILABLE_BLOCKS.get(mapped_color, {}).get("name", target_term.capitalize())
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "GESTURE",
                "gesture_name": "point",
                "target_name": target_term,
                "thought": f"Executing dynamic pointing gesture targeting {clean_name}.",
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
        # 6B. COLORSEEK: In-Tab Closed-Loop Visual Servoing
        # -------------------------------------------------------------
        if any(w in cmd for w in ("colorseek", "color seek", "visual servo", "visual servoing", "vision policy", "pixel policy", "blob tracker")):
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "COLORSEEK",
                "thought": "Executing ColorSeek closed-loop visual servoing policy (RGB pixel centroid tracking at 5 Hz).",
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
        # 8B. MULTI-TOWERS ASSEMBLY (2 to 5 Distinct Towers)
        # -------------------------------------------------------------
        is_multi_towers = (
            any(p in cmd for p in (
                "2 towers", "two towers", "dual towers", "dual tower",
                "two distinct towers", "2 distinct towers",
                "two separate towers", "2 separate towers",
                "two different towers", "2 different towers",
                "pair of towers",
                "3 towers", "three towers", "triple towers",
                "three distinct towers", "3 distinct towers",
                "three separate towers", "3 separate towers",
                "4 towers", "four towers", "4 distinct towers",
                "5 towers", "five towers", "5 distinct towers",
            ))
            or bool(re.search(r"\b(2|two|3|three|4|four|5|five)\s+(?:distinct\s+|separate\s+|different\s+)?towers?\b", cmd))
        )
        if is_multi_towers:
            m_towers = re.search(r"\b(2|two|3|three|4|four|5|five)\s+(?:distinct\s+|separate\s+|different\s+)?towers?\b", cmd)
            num_towers = parse_num_token(m_towers.group(1)) if m_towers else 2
            distribution = parse_tower_distribution(cmd, num_towers)

            known_colors = [
                ("red", "block_magenta", "Red Cube"),
                ("blue", "block_cyan", "Blue Cube"),
                ("yellow", "block_yellow", "Yellow Cube"),
                ("green", "block_emerald", "Green Cube"),
                ("orange", "block_orange", "Orange Cube"),
                ("cyan", "block_cyan", "Blue Cube"),
                ("magenta", "block_magenta", "Red Cube"),
                ("emerald", "block_emerald", "Green Cube"),
            ]
            found_colors = []
            for alias, b_id, label in known_colors:
                for match in re.finditer(rf"\b{re.escape(alias)}\b", cmd):
                    found_colors.append((match.start(), b_id, label))
            found_colors.sort(key=lambda x: x[0])

            mentioned_b_ids = []
            for _, b_id, _ in found_colors:
                if b_id not in mentioned_b_ids:
                    mentioned_b_ids.append(b_id)

            all_blocks = ["block_cyan", "block_orange", "block_magenta", "block_yellow", "block_emerald"]
            pool = list(mentioned_b_ids)
            for b in all_blocks:
                if b not in pool:
                    pool.append(b)

            tower_blocks_list = []
            curr_idx = 0
            for count in distribution:
                t_blocks = pool[curr_idx:curr_idx + count]
                tower_blocks_list.append(t_blocks)
                curr_idx += count

            desc_parts = []
            for i, tb in enumerate(tower_blocks_list):
                names = [AVAILABLE_BLOCKS.get(b, {}).get("name", b) for b in tb]
                desc_parts.append(f"Tower {i+1} ({len(tb)} blocks: [{', '.join(names)}])")

            action = "STACK_DUAL_TOWERS" if len(tower_blocks_list) == 2 else "STACK_MULTI_TOWERS"
            return {
                "is_possible": True,
                "status": "OK",
                "action_type": action,
                "num_towers": len(tower_blocks_list),
                "distribution": distribution,
                "tower_blocks_list": tower_blocks_list,
                "tower_a_blocks": tower_blocks_list[0] if len(tower_blocks_list) > 0 else [],
                "tower_b_blocks": tower_blocks_list[1] if len(tower_blocks_list) > 1 else [],
                "thought": (
                    f"Interpreted request for {len(tower_blocks_list)} distinct towers ({' + '.join(str(d) for d in distribution)} blocks). "
                    f"Spatial layout: {'; '.join(desc_parts)}. "
                    f"Kinematic paths will maintain elevated collision clearance over previously assembled towers."
                ),
            }

        # -------------------------------------------------------------
        # 8C. PYRAMID ASSEMBLY (Multi-Contact Stepped / Square Pyramid)
        # -------------------------------------------------------------
        if any(w in cmd for w in ("pyramid", "pyramind", "stepped tower", "triangle tower", "triangular tower", "pyramidal")):
            is_5_block = any(w in cmd for w in ("5 block", "5-block", "5 cube", "5-cube", "five block", "five cube", "all 5", "square pyramid", "4 base"))
            pyramid_type = "5_block_square" if is_5_block else "3_block_stepped"

            known_colors = [
                ("red", "block_magenta", "Red Cube"),
                ("blue", "block_cyan", "Blue Cube"),
                ("yellow", "block_yellow", "Yellow Cube"),
                ("green", "block_emerald", "Green Cube"),
                ("orange", "block_orange", "Orange Cube"),
                ("cyan", "block_cyan", "Blue Cube"),
                ("magenta", "block_magenta", "Red Cube"),
                ("emerald", "block_emerald", "Green Cube"),
            ]
            found_colors = []
            for alias, b_id, label in known_colors:
                for match in re.finditer(rf"\b{re.escape(alias)}\b", cmd):
                    found_colors.append((match.start(), b_id, label))
            found_colors.sort(key=lambda x: x[0])

            mentioned_b_ids = []
            for _, b_id, _ in found_colors:
                if b_id not in mentioned_b_ids:
                    mentioned_b_ids.append(b_id)

            all_blocks = ["block_cyan", "block_orange", "block_magenta", "block_yellow", "block_emerald"]
            pool = list(mentioned_b_ids)
            for b in all_blocks:
                if b not in pool:
                    pool.append(b)

            needed_count = 5 if pyramid_type == "5_block_square" else 3
            pyramid_blocks = pool[:needed_count]
            block_names = [AVAILABLE_BLOCKS.get(b, {}).get("name", b) for b in pyramid_blocks]

            if pyramid_type == "5_block_square":
                thought = (
                    f"Interpreted request for a 5-block 3D square pyramid. "
                    f"Base layer (2x2 grid): [{', '.join(block_names[:4])}]. "
                    f"Apex cap: [{block_names[4]}]. "
                    "Applying zero-impulse vertical descent to preserve base contact stability."
                )
            else:
                unused_names = [AVAILABLE_BLOCKS.get(b, {}).get("name", b) for b in pool[3:]]
                thought = (
                    f"Interpreted request for a 3-block stepped pyramid. "
                    f"Base layer (touching pair): [{block_names[0]}, {block_names[1]}]. "
                    f"Apex cap: [{block_names[2]}]. "
                    f"Remaining blocks ({', '.join(unused_names)}) remain resting on table."
                )

            return {
                "is_possible": True,
                "status": "OK",
                "action_type": "BUILD_PYRAMID",
                "pyramid_type": pyramid_type,
                "target_blocks": pyramid_blocks,
                "thought": thought,
            }

        # -------------------------------------------------------------
        # 9. TOWER STACKING (Normalized Layer Count & Color Mapping)
        # -------------------------------------------------------------
        # Typo tolerance: "twer", "towr", "stack", "build", "layer"
        is_tower_request = any(w in cmd for w in ("tower", "twer", "towr", "stack", "build", "layer", "tier", "pad"))
        if is_tower_request or any(c in cmd for c in ("cyan", "orange", "magenta", "yellow", "emerald", "blue", "red", "green")):
            # Parse requested layer count (1 to 5)
            layer_count = 5
            for lm in re.findall(r"\b(\d+|one|two|three|four|five)[-\s]*(?:blocks?|layers?|cubes?|tiers?|levels?)?\b", cmd):
                parsed_val = parse_num_token(lm)
                if parsed_val is not None and 1 <= parsed_val <= MAX_PHYSICAL_BLOCKS:
                    layer_count = parsed_val
                    break

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
            
            known_colors = [
                ("red", "block_magenta", "Red Cube"),
                ("crimson", "block_magenta", "Red Cube"),
                ("ruby", "block_magenta", "Red Cube"),
                ("magenta", "block_magenta", "Red Cube"),
                ("blue", "block_cyan", "Blue Cube"),
                ("cyan", "block_cyan", "Blue Cube"),
                ("cobalt", "block_cyan", "Blue Cube"),
                ("light blue", "block_cyan", "Blue Cube"),
                ("yellow", "block_yellow", "Yellow Cube"),
                ("gold", "block_yellow", "Yellow Cube"),
                ("green", "block_emerald", "Green Cube"),
                ("emerald", "block_emerald", "Green Cube"),
                ("orange", "block_orange", "Orange Cube"),
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

        # Check for presence of valid manipulation or object keywords before defaulting
        has_manipulation_keywords = any(w in cmd for w in (
            "stack", "tower", "cube", "block", "pad", "pyramid", "pick", "place", "move",
            "grab", "lift", "build", "assemble", "clear", "wave", "nod", "point", "scan",
            "explore", "colorseek", "servo", "rest", "home", "hello", "hi", "all",
            "red", "blue", "yellow", "green", "orange", "cyan", "magenta", "emerald"
        ))
        if not has_manipulation_keywords:
            return {
                "is_possible": False,
                "status": "IMPOSSIBLE",
                "action_type": "REJECTED",
                "reason": (
                    f"❌ Unrecognized Command: Could not extract an actionable manipulation task or gesture from '{raw_cmd}'. "
                    "Please specify an action such as stacking a tower (e.g. 'Make a 3-layer tower of red, blue and yellow'), "
                    "building 2 distinct towers, assembling a pyramid, or executing a gesture."
                ),
                "thought": f"Command rejected: Instruction '{raw_cmd}' does not contain recognized robotic manipulation keywords."
            }

        # Default fallback: General 5-block tower assembly when general stacking keywords were provided
        return {
            "is_possible": True,
            "status": "OK",
            "action_type": "STACK_TOWER",
            "layer_count": 5,
            "target_blocks": ["block_cyan", "block_orange", "block_magenta", "block_yellow", "block_emerald"],
            "thought": f"Interpreted general manipulation instruction: '{raw_cmd}'. Planning full 5-block assembly on target pad.",
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
- Available Blocks: Exactly 5 rigid cubes (1 of each color):
  1. Red Cube (#FF1E27; aliases: red, crimson, ruby, magenta)
  2. Blue Cube (#1E60FF; aliases: blue, cyan, cobalt)
  3. Yellow Cube (#FFC300; aliases: yellow, gold)
  4. Green Cube (#00B341; aliases: green, emerald)
  5. Orange Cube (#FF5500; aliases: orange)
- Available Landing Zones:
  - 1 Central Target Pad (for 1 single tower or pyramid)
  - 2 Dedicated Isolated Landing Zones: Pad A (North) and Pad B (South) (allows AT MOST 2 distinct towers)
  - Towers > 2 distinct towers are physically IMPOSSIBLE.
- Robot Physical Limits:
  - Stationary, rigidly table-mounted (CANNOT fly, drive, roll, leave the table)
  - Non-destructive parallel-jaw gripper (CANNOT break, crush, cut, melt, destroy blocks)
  - Controlled speed (CANNOT throw or launch objects)
  - Maximum 5 blocks total (towers > 5 layers, or requests for > 5 blocks, are physically IMPOSSIBLE)
  - Maximum 1 cube per color (requests for 2 red cubes, etc. are physically IMPOSSIBLE)
- Gestures supported: "wave_high", "wave_hello", "nod", "celebrate", "point_pad", "shake_head".

User Instruction: "{raw_cmd}"

Determine:
1. Is this command physically possible and safe?
   - If user asks for > 2 distinct towers (e.g. 3, 4, 6 distinct towers), tower > 5 layers, duplicate colors, break/destroy blocks, throw objects, fly, or items not on table -> "is_possible": false with a clear explanation in "reason".
   - If user asks for gestures ("wave high", "nod", "say hi", etc.) -> "is_possible": true, "action_type": "GESTURE", "gesture_name": "wave_high" (or appropriate gesture).
   - If user asks for 2 distinct towers -> "is_possible": true, "action_type": "STACK_DUAL_TOWERS".
   - If user asks for pyramid -> "is_possible": true, "action_type": "BUILD_PYRAMID".
   - If user asks to build a tower with N layers (e.g. 3 layers with red, blue, yellow) -> "is_possible": true, "action_type": "STACK_TOWER", "layer_count": 3, "target_blocks": ["block_magenta", "block_cyan", "block_yellow"].

Respond with ONLY a JSON object with this format:
{{
  "is_possible": true or false,
  "status": "OK" or "IMPOSSIBLE",
  "reason": "Clear explanation if impossible",
  "action_type": "STACK_TOWER" or "STACK_DUAL_TOWERS" or "BUILD_PYRAMID" or "GESTURE" or "EXPLORE" or "CLEAR" or "REST" or "REJECTED",
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
                if "is_possible" in parsed:
                    # Run deterministic physical feasibility validation as hard guardrail override
                    override = self._check_physical_feasibility(raw_cmd)
                    if override is not None:
                        return override
                    return parsed
            except Exception:
                pass
        return None
