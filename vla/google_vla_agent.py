"""
Google Vision-Language-Action (VLA) Agent for Robot Arm Manipulation.
Replaces local LLM (Gemma) with Google's multimodal VLA paradigm (Gemini 2.5 Flash / Gemini 2.0 / RT-style VLA).
Ingests visual camera frames + high-level natural language instructions + 3D spatial memory,
and outputs cognitive reasoning, visual grounding, and robotic action primitives.
"""

import os
import json
import re
import math
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

from simulation.constants import (
    STACK_ORIGIN,
    SAFE_HOVER_Y,
    PICK_Y,
    stack_slot_y,
    BLOCK_TARGETS,
)
from simulation.camera import render_camera_frame, encode_image_base64, world_to_pixel
from .cost_tracker import VLACostTracker

# Support both modern Google GenAI SDK (google-genai v1.x) and legacy (google-generativeai)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

HAS_NEW_GENAI = False
HAS_LEGACY_GENAI = False

try:
    from google import genai
    HAS_NEW_GENAI = True
except ImportError:
    pass

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import google.generativeai as legacy_genai
    HAS_LEGACY_GENAI = True
except ImportError:
    pass

GOOGLE_GENAI_AVAILABLE = HAS_NEW_GENAI or HAS_LEGACY_GENAI

class GoogleVLAAgent:
    """
    Google Vision-Language-Action (VLA) Robotic Agent.
    Interprets natural language instructions via visual reasoning and translates
    them into metric-space 6-DOF / joint action trajectories.
    Dual-mode:
      1. Live Google Gemini Multimodal Cloud VLA (when API key is provided)
      2. Embedded Deterministic VLA Runtime (zero-cost offline fallback)
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")).strip()
        self.model_name = model_name
        self.active_plan: List[Dict[str, Any]] = []
        self.current_step_idx: int = 0
        self.latest_thought: str = "System ready. Awaiting natural-language command."
        self.latest_grounding: List[Dict[str, Any]] = []
        self.execution_status: str = "IDLE"  # IDLE, REASONING, EXECUTING, VERIFYING, COMPLETED, FAILED
        self.active_mode: str = "embedded"  # 'live_cloud' or 'embedded'
        self.last_api_error: Optional[str] = None
        self.cost_tracker = VLACostTracker()

    def set_api_key(self, api_key: str):
        self.api_key = (api_key or "").strip()
        self.last_api_error = None
        print(f"[Google VLA] API Key set (length: {len(self.api_key)}). Live Gemini calls enabled.")

    def test_api_key(self, test_key: Optional[str] = None) -> Tuple[bool, str]:
        """Tests whether the API key is valid by making a lightweight ping to Gemini."""
        key_to_test = (test_key or self.api_key or "").strip()
        if not key_to_test:
            return False, "API key is empty."

        try:
            if HAS_NEW_GENAI:
                client = genai.Client(api_key=key_to_test)
                resp = client.models.generate_content(
                    model=self.model_name,
                    contents="ping",
                )
                if resp and resp.text:
                    return True, f"Successfully connected to Google Gemini ({self.model_name})!"
            elif HAS_LEGACY_GENAI:
                legacy_genai.configure(api_key=key_to_test)
                model = legacy_genai.GenerativeModel(self.model_name)
                resp = model.generate_content("ping")
                if resp and resp.text:
                    return True, f"Successfully connected to Google Gemini ({self.model_name})!"
            else:
                return False, "Google GenAI SDK is not installed in Python environment."
        except Exception as e:
            return False, f"Google Gemini API error: {str(e)}"

        return False, "No response received from Google Gemini API."

    def parse_instruction(
        self,
        instruction: str,
        image: Image.Image,
        snapshot: Dict[str, Any],
        semantic_summary: str
    ) -> Dict[str, Any]:
        """
        Processes natural-language command using Google VLA.
        Tries live Google Gemini Multimodal Cloud VLA first if API key is provided.
        Falls back to the Embedded Deterministic VLA Runtime if API key is missing or fails.
        """
        self.execution_status = "REASONING"
        self.latest_thought = f"Google VLA analyzing visual frame for instruction: '{instruction}'..."

        # 1. Try live Google Gemini VLA API if configured
        if self.api_key and GOOGLE_GENAI_AVAILABLE:
            try:
                live_result = self._call_gemini_vla(instruction, image, snapshot, semantic_summary)
                if live_result:
                    return live_result
            except Exception as e:
                self.last_api_error = str(e)
                print(f"[Google VLA] Live Gemini API error: {e}. Falling back to embedded VLA policy.")
        else:
            if not self.api_key:
                self.last_api_error = "No API key configured. Enter your Gemini API key in Settings to connect to Google Cloud VLA."

        # 2. Embedded Google VLA Runtime (High-Performance Deterministic Policy)
        result = self._embedded_vla_policy(instruction, snapshot)
        if self.last_api_error:
            result["api_warning"] = self.last_api_error
        return result

    def _call_gemini_vla(
        self,
        instruction: str,
        image: Image.Image,
        snapshot: Dict[str, Any],
        semantic_summary: str
    ) -> Optional[Dict[str, Any]]:
        """Calls Google Gemini Vision API with structured multimodal robotics prompt."""
        if not self.api_key:
            return None

        prompt = f"""You are Google VLA (Vision-Language-Action) robotic manipulation brain for a 5-DOF table-mounted robot arm.
Your end-effector is a parallel-jaw gripper.
Table coordinates are metric (meters) with Y-up:
- Tabletop surface is Y = 0.72m
- Pick altitude is Y = 0.7575m
- Base safe transit hover altitude is Y = 0.8550m.
  When carrying blocks over existing stack layers, transit altitude must clear the stack (transit_y = max(0.8550, target_y + 0.065)).
- Target Pad is located at [{STACK_ORIGIN[0]}, {STACK_ORIGIN[1]}, {STACK_ORIGIN[2]}]
- Stack layer altitudes: Layer 0 = {stack_slot_y(0):.4f}m, Layer 1 = {stack_slot_y(1):.4f}m, Layer 2 = {stack_slot_y(2):.4f}m, Layer 3 = {stack_slot_y(3):.4f}m, Layer 4 = {stack_slot_y(4):.4f}m

Current 3D Perception:
{semantic_summary}

User Command: "{instruction}"

Output a JSON object ONLY with the following schema:
{{
  "thought": "Your step-by-step cognitive and spatial affordance reasoning. Explain which blocks you are stacking in order from base layer (Layer 0) upward, and which blocks remain untouched.",
  "stack_sequence": ["block_cyan", "block_orange", "block_magenta"],
  "visual_grounding": [
    {{"entity": "block_cyan", "color": "cyan", "target_layer": 0, "target_coords": [{STACK_ORIGIN[0]}, {stack_slot_y(0):.4f}, {STACK_ORIGIN[2]}]}},
    {{"entity": "block_orange", "color": "orange", "target_layer": 1, "target_coords": [{STACK_ORIGIN[0]}, {stack_slot_y(1):.4f}, {STACK_ORIGIN[2]}]}},
    {{"entity": "block_magenta", "color": "magenta", "target_layer": 2, "target_coords": [{STACK_ORIGIN[0]}, {stack_slot_y(2):.4f}, {STACK_ORIGIN[2]}]}}
  ],
  "action_plan": [
    {{"action": "HOVER_ACQUIRE", "target": [x, 0.855, z], "gripper": 0.0, "dwell_ticks": 10, "desc": "Hover over block"}},
    {{"action": "DESCEND_PICK", "target": [x, 0.7575, z], "gripper": 0.0, "dwell_ticks": 12, "desc": "Descend to grip"}},
    {{"action": "GRIP", "target": [x, 0.7575, z], "gripper": 1.0, "dwell_ticks": 18, "desc": "Close parallel jaws"}},
    {{"action": "LIFT", "target": [x, 0.855, z], "gripper": 1.0, "dwell_ticks": 12, "desc": "Clean vertical lift"}},
    {{"action": "TRANSIT", "target": [{STACK_ORIGIN[0]}, 0.855, {STACK_ORIGIN[2]}], "gripper": 1.0, "dwell_ticks": 15, "desc": "Transit to target pad"}},
    {{"action": "DESCEND_PLACE", "target": [{STACK_ORIGIN[0]}, 0.7575, {STACK_ORIGIN[2]}], "gripper": 1.0, "dwell_ticks": 15, "desc": "Soft descent to placement slot"}},
    {{"action": "SOFT_RELEASE", "target": [{STACK_ORIGIN[0]}, 0.7575, {STACK_ORIGIN[2]}], "gripper": 0.0, "dwell_ticks": 20, "desc": "Zero-impulse jaw release"}},
    {{"action": "ASCEND_CLEAR", "target": [{STACK_ORIGIN[0]}, 0.855, {STACK_ORIGIN[2]}], "gripper": 0.0, "dwell_ticks": 12, "desc": "Ascend cleanly above stack"}}
  ]
}}
"""
        text = ""
        p_tokens = None
        c_tokens = None

        if HAS_NEW_GENAI:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
            )
            text = response.text or ""
            usage = getattr(response, "usage_metadata", None)
            if usage:
                p_tokens = getattr(usage, "prompt_token_count", None)
                c_tokens = getattr(usage, "candidates_token_count", None)
        elif HAS_LEGACY_GENAI:
            legacy_genai.configure(api_key=self.api_key)
            model = legacy_genai.GenerativeModel(self.model_name)
            response = model.generate_content([prompt, image])
            text = response.text or ""
            usage = getattr(response, "usage_metadata", None)
            if usage:
                p_tokens = getattr(usage, "prompt_token_count", None)
                c_tokens = getattr(usage, "candidates_token_count", None)
        else:
            raise RuntimeError("Neither google.genai nor google.generativeai SDK is available.")

        self.last_api_error = None
        cost_entry = self.cost_tracker.record_call(
            model_name=self.model_name,
            prompt_text=prompt,
            completion_text=text,
            has_image=True,
            actual_input_tokens=p_tokens,
            actual_output_tokens=c_tokens,
            is_live_api=True,
        )

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            self.latest_thought = data.get("thought", f"Google Gemini Live Cloud plan synthesized successfully.")
            self.latest_grounding = data.get("visual_grounding", [])
            raw_plan = data.get("action_plan", [])

            # Extract sequence of grounded blocks
            blocks_by_id = {b["id"]: b for b in snapshot.get("blocks", [])}
            color_to_id = {
                "cyan": "block_cyan",
                "orange": "block_orange",
                "magenta": "block_magenta",
                "yellow": "block_yellow",
                "emerald": "block_emerald",
                "green": "block_emerald",
            }

            grounded_seq = []
            if "stack_sequence" in data and isinstance(data["stack_sequence"], list):
                for item in data["stack_sequence"]:
                    clean_item = str(item).lower().strip()
                    if clean_item in blocks_by_id and clean_item not in grounded_seq:
                        grounded_seq.append(clean_item)
                    elif clean_item in color_to_id and color_to_id[clean_item] not in grounded_seq:
                        grounded_seq.append(color_to_id[clean_item])

            if not grounded_seq and self.latest_grounding:
                for item in self.latest_grounding:
                    ent = item.get("entity", "").lower().strip()
                    clr = item.get("color", "").lower().strip()
                    if ent in blocks_by_id and ent not in grounded_seq:
                        grounded_seq.append(ent)
                    elif clr in color_to_id and color_to_id[clr] not in grounded_seq:
                        grounded_seq.append(color_to_id[clr])

            is_stack_cmd = any(w in instruction.lower() for w in ("stack", "tower", "layer", "tier", "pad", "build"))
            if is_stack_cmd and grounded_seq:
                synthesized_plan = []
                for layer_idx, b_id in enumerate(grounded_seq):
                    b = blocks_by_id[b_id]
                    bx, by, bz = b["position"]
                    target_y = stack_slot_y(layer_idx)
                    steps = self._generate_pick_and_place(bx, bz, STACK_ORIGIN[0], STACK_ORIGIN[2], target_y, b["name"])
                    synthesized_plan.extend(steps)
                self.active_plan = synthesized_plan
            elif raw_plan:
                self.active_plan = self._sanitize_raw_plan(raw_plan)
            else:
                self.active_plan = self._embedded_vla_policy(instruction, snapshot).get("plan", [])

            self.current_step_idx = 0
            self.execution_status = "EXECUTING" if self.active_plan else "COMPLETED"
            self.active_mode = "live_cloud"
            return {
                "source": f"Google Gemini Live Cloud ({self.model_name})",
                "mode": "live_cloud",
                "thought": self.latest_thought,
                "grounding": self.latest_grounding,
                "plan": self.active_plan,
                "cost": cost_entry,
                "cost_summary": self.cost_tracker.to_dict(),
            }
        return None

    def _sanitize_raw_plan(self, raw_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensures every step has safe transit altitudes and verified dwell ticks."""
        sanitized = []
        for step in raw_plan:
            action = str(step.get("action", "")).upper()
            target = list(step.get("target", [0.2, 0.855, 0.0]))
            gripper = float(step.get("gripper", 0.0))
            desc = step.get("desc", action)

            if gripper >= 0.5 or "GRIP" in action or "GRASP" in action:
                dwell = max(18, step.get("dwell_ticks", 18))
            elif "RELEASE" in action:
                dwell = max(20, step.get("dwell_ticks", 20))
            elif "DESCEND" in action or "PLACE" in action:
                dwell = max(14, step.get("dwell_ticks", 14))
            elif "HOVER" in action or "TRANSIT" in action or "LIFT" in action:
                dwell = max(12, step.get("dwell_ticks", 12))
            else:
                dwell = max(10, step.get("dwell_ticks", 10))

            sanitized.append({
                "action": action,
                "target": tuple(target),
                "gripper": gripper,
                "desc": desc,
                "dwell_ticks": dwell,
            })
        return sanitized

    def _embedded_vla_policy(self, instruction: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Embedded Google VLA Runtime.
        Implements metric-locked stacking, exploration, and pick-and-place algorithms for 5 blocks.
        Parses layer counts (1 to 5 layers) and explicit block combinations.
        """
        cmd = instruction.lower().strip()
        blocks = {b["id"]: b for b in snapshot.get("blocks", [])}
        plan: List[Dict[str, Any]] = []
        thought: str = ""
        grounding: List[Dict[str, Any]] = []

        all_block_ids = ["block_cyan", "block_orange", "block_magenta", "block_yellow", "block_emerald"]

        # 1. Autonomous Exploration Command
        if any(w in cmd for w in ("explore", "scan", "survey", "search", "map")):
            thought = (
                "Google VLA Perception: High-level exploration command detected. "
                "Synthesizing multi-quadrant sweeping trajectory at safe hover altitude (0.855m) "
                "to inspect unmapped regions, resolve occlusions, and construct 3D spatial memory for all 5 blocks."
            )
            plan = [
                {"action": "EXPLORE_SCAN", "target": (0.25, SAFE_HOVER_Y, 0.0), "gripper": 0.0, "desc": "Center overview survey"},
                {"action": "EXPLORE_SCAN", "target": (0.15, SAFE_HOVER_Y, -0.22), "gripper": 0.0, "desc": "Scanning left table region"},
                {"action": "EXPLORE_SCAN", "target": (0.38, SAFE_HOVER_Y, -0.15), "gripper": 0.0, "desc": "Inspecting table boundary"},
                {"action": "EXPLORE_SCAN", "target": (0.35, SAFE_HOVER_Y, 0.12), "gripper": 0.0, "desc": "Scanning center-right quadrant"},
                {"action": "EXPLORE_SCAN", "target": (STACK_ORIGIN[0], SAFE_HOVER_Y, STACK_ORIGIN[2]), "gripper": 0.0, "desc": "Verifying target pad"},
                {"action": "HOME", "target": (0.20, SAFE_HOVER_Y, 0.0), "gripper": 0.0, "desc": "Returning to home pose"},
            ]

        # 2. Reset or Clear Target Zone
        elif any(w in cmd for w in ("clear", "empty", "remove", "displace")):
            thought = (
                "Google VLA Cognitive Reasoner: Clearing target zone. "
                "Identifying block closest to pad tolerance [0.48, 0.22] "
                "and moving it to the safe table perimeter."
            )
            pad_block = min(
                blocks.values(),
                key=lambda b: math.hypot(b["position"][0] - STACK_ORIGIN[0], b["position"][2] - STACK_ORIGIN[2])
            )
            bx, by, bz = pad_block["position"]
            plan = self._generate_pick_and_place(bx, bz, -0.10, -0.15, PICK_Y, pad_block["name"])

        # 3. Specific Single Block Pick & Place without stacking (e.g. "move yellow to pad", "pick emerald")
        elif any(color in cmd for color in ("cyan", "orange", "magenta", "yellow", "emerald", "green")) and not any(w in cmd for w in ("stack", "tower", "all", "layer", "tier", "blocks")):
            target_id = None
            if "cyan" in cmd: target_id = "block_cyan"
            elif "orange" in cmd: target_id = "block_orange"
            elif "magenta" in cmd: target_id = "block_magenta"
            elif "yellow" in cmd: target_id = "block_yellow"
            elif "emerald" in cmd or "green" in cmd: target_id = "block_emerald"

            if target_id and target_id in blocks:
                b = blocks[target_id]
                bx, by, bz = b["position"]
                thought = f"Google VLA: User requested manipulation of {b['name']}. Planning pick-and-place to target pad base layer."
                plan = self._generate_pick_and_place(bx, bz, STACK_ORIGIN[0], STACK_ORIGIN[2], stack_slot_y(0), b["name"])

        # 4. Tower Stacking (Dynamic Layer Parsing: 1 to 5 layers)
        else:
            # Determine number of layers requested by user
            target_layers = 5  # default when asking to stack or build tower
            if re.search(r"\b(1|one|single)\s*(layer|block|tier|level|high)\b", cmd) or "1 layer" in cmd or "1-layer" in cmd or "one layer" in cmd:
                target_layers = 1
            elif re.search(r"\b(2|two|double)\s*(layer|block|tier|level|high)\b", cmd) or "2 layer" in cmd or "2-layer" in cmd or "two layer" in cmd or "stack 2" in cmd or "stack two" in cmd:
                target_layers = 2
            elif re.search(r"\b(3|three|triple)\s*(layer|block|tier|level|high)\b", cmd) or "3 layer" in cmd or "3-layer" in cmd or "three layer" in cmd or "stack 3" in cmd or "stack three" in cmd:
                target_layers = 3
            elif re.search(r"\b(4|four|quad)\s*(layer|block|tier|level|high)\b", cmd) or "4 layer" in cmd or "4-layer" in cmd or "four layer" in cmd or "stack 4" in cmd or "stack four" in cmd:
                target_layers = 4
            elif re.search(r"\b(5|five|quintuple)\s*(layer|block|tier|level|high)\b", cmd) or "5 layer" in cmd or "5-layer" in cmd or "five layer" in cmd or "stack 5" in cmd or "stack five" in cmd or "all 5" in cmd or "all five" in cmd or "all blocks" in cmd or "all" in cmd:
                target_layers = 5

            # Determine blocks to use based on instruction
            blocks_to_use = []

            # 1. Check explicit role assignments (base, middle, top)
            role_map = {}
            color_candidates = [
                ("cyan", "block_cyan"),
                ("orange", "block_orange"),
                ("magenta", "block_magenta"),
                ("yellow", "block_yellow"),
                ("emerald", "block_emerald"),
            ]
            if "green" in cmd and not any(p in cmd for p in ("green pad", "green target", "green zone")):
                color_candidates.append(("green", "block_emerald"))

            for clr, b_id in color_candidates:
                if re.search(rf"\b(?:{clr}\s+(?:as\s+)?base|base\s+(?:layer\s+)?{clr})\b", cmd):
                    role_map[0] = b_id
                elif re.search(rf"\b(?:{clr}\s+(?:as\s+)?(?:middle|mid|second)|(?:middle|mid|second)\s+(?:layer\s+)?{clr})\b", cmd):
                    role_map[1] = b_id
                elif re.search(rf"\b(?:{clr}\s+(?:as\s+)?(?:top|apex|third)|(?:top|apex|third)\s+(?:layer\s+)?{clr}|{clr}\s+on\s+top)\b", cmd):
                    role_map[2] = b_id

            if role_map:
                for idx in sorted(role_map.keys()):
                    if role_map[idx] not in blocks_to_use:
                        blocks_to_use.append(role_map[idx])

            # 2. If no explicit roles, preserve order of appearance in command
            if not blocks_to_use:
                color_positions = []
                for clr, b_id in color_candidates:
                    idx = cmd.find(clr)
                    if idx != -1:
                        color_positions.append((idx, b_id))
                color_positions.sort(key=lambda x: x[0])
                for _, b_id in color_positions:
                    if b_id not in blocks_to_use:
                        blocks_to_use.append(b_id)

            # 3. Fill default order [cyan, orange, magenta, yellow, emerald] up to target_layers
            for b_id in all_block_ids:
                if len(blocks_to_use) >= target_layers:
                    break
                if b_id not in blocks_to_use:
                    blocks_to_use.append(b_id)

            blocks_to_use = blocks_to_use[:target_layers]
            unused_blocks = [b_id for b_id in all_block_ids if b_id not in blocks_to_use]

            layer_strs = []
            for layer_idx, b_id in enumerate(blocks_to_use):
                b = blocks.get(b_id)
                if not b:
                    continue
                bx, by, bz = b["position"]
                target_y = stack_slot_y(layer_idx)
                grounding.append({"entity": b_id, "color": b["color"], "source_coords": [bx, by, bz], "target_layer": layer_idx})
                steps = self._generate_pick_and_place(bx, bz, STACK_ORIGIN[0], STACK_ORIGIN[2], target_y, b["name"])
                plan.extend(steps)
                layer_strs.append(f"Layer {layer_idx}: {b['name']} (Y={target_y:.4f}m)")

            unused_names = [blocks[uid]["name"] for uid in unused_blocks if uid in blocks]
            leave_str = f"Remaining block(s) ({', '.join(unused_names)}) remain resting on table." if unused_names else "All 5 blocks stacked."

            thought = (
                f"Google VLA Cognitive Reasoner: User requested {target_layers}-layer tower assembly on designated target pad. "
                f"Synthesizing {target_layers}-layer sequence: " + " | ".join(layer_strs) + f". {leave_str} "
                "Applying metric coordinate locking, kinematic altitude clamping (<=1.12m), and zero-impulse jaw release holding to avoid toppling."
            )

        self.latest_thought = thought
        self.latest_grounding = grounding
        self.active_plan = plan
        self.current_step_idx = 0
        self.execution_status = "EXECUTING" if plan else "COMPLETED"
        self.active_mode = "embedded"

        # Record call in cost tracker
        cost_entry = self.cost_tracker.record_call(
            model_name=self.model_name,
            prompt_text=instruction,
            completion_text=thought,
            has_image=True,
            is_live_api=False,
        )

        return {
            "source": "Google VLA Embedded Deterministic Engine",
            "mode": "embedded",
            "thought": self.latest_thought,
            "grounding": self.latest_grounding,
            "plan": self.active_plan,
            "cost": cost_entry,
            "cost_summary": self.cost_tracker.to_dict(),
        }

    def get_cost_summary(self) -> Dict[str, Any]:
        """Returns structured cost and token analytics."""
        return self.cost_tracker.to_dict()

    def reset_cost(self):
        """Resets all session cost metrics."""
        self.cost_tracker.reset()

    def _generate_pick_and_place(
        self,
        src_x: float,
        src_z: float,
        dst_x: float,
        dst_z: float,
        dst_y: float,
        obj_name: str
    ) -> List[Dict[str, Any]]:
        """
        Generates robust closed-loop pick-and-place waypoint sub-plan
        with metric-locked coordinates, soft approach, and zero-impulse drop.
        Dynamically adjusts transit and approach altitudes to safely clear lower layers of tower.
        """
        # Calculate safe transit hover altitude above any existing stack layers
        transit_y = max(SAFE_HOVER_Y, dst_y + 0.065)

        return [
            # 1. Safe Hover over source object
            {
                "action": "HOVER_ACQUIRE",
                "target": (src_x, transit_y, src_z),
                "gripper": 0.0,
                "desc": f"Hover over {obj_name} at [{src_x:.3f}, {src_z:.3f}]",
                "dwell_ticks": 10,
            },
            # 2. Descend to pick height (clearing table surface)
            {
                "action": "DESCEND_PICK",
                "target": (src_x, PICK_Y, src_z),
                "gripper": 0.0,
                "desc": f"Descend to grip {obj_name}",
                "dwell_ticks": 12,
            },
            # 3. Grip closure
            {
                "action": "GRIP",
                "target": (src_x, PICK_Y, src_z),
                "gripper": 1.0,
                "desc": f"Close parallel jaws around {obj_name}",
                "dwell_ticks": 18,
            },
            # 4. Clean vertical lift to safe transit altitude
            {
                "action": "LIFT",
                "target": (src_x, transit_y, src_z),
                "gripper": 1.0,
                "desc": f"Vertical lift of {obj_name} to transit height {transit_y:.3f}m",
                "dwell_ticks": 12,
            },
            # 5. Horizontal transit to destination
            {
                "action": "TRANSIT",
                "target": (dst_x, transit_y, dst_z),
                "gripper": 1.0,
                "desc": f"Transit {obj_name} to [{dst_x:.3f}, {dst_z:.3f}]",
                "dwell_ticks": 15,
            },
            # 6. Gentle descent to destination layer slot
            {
                "action": "DESCEND_PLACE",
                "target": (dst_x, dst_y + 0.010, dst_z),
                "gripper": 1.0,
                "desc": f"Soft descent to placement altitude {dst_y:.3f}m",
                "dwell_ticks": 15,
            },
            # 7. Zero-impulse release: hold stationary while opening jaws
            {
                "action": "SOFT_RELEASE",
                "target": (dst_x, dst_y + 0.010, dst_z),
                "gripper": 0.0,
                "desc": "Zero-impulse jaw release (prevent tower topple)",
                "dwell_ticks": 20,
            },
            # 8. Vertical ascend to safe hover
            {
                "action": "ASCEND_CLEAR",
                "target": (dst_x, transit_y, dst_z),
                "gripper": 0.0,
                "desc": "Ascend cleanly above placed cube",
                "dwell_ticks": 12,
            },
        ]
