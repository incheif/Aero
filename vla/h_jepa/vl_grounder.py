"""
vla/h_jepa/vl_grounder.py

VL-JEPA Style Language Grounding Engine.
Embeds natural language instructions directly into Level 2 Macro Latent Space:
  G(text_prompt) -> z_goal^(2) in R^128
Translates high-level multi-box manipulation directives (e.g. 3-block, 5-block,
6-block stepped pyramids, linear trains) into goal and intermediate sub-goal embeddings.
"""

import math
import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from vla.instruction_interpreter import COLOR_ALIAS_MAP, AVAILABLE_BLOCKS
from simulation.constants import (
    STACK_ORIGIN,
    get_pyramid_slots,
    get_train_slots,
    get_letter_slots,
    get_distinct_tower_origins,
    TABLE_TOP_Y,
    CUBE_SIZE,
    CUBE_HALF,
)
from vla.instruction_interpreter import AVAILABLE_BLOCKS
from .encoders import Level2ObservationEncoder

class VLJEPAGrounder:
    """
    Grounds high-level natural language instructions directly into
    Level 2 Macro Latent Space z_goal^(2) in R^128.
    """
    def __init__(self, encoder_l2: Optional[Level2ObservationEncoder] = None):
        self.encoder_l2 = encoder_l2 or Level2ObservationEncoder()
        self.latent_dim = 128

        # Semantic projection matrix from text token / feature representation to R^128
        rng = np.random.RandomState(60)
        self.text_W = rng.randn(64, self.latent_dim) * math.sqrt(2.0 / 64)
        self.text_b = np.zeros(self.latent_dim)

    def extract_text_features(self, text: str) -> np.ndarray:
        """
        Extracts a normalized 64-dim semantic feature vector from instruction text
        using keyword priors and deterministic token hashing.
        """
        text_lower = text.lower().strip()
        feats = np.zeros(64, dtype=np.float32)

        # Semantic intent flags
        feats[0] = 1.0 if any(w in text_lower for w in ["pyramid", "pyramind", "stepped", "triangle"]) else 0.0
        feats[1] = 1.0 if any(w in text_lower for w in ["tower", "stack", "vertical"]) else 0.0
        feats[2] = 1.0 if any(w in text_lower for w in ["train", "line", "row", "horizontal"]) else 0.0
        feats[3] = 1.0 if any(w in text_lower for w in ["3", "three"]) else 0.0
        feats[4] = 1.0 if any(w in text_lower for w in ["5", "five", "square"]) else 0.0
        feats[5] = 1.0 if any(w in text_lower for w in ["6", "six"]) else 0.0

        # Colors mentioned
        color_map = {
            "red": 10, "blue": 11, "green": 12, "yellow": 13,
            "cyan": 14, "purple": 15, "pink": 16
        }
        for color, idx in color_map.items():
            if color in text_lower:
                feats[idx] = 1.0

        # Deterministic n-gram hash features for general prompt coverage
        tokens = text_lower.split()
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            slot = 20 + (h % 44)
            feats[slot] += 1.0

        # Normalize
        norm = np.linalg.norm(feats)
        if norm > 1e-6:
            feats = feats / norm
        return feats

    def ground_goal_embedding(
        self,
        instruction: str,
        target_slots: Optional[List[List[float]]] = None,
        block_ids: Optional[List[str]] = None,
        initial_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Synthesizes the Level 2 target goal embedding z_goal^(2) in R^128.
        Combines linguistic intent with idealized target configuration.
        """
        text_feats = self.extract_text_features(instruction)
        z_text = text_feats @ self.text_W + self.text_b
        z_text = z_text / max(1e-6, np.linalg.norm(z_text))

        # Determine target structure configuration
        inst_lower = instruction.lower()
        is_letter = None
        for ltr in ["d", "l", "t", "o", "c", "u", "h"]:
            matches = [
                f"letter {ltr}", f"make {ltr}", f"form {ltr}", f"shape {ltr}",
                f"pattern {ltr}", f"{ltr} shape", f"{ltr} pattern", f"{ltr} with blocks",
                f"'{ltr}'", f'"{ltr}"'
            ]
            if any(p in inst_lower for p in matches) or (
                f" {ltr} " in f" {inst_lower} " and any(w in inst_lower for w in ["make", "build", "place", "form", "pattern", "letter", "shape", "blocks"])
            ):
                is_letter = ltr.upper()
                break

        is_multi_towers = (
            any(p in inst_lower for p in (
                "distinct tower", "distinct towers", "separate tower", "separate towers",
                "different tower", "different towers", "2 towers", "two towers",
                "3 towers", "three towers", "dual towers", "dual tower", "pair of towers"
            ))
            or bool(re.search(r"\b(2|two|3|three|4|four|5|five)\s+(?:distinct\s+|separate\s+|different\s+)?towers?\b", inst_lower))
        )
        is_tower = any(w in inst_lower for w in ["tower", "stack", "vertical"]) and not any(w in inst_lower for w in ["pyramid", "pyramind", "stepped", "triangle"]) and not is_multi_towers
        is_train = "train" in inst_lower or "line" in inst_lower or "row" in inst_lower
        
        pyr_type = "3_block_stepped"
        if any(w in inst_lower for w in ["6", "six"]):
            pyr_type = "6_block_stepped"
        elif any(w in inst_lower for w in ["5", "five", "square"]):
            pyr_type = "5_block_square"
        
        if target_slots is None:
            if is_letter:
                target_slots = get_letter_slots(is_letter, (0.35, TABLE_TOP_Y, 0.0))
            elif is_train:
                if any(w in inst_lower for w in ["all", "every", "7", "seven"]):
                    num_t = 7
                else:
                    num_t = 4
                    for token, cnt in [("2", 2), ("3", 3), ("4", 4), ("5", 5), ("6", 6), ("7", 7)]:
                        if token in inst_lower:
                            num_t = cnt
                            break
                target_slots = get_train_slots(num_t, (0.38, TABLE_TOP_Y, 0.0))
            elif is_multi_towers:
                m_t = re.search(r"\b(2|two|3|three|4|four|5|five)\s+(?:distinct\s+|separate\s+|different\s+)?towers?\b", inst_lower)
                word_to_num = {"1": 1, "one": 1, "2": 2, "two": 2, "3": 3, "three": 3, "4": 4, "four": 4, "5": 5, "five": 5}
                num_towers = word_to_num.get(m_t.group(1), 2) if m_t else 2
                m_b = re.search(r"\b(1|one|2|two|3|three|4|four|5|five)\s+(?:blocks?|cubes?|layers?)\s+(?:each|per)", inst_lower)
                if not m_b:
                    m_b = re.search(r"(?:each|per)\s+(?:tower\s+)?(?:has\s+)?(1|one|2|two|3|three|4|four|5|five)\s+(?:blocks?|cubes?|layers?)", inst_lower)
                blocks_per_tower = word_to_num.get(m_b.group(1), 2) if m_b else 2
                origins = get_distinct_tower_origins(num_towers)
                target_slots = []
                for t_idx in range(num_towers):
                    t_origin = origins[t_idx] if t_idx < len(origins) else STACK_ORIGIN
                    for layer in range(blocks_per_tower):
                        target_slots.append([
                            t_origin[0],
                            TABLE_TOP_Y + CUBE_SIZE * (layer + 0.5),
                            t_origin[2]
                        ])
            elif is_tower:
                num_layers = 1
                for token, cnt in [
                    ("1", 1), ("one", 1), ("single", 1),
                    ("2", 2), ("two", 2), ("double", 2),
                    ("3", 3), ("three", 3), ("triple", 3),
                    ("4", 4), ("four", 4), ("quad", 4),
                    ("5", 5), ("five", 5),
                    ("6", 6), ("six", 6),
                    ("7", 7), ("seven", 7), ("all", 7),
                ]:
                    if token in inst_lower:
                        num_layers = max(num_layers, cnt)
                target_slots = [
                    [STACK_ORIGIN[0], TABLE_TOP_Y + CUBE_SIZE * (layer + 0.5), STACK_ORIGIN[2]]
                    for layer in range(num_layers)
                ]
            else:
                target_slots = get_pyramid_slots(pyr_type, STACK_ORIGIN)

        if block_ids:
            b_ids = block_ids[:len(target_slots)]
        else:
            # Extract color mentions from the instruction text
            sorted_aliases = sorted(COLOR_ALIAS_MAP.keys(), key=lambda x: len(x), reverse=True)
            found_colors = []
            for alias in sorted_aliases:
                for m in re.finditer(rf"\b{re.escape(alias)}\b", inst_lower):
                    if alias == "green" and any(p in inst_lower for p in ("green pad", "green target", "green zone", "green circle")):
                        continue
                    found_colors.append((m.start(), COLOR_ALIAS_MAP[alias]))
            found_colors.sort(key=lambda x: x[0])
            mentioned = []
            for _, b_id in found_colors:
                if b_id not in mentioned:
                    mentioned.append(b_id)

            if is_multi_towers:
                split_pattern = r"\b(?:next\s+tower|second\s+tower|2nd\s+tower|other\s+tower|tower\s+2|tower\s+two|third\s+tower|3rd\s+tower|tower\s+3|tower\s+three|fourth\s+tower|tower\s+4|fifth\s+tower|tower\s+5|and\s+then\s+(?:another\s+)?tower)\b"
                parts = re.split(split_pattern, inst_lower)
                all_blocks = [b.get("id") for b in initial_blocks] if initial_blocks else list(AVAILABLE_BLOCKS.keys())
                allocated = set()
                towers = []
                if len(parts) >= num_towers:
                    for i in range(num_towers):
                        part_c = []
                        for alias in sorted_aliases:
                            for m in re.finditer(rf"\b{re.escape(alias)}\b", parts[i]):
                                if alias == "green" and any(p in parts[i] for p in ("green pad", "green target")):
                                    continue
                                part_c.append((m.start(), COLOR_ALIAS_MAP[alias]))
                        part_c.sort(key=lambda x: x[0])
                        t_b = [b_id for _, b_id in part_c if b_id not in allocated][:blocks_per_tower]
                        for b_id in t_b:
                            allocated.add(b_id)
                        towers.append(t_b)
                else:
                    curr = 0
                    for _ in range(num_towers):
                        t_b = [b for b in mentioned[curr:curr + blocks_per_tower] if b not in allocated]
                        for b in t_b:
                            allocated.add(b)
                        towers.append(t_b)
                        curr += blocks_per_tower

                remaining_pool = [b for b in all_blocks if b not in allocated]
                for i in range(len(towers)):
                    while len(towers[i]) < blocks_per_tower and remaining_pool:
                        fill_b = remaining_pool.pop(0)
                        towers[i].append(fill_b)
                        allocated.add(fill_b)
                b_ids = [b for t in towers for b in t][:len(target_slots)]
            else:
                all_available = [b.get("id") for b in initial_blocks] if initial_blocks else list(AVAILABLE_BLOCKS.keys())
                b_ids = list(mentioned)
                for b in all_available:
                    if b not in b_ids:
                        b_ids.append(b)
                b_ids = b_ids[:len(target_slots)]

        # Synthesize virtual ideal target state for encoder evaluation
        ideal_blocks = []
        for idx, (b_id, slot) in enumerate(zip(b_ids, target_slots)):
            ideal_blocks.append({
                "id": b_id,
                "position": [slot[0], slot[1], slot[2]],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "in_pad": True,
                "is_grasped": False,
            })

        # Encode ideal geometric state via E^(2)
        z_geom = self.encoder_l2.encode(
            blocks=ideal_blocks,
            tcp_pos=[STACK_ORIGIN[0], STACK_ORIGIN[1] + 0.25, STACK_ORIGIN[2]],
            target_origin=STACK_ORIGIN,
        )

        # Fused goal embedding (50% text intent prior + 50% geometric target attractor)
        z_goal = 0.5 * z_text + 0.5 * z_geom
        z_goal = z_goal / max(1e-6, np.linalg.norm(z_goal))

        st_type = f"{num_towers}_distinct_towers" if is_multi_towers else (f"{len(target_slots)}_block_tower" if is_tower else ("linear_train" if is_train else pyr_type))
        metadata = {
            "structure_type": st_type,
            "num_blocks": len(target_slots),
            "target_slots": target_slots,
            "target_blocks": b_ids,
        }
        return z_goal.astype(np.float32), metadata

    def generate_subgoal_embeddings(
        self,
        target_slots: List[List[float]],
        block_ids: List[str],
        initial_blocks: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates the sequential intermediate sub-goal latent embeddings z_sub^(2)
        corresponding to each step of the pyramid, tower, or train assembly.
        If any existing block physically obstructs a target slot before its turn,
        a pre-clearing subgoal is prepended to move that block to a distant staging location.
        """
        subgoals = []
        curr_state_blocks = [dict(b) for b in initial_blocks] if initial_blocks else []

        # 1. Path Obstruction Detection & Pre-clearing Subgoals
        staging_positions = [
            [0.15, TABLE_TOP_Y + CUBE_HALF, 0.08],
            [0.15, TABLE_TOP_Y + CUBE_HALF, -0.12],
            [0.12, TABLE_TOP_Y + CUBE_HALF, 0.0],
        ]
        staging_idx = 0

        curr_positions = {}
        if initial_blocks:
            for b in initial_blocks:
                if b.get("id"):
                    curr_positions[b["id"]] = list(b.get("position", [0.0, 0.0, 0.0]))

        cleared_blocks = set()
        clearing_subgoals = []

        for step_idx, (b_id, slot) in enumerate(zip(block_ids, target_slots)):
            for other_id, other_pos in list(curr_positions.items()):
                if other_id == b_id or other_id in cleared_blocks:
                    continue
                horiz_dist = math.hypot(other_pos[0] - slot[0], other_pos[2] - slot[2])
                vert_dist = abs(other_pos[1] - slot[1])
                if horiz_dist < CUBE_SIZE * 0.95 and vert_dist < CUBE_SIZE:
                    other_scheduled_idx = block_ids.index(other_id) if other_id in block_ids else 999
                    if other_scheduled_idx > step_idx:
                        stage_slot = staging_positions[staging_idx % len(staging_positions)]
                        staging_idx += 1

                        for b in curr_state_blocks:
                            if b.get("id") == other_id:
                                b["position"] = list(stage_slot)
                                b["is_grasped"] = False
                                break
                        curr_positions[other_id] = list(stage_slot)
                        cleared_blocks.add(other_id)

                        z_clear = self.encoder_l2.encode(
                            blocks=curr_state_blocks,
                            tcp_pos=[stage_slot[0], stage_slot[1] + 0.08, stage_slot[2]],
                            target_origin=STACK_ORIGIN,
                        )
                        clearing_subgoals.append({
                            "block_id": other_id,
                            "target_slot": stage_slot,
                            "z_sub": z_clear,
                            "description": f"Pre-clear obstructing {other_id} from path to distant location ({stage_slot[0]:.3f}, {stage_slot[1]:.3f}, {stage_slot[2]:.3f})",
                        })

        for cs in clearing_subgoals:
            cs["subgoal_index"] = len(subgoals)
            subgoals.append(cs)

        # 2. Main Assembly Subgoals
        for step_idx, (b_id, slot) in enumerate(zip(block_ids, target_slots)):
            for b in curr_state_blocks:
                if b.get("id") == b_id:
                    b["position"] = [slot[0], slot[1], slot[2]]
                    b["rotation"] = [0.0, 0.0, 0.0, 1.0]
                    b["in_pad"] = True
                    b["is_grasped"] = False
                    break

            z_sub = self.encoder_l2.encode(
                blocks=curr_state_blocks,
                tcp_pos=[slot[0], slot[1] + 0.08, slot[2]],
                target_origin=STACK_ORIGIN,
            )

            tier_desc = f"Tier {round(max(0, slot[1] - TABLE_TOP_Y) / CUBE_SIZE)}"
            subgoals.append({
                "subgoal_index": len(subgoals),
                "block_id": b_id,
                "target_slot": slot,
                "z_sub": z_sub,
                "description": f"Assemble {b_id} at {tier_desc} slot ({slot[0]:.3f}, {slot[1]:.3f}, {slot[2]:.3f})",
            })

        return subgoals
