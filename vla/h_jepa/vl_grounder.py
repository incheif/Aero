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

from simulation.constants import (
    STACK_ORIGIN,
    get_pyramid_slots,
    get_train_slots,
    get_letter_slots,
    get_distinct_tower_origins,
    TABLE_TOP_Y,
    CUBE_SIZE,
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
        is_tower = any(w in inst_lower for w in ["tower", "stack", "vertical"]) and not any(w in inst_lower for w in ["pyramid", "stepped", "triangle"]) and not is_multi_towers
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
        elif initial_blocks:
            b_ids = [b.get("id") for b in initial_blocks if b.get("id")][:len(target_slots)]
        else:
            default_pool = list(AVAILABLE_BLOCKS.keys()) if AVAILABLE_BLOCKS else [
                "cube_red", "cube_blue", "cube_green", "cube_yellow", "cube_cyan", "cube_purple", "cube_pink"
            ]
            b_ids = default_pool[:len(target_slots)]

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
        initial_blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generates the sequential intermediate sub-goal latent embeddings z_sub^(2)
        corresponding to each step of the pyramid or tower assembly:
        e.g., Step 1: Base-1 placed
              Step 2: Base-2 placed
              Step 3: Apex placed
        """
        subgoals = []
        curr_state_blocks = [dict(b) for b in initial_blocks]

        for step_idx, (b_id, slot) in enumerate(zip(block_ids, target_slots)):
            # Update virtual state: place block at its designated target
            for b in curr_state_blocks:
                if b.get("id") == b_id:
                    b["position"] = [slot[0], slot[1], slot[2]]
                    b["rotation"] = [0.0, 0.0, 0.0, 1.0]
                    b["in_pad"] = True
                    b["is_grasped"] = False
                    break

            # Encode intermediate scene state
            z_sub = self.encoder_l2.encode(
                blocks=curr_state_blocks,
                tcp_pos=[slot[0], slot[1] + 0.08, slot[2]],
                target_origin=STACK_ORIGIN,
            )

            tier_desc = f"Tier {round(max(0, slot[1] - TABLE_TOP_Y) / CUBE_SIZE)}"
            subgoals.append({
                "subgoal_index": step_idx,
                "block_id": b_id,
                "target_slot": slot,
                "z_sub": z_sub,
                "description": f"Assemble {b_id} at {tier_desc} slot ({slot[0]:.3f}, {slot[1]:.3f}, {slot[2]:.3f})",
            })

        return subgoals
