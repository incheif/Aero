"""
vla/jepa/level2_world_model.py

Level 2 Macro Subgoal Planner, Consequence Predictor, and Stability Probe.
Operates at subtask boundaries (1 step = 1 complete pick-and-place phase).
- F^(2): (z^(2)_k, u_k) -> \hat{z}^(2)_{k+1} in latent space R^512.
- h_stab(z^(2)): Evaluates tower equilibrium and overhang stability.
- Dynamic CoM Lookahead: Shifts target placement coordinates toward the actual
  center of mass of supporting blocks to prevent toppling.
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from simulation.constants import (
    STACK_ORIGIN,
    STACK_ORIGIN_A,
    STACK_ORIGIN_B,
    TABLE_TOP_Y,
    CUBE_SIZE,
    CUBE_HALF,
    stack_slot_y,
    get_pyramid_slots,
    get_train_slots,
    get_letter_slots,
    get_distinct_tower_origins,
)
from .encoders import (
    Level2ObservationEncoder,
    VLTextGoalEncoder,
    layer_norm,
    gelu,
)
from .schemas import SubgoalSpecification


class StabilityProbe:
    """
    Auxiliary stability probe h_stab(z^(2)) in [0, 1].
    Trained to predict whether a multi-block structure will remain standing
    or topple under gravity given the Level 2 latent representation.
    """
    def __init__(self, latent_dim: int = 512):
        self.latent_dim = latent_dim
        rng = np.random.RandomState(301)
        self.W1 = rng.randn(latent_dim, 64) * math.sqrt(2.0 / latent_dim)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = rng.randn(64, 1) * math.sqrt(2.0 / 64)
        self.b2 = np.zeros(1, dtype=np.float32)

    def predict_stability(self, z2: np.ndarray) -> float:
        """Outputs equilibrium stability probability in [0, 1]."""
        h = gelu(z2 @ self.W1 + self.b1)
        out = np.asarray(h @ self.W2 + self.b2).reshape(-1)
        logit = float(out[0])
        prob = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, logit))))
        return float(prob)



class Level2Predictor:
    """
    F^(2): Macro-level latent world model predictor.
    Forecasts multi-phase state evolution across macro temporal jumps:
      (z^(2)_k, u_k) -> \hat{z}^(2)_{k+1} in R^{latent_dim} (512).
    """
    def __init__(self, state_dim: int = 512, macro_action_dim: int = 16):
        self.state_dim = state_dim
        self.macro_action_dim = macro_action_dim
        in_dim = state_dim + macro_action_dim

        rng = np.random.RandomState(302)
        self.W1 = rng.randn(in_dim, state_dim) * math.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(state_dim, dtype=np.float32)
        self.W2 = rng.randn(state_dim, state_dim) * math.sqrt(2.0 / state_dim)
        self.b2 = np.zeros(state_dim, dtype=np.float32)

    def predict(self, z2: np.ndarray, u_macro: np.ndarray) -> np.ndarray:
        single = (z2.ndim == 1)
        if single:
            z2 = z2.reshape(1, -1)
            u_macro = u_macro.reshape(1, -1)

        cat = np.concatenate([z2, u_macro], axis=-1)
        h = gelu(cat @ self.W1 + self.b1)
        h = layer_norm(h)
        delta_z = h @ self.W2 + self.b2

        z_next = z2 + delta_z
        norm = np.linalg.norm(z_next, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1e-6)
        z_next = z_next / norm

        if single:
            return z_next[0].astype(np.float32)
        return z_next.astype(np.float32)


class Level2TaskPlanner:
    """
    Level 2 Semantic Task Decomposer and Macro Subgoal Generator.
    Decomposes compositional instructions ('5-block tower', '3-block pyramid', 'train')
    into ordered SubgoalSpecification queues with CoM lookahead alignment.
    """
    def __init__(
        self,
        encoder_l2: Optional[Level2ObservationEncoder] = None,
        text_encoder: Optional[VLTextGoalEncoder] = None,
        predictor_l2: Optional[Level2Predictor] = None,
        stability_probe: Optional[StabilityProbe] = None,
    ):
        self.encoder_l2 = encoder_l2 or Level2ObservationEncoder()
        self.text_encoder = text_encoder or VLTextGoalEncoder()
        self.predictor_l2 = predictor_l2 or Level2Predictor()
        self.stability_probe = stability_probe or StabilityProbe()

    def decompose_task(
        self,
        instruction: str,
        current_blocks: List[Dict[str, Any]],
        target_origin: Tuple[float, float, float] = STACK_ORIGIN,
    ) -> List[SubgoalSpecification]:
        """
        Decomposes natural language instruction into ordered SubgoalSpecification sequence.
        Supports N-block towers (1 to 7 blocks), stepped pyramids (3, 5, 6 blocks), and trains.
        """
        inst_lower = instruction.lower().strip()
        subgoals: List[SubgoalSpecification] = []

        # Available block candidates in workcell
        available_ids = [b["id"] for b in current_blocks if b.get("id")]
        if not available_ids:
            available_ids = [
                "block_magenta", "block_cyan", "block_yellow",
                "block_emerald", "block_orange", "block_purple", "block_pink"
            ]

        # 1. Check for Pyramid Structure
        if any(w in inst_lower for w in ["pyramid", "stepped"]):
            pyr_style = "3_block_stepped"
            num_blocks = 3
            if any(w in inst_lower for w in ["6", "six"]):
                pyr_style = "6_block_stepped"
                num_blocks = 6
            elif any(w in inst_lower for w in ["5", "five", "square"]):
                pyr_style = "5_block_square"
                num_blocks = 5

            slots = get_pyramid_slots(pyr_style, target_origin)
            task_blocks = available_ids[:len(slots)]

            for idx, (b_id, slot) in enumerate(zip(task_blocks, slots)):
                tier = round(max(0.0, slot[1] - TABLE_TOP_Y) / CUBE_SIZE)
                desc = f"Pyramid Subgoal {idx+1}/{len(slots)}: Place {b_id} at Tier {tier} ({slot[0]:.3f}, {slot[1]:.3f}, {slot[2]:.3f})"
                z_txt = self.text_encoder.encode_text(f"{instruction} step {idx+1} {b_id}")
                prob = self.stability_probe.predict_stability(z_txt)

                subgoals.append(SubgoalSpecification(
                    subgoal_id=idx,
                    action_type="stack" if tier > 0 else "pick_and_place",
                    source_label=b_id,
                    source_color=b_id.replace("block_", "").replace("cube_", ""),
                    target_xyz=(float(slot[0]), float(slot[1]), float(slot[2])),
                    support_label="table" if tier == 0 else task_blocks[max(0, idx - 1)],
                    target_latent_l2=z_txt,
                    predicted_stability_prob=max(0.85, prob),
                    description=desc,
                ))
            return subgoals

        # 2. Check for Linear Train / Line of blocks
        if any(w in inst_lower for w in ["train", "line", "row", "horizontal"]):
            if any(w in inst_lower for w in ["all", "every", "7", "seven"]):
                num_blocks = len(available_ids)
            else:
                num_blocks = min(len(available_ids), 4)
                for w, n in [("2", 2), ("3", 3), ("4", 4), ("5", 5), ("6", 6), ("7", 7)]:
                    if w in inst_lower:
                        num_blocks = n
                        break
            slots = get_train_slots(num_blocks, (0.38, TABLE_TOP_Y, 0.0))
            task_blocks = available_ids[:len(slots)]

            for idx, (b_id, slot) in enumerate(zip(task_blocks, slots)):
                desc = f"Train Subgoal {idx+1}/{len(slots)}: Align {b_id} at slot ({slot[0]:.3f}, {slot[1]:.3f}, {slot[2]:.3f})"
                z_txt = self.text_encoder.encode_text(f"train step {idx+1} {b_id}")
                subgoals.append(SubgoalSpecification(
                    subgoal_id=idx,
                    action_type="pick_and_place",
                    source_label=b_id,
                    source_color=b_id.replace("block_", ""),
                    target_xyz=(float(slot[0]), float(slot[1]), float(slot[2])),
                    support_label="table",
                    target_latent_l2=z_txt,
                    predicted_stability_prob=0.99,
                    description=desc,
                ))
            return subgoals

        # 3. Check for Letter / Alphabet / Pattern Placement (e.g. 'D', 'L', 'T', 'O', 'C', 'U', 'H')
        letter_target = None
        for ltr in ["d", "l", "t", "o", "c", "u", "h"]:
            matches = [
                f"letter {ltr}", f"make {ltr}", f"form {ltr}", f"shape {ltr}",
                f"pattern {ltr}", f"{ltr} shape", f"{ltr} pattern", f"{ltr} with blocks",
                f"'{ltr}'", f'"{ltr}"'
            ]
            if any(p in inst_lower for p in matches) or (
                f" {ltr} " in f" {inst_lower} " and any(w in inst_lower for w in ["make", "build", "place", "form", "pattern", "letter", "shape", "blocks"])
            ):
                letter_target = ltr.upper()
                break

        if letter_target:
            slots = get_letter_slots(letter_target, (0.35, TABLE_TOP_Y, 0.0))
            task_blocks = available_ids[:len(slots)]
            for idx, (b_id, slot) in enumerate(zip(task_blocks, slots)):
                desc = f"Letter '{letter_target}' Subgoal {idx+1}/{len(slots)}: Place {b_id} at slot ({slot[0]:.3f}, {slot[1]:.3f}, {slot[2]:.3f})"
                z_txt = self.text_encoder.encode_text(f"letter {letter_target} placement step {idx+1} {b_id}")
                subgoals.append(SubgoalSpecification(
                    subgoal_id=idx,
                    action_type="pick_and_place",
                    source_label=b_id,
                    source_color=b_id.replace("block_", ""),
                    target_xyz=(float(slot[0]), float(slot[1]), float(slot[2])),
                    support_label="table",
                    target_latent_l2=z_txt,
                    predicted_stability_prob=0.99,
                    description=desc,
                ))
            return subgoals

        # 3b. Check for Multi-Tower Distinct Assembly
        is_multi_towers = (
            any(p in inst_lower for p in (
                "distinct tower", "distinct towers", "separate tower", "separate towers",
                "different tower", "different towers", "2 towers", "two towers",
                "3 towers", "three towers", "dual towers", "dual tower", "pair of towers"
            ))
            or bool(re.search(r"\b(2|two|3|three|4|four|5|five)\s+(?:distinct\s+|separate\s+|different\s+)?towers?\b", inst_lower))
        )
        if is_multi_towers:
            m_t = re.search(r"\b(2|two|3|three|4|four|5|five)\s+(?:distinct\s+|separate\s+|different\s+)?towers?\b", inst_lower)
            word_to_num = {"1": 1, "one": 1, "2": 2, "two": 2, "3": 3, "three": 3, "4": 4, "four": 4, "5": 5, "five": 5}
            num_towers = word_to_num.get(m_t.group(1), 2) if m_t else 2
            m_b = re.search(r"\b(1|one|2|two|3|three|4|four|5|five)\s+(?:blocks?|cubes?|layers?)\s+(?:each|per)", inst_lower)
            if not m_b:
                m_b = re.search(r"(?:each|per)\s+(?:tower\s+)?(?:has\s+)?(1|one|2|two|3|three|4|four|5|five)\s+(?:blocks?|cubes?|layers?)", inst_lower)
            blocks_per_tower = word_to_num.get(m_b.group(1), 2) if m_b else 2
            origins = get_distinct_tower_origins(num_towers)

            total_needed = num_towers * blocks_per_tower
            task_blocks = available_ids[:total_needed]

            subgoal_idx = 0
            for t_idx in range(num_towers):
                t_origin = origins[t_idx] if t_idx < len(origins) else target_origin
                prev_id = "table"
                for layer in range(blocks_per_tower):
                    if subgoal_idx >= len(task_blocks):
                        break
                    b_id = task_blocks[subgoal_idx]
                    slot_y = stack_slot_y(layer)
                    target_xyz = (float(t_origin[0]), float(slot_y), float(t_origin[2]))
                    support = "table" if layer == 0 else prev_id
                    prev_id = b_id

                    desc = f"Distinct Tower {t_idx+1} Subgoal {layer+1}/{blocks_per_tower}: Place {b_id} onto Tier {layer} ({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f})"
                    z_txt = self.text_encoder.encode_text(f"tower {t_idx+1} layer {layer} {b_id} on {support}")
                    subgoals.append(SubgoalSpecification(
                        subgoal_id=subgoal_idx,
                        action_type="stack" if layer > 0 else "pick_and_place",
                        source_label=b_id,
                        source_color=b_id.replace("block_", ""),
                        target_xyz=target_xyz,
                        support_label=support,
                        target_latent_l2=z_txt,
                        predicted_stability_prob=0.98,
                        description=desc,
                    ))
                    subgoal_idx += 1
            return subgoals

        # 4. Default / Tower Structure (1 to 7 blocks vertical stack)
        num_layers = 1
        # Detect count
        count_words = {
            "1": 1, "one": 1, "single": 1,
            "2": 2, "two": 2, "double": 2,
            "3": 3, "three": 3, "triple": 3,
            "4": 4, "four": 4, "quad": 4,
            "5": 5, "five": 5,
            "6": 6, "six": 6,
            "7": 7, "seven": 7, "all": min(7, len(available_ids)),
        }
        for token, count in count_words.items():
            if token in inst_lower:
                num_layers = max(num_layers, count)

        num_layers = min(num_layers, len(available_ids), 7)
        task_blocks = available_ids[:num_layers]

        for layer in range(num_layers):
            b_id = task_blocks[layer]
            slot_y = stack_slot_y(layer)
            target_xyz = (target_origin[0], slot_y, target_origin[2])
            support = "table" if layer == 0 else task_blocks[layer - 1]

            desc = f"Tower Subgoal {layer+1}/{num_layers}: Stack {b_id} onto Tier {layer} ({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f})"
            z_txt = self.text_encoder.encode_text(f"tower layer {layer} {b_id} on {support}")
            prob = self.stability_probe.predict_stability(z_txt)

            subgoals.append(SubgoalSpecification(
                subgoal_id=layer,
                action_type="stack" if layer > 0 else "pick_and_place",
                source_label=b_id,
                source_color=b_id.replace("block_", ""),
                target_xyz=target_xyz,
                support_label=support,
                target_latent_l2=z_txt,
                predicted_stability_prob=max(0.80, prob),
                description=desc,
            ))

        return subgoals

    def adjust_target_com_lookahead(
        self,
        subgoal: SubgoalSpecification,
        current_blocks: List[Dict[str, Any]],
    ) -> Tuple[float, float, float]:
        """
        Dynamic CoM Alignment: If placing on top of a lower block (support_label != 'table'),
        queries the real observed 3D position of that support block. If the lower block is
        offset by a few millimeters, shifts the target placement (X, Z) directly above
        its actual center of mass rather than the nominal pad center.
        Prevents STACK_OFFSET_INSTABILITY = 0.70 toppling.
        """
        if not subgoal.support_label or subgoal.support_label == "table":
            return subgoal.target_xyz

        for b in current_blocks:
            if b.get("id") == subgoal.support_label:
                pos = b.get("position", [0.0, 0.0, 0.0])
                # Shift X and Z to match lower block's center of mass
                adjusted_xyz = (
                    float(pos[0]),
                    float(subgoal.target_xyz[1]),
                    float(pos[2]),
                )
                return adjusted_xyz

        return subgoal.target_xyz
