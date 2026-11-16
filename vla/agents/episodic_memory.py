"""
vla/agents/episodic_memory.py

Episodic Experience & Failure Memory Agent for Robotic Manipulation.
Records execution anomalies, slips, collisions, and human corrections in persistent
storage (`models/episodic_memory.json`).

Before executing subgoals, the arm queries past episodes to retrieve learned
adjustments and prevent repeating past mistakes.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()
    elif hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


class EpisodicMemoryAgent:
    """
    Learns from mistakes and human interventions by storing and indexing
    episodic failure-recovery tuples.
    """

    def __init__(self, storage_path: str = "models/episodic_memory.json"):
        self.storage_path = storage_path
        self.episodes: List[Dict[str, Any]] = []
        self._load_memory()

    def _load_memory(self):
        """Loads persistent episodic memory from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.episodes = data
            except Exception as e:
                print(f"[EpisodicMemoryAgent] Warning: Failed to load {self.storage_path}: {e}")
                self.episodes = []
        else:
            self.episodes = []

    def _save_memory(self):
        """Saves current episodic memory to disk."""
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.episodes, f, indent=2)
        except Exception as e:
            print(f"[EpisodicMemoryAgent] Warning: Failed to save {self.storage_path}: {e}")

    def record_episode(
        self,
        task_instruction: str,
        subgoal_id: str,
        failure_type: str,
        entity: Optional[str],
        human_guidance: str,
        adaptation: Dict[str, Any],
        outcome: str = "SUCCESS",
    ) -> Dict[str, Any]:
        """
        Records an interaction episode where a mistake or anomaly was resolved.
        """
        episode = {
            "episode_id": f"ep_{int(time.time() * 1000)}",
            "timestamp": time.time(),
            "task_instruction": str(task_instruction or ""),
            "subgoal_id": str(subgoal_id or ""),
            "failure_type": str(failure_type or ""),
            "entity": entity,
            "human_guidance": str(human_guidance or ""),
            "adaptation": _sanitize_for_json(adaptation or {}),
            "outcome": outcome,
        }
        self.episodes.append(episode)
        if len(self.episodes) > 500:
            self.episodes.pop(0)
        self._save_memory()
        return episode

    def query_experience(
        self,
        task_instruction: str,
        entity: Optional[str] = None,
        failure_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves relevant past experiences to proactively avoid repeating a mistake.
        Matches on entity, failure_type, or semantic keywords in task_instruction.
        """
        task_lower = str(task_instruction or "").lower()
        candidates = []

        for ep in reversed(self.episodes):
            if ep.get("outcome") not in ("SUCCESS", "RESOLVED", "RESOLVED_BY_USER"):
                continue
            
            score = 0
            if entity and ep.get("entity") == entity:
                score += 3
            if failure_type and ep.get("failure_type") == failure_type:
                score += 2
            
            ep_task = ep.get("task_instruction", "").lower()
            if ep_task and any(word in task_lower for word in ep_task.split() if len(word) > 3):
                score += 1
            
            if score >= 3:
                candidates.append((score, ep))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_match = candidates[0][1]
            return {
                "matched_episode_id": best_match["episode_id"],
                "learned_from_guidance": best_match["human_guidance"],
                "recommended_adaptation": best_match["adaptation"],
            }

        return None

    def get_summary(self) -> Dict[str, Any]:
        """Returns statistics on memory buffer for UI telemetry."""
        success_count = sum(1 for ep in self.episodes if ep.get("outcome") == "SUCCESS")
        return {
            "total_episodes": len(self.episodes),
            "successful_recoveries": success_count,
            "learned_patterns": len(set(ep.get("entity") for ep in self.episodes if ep.get("entity"))),
        }

