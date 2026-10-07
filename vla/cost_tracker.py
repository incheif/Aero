"""
LLM / Google VLA Cost & Token Tracking Module.
Maintains precise accounting of prompt tokens, multimodal image tokens,
completion tokens, and financial cost (USD) for Google Gemini VLA models.
"""

import time
from typing import Dict, List, Any, Optional

# Official Google Gemini API Pricing (USD per 1,000,000 tokens)
# Note: Gemini charges 258 tokens per standard image frame (<= 384x384 / 128x128)
PRICING_TABLE: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash": {
        "input_per_million": 0.075,
        "output_per_million": 0.300,
        "image_tokens": 258,
    },
    "gemini-2.0-flash": {
        "input_per_million": 0.100,
        "output_per_million": 0.400,
        "image_tokens": 258,
    },
    "gemini-1.5-pro": {
        "input_per_million": 1.250,
        "output_per_million": 5.000,
        "image_tokens": 258,
    },
}

DEFAULT_MODEL = "gemini-2.5-flash"

class VLACostTracker:
    """
    Session-wide Cost and Token Tracker for Google VLA / Gemini multimodal calls.
    """
    def __init__(self):
        self.total_calls: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0

        self.last_call_prompt_tokens: int = 0
        self.last_call_completion_tokens: int = 0
        self.last_call_cost_usd: float = 0.0
        self.last_call_model: str = DEFAULT_MODEL
        self.last_call_time: Optional[str] = None

        self.history: List[Dict[str, Any]] = []
        self.max_history_entries: int = 50

    def record_call(
        self,
        model_name: str,
        prompt_text: str,
        completion_text: str,
        has_image: bool = True,
        actual_input_tokens: Optional[int] = None,
        actual_output_tokens: Optional[int] = None,
        is_live_api: bool = False
    ) -> Dict[str, Any]:
        """
        Records a VLA inference call, calculates token metrics and cost in USD.
        """
        pricing = PRICING_TABLE.get(model_name, PRICING_TABLE[DEFAULT_MODEL])

        # 1. Determine input tokens
        if actual_input_tokens is not None and actual_input_tokens > 0:
            input_tokens = actual_input_tokens
        else:
            # Estimate text tokens (~4 chars per token)
            text_tokens = max(1, len(prompt_text) // 4)
            image_tokens = pricing["image_tokens"] if has_image else 0
            input_tokens = text_tokens + image_tokens

        # 2. Determine output tokens
        if actual_output_tokens is not None and actual_output_tokens > 0:
            output_tokens = actual_output_tokens
        else:
            output_tokens = max(1, len(completion_text) // 4)

        # 3. Calculate cost in USD
        input_cost = (input_tokens / 1_000_000.0) * pricing["input_per_million"]
        output_cost = (output_tokens / 1_000_000.0) * pricing["output_per_million"]
        call_cost = input_cost + output_cost

        # 4. Update session aggregations
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += (input_tokens + output_tokens)
        self.total_cost_usd += call_cost

        self.last_call_prompt_tokens = input_tokens
        self.last_call_completion_tokens = output_tokens
        self.last_call_cost_usd = call_cost
        self.last_call_model = model_name
        self.last_call_time = time.strftime("%H:%M:%S")

        # 5. Append to history
        entry = {
            "id": self.total_calls,
            "timestamp": self.last_call_time,
            "model": model_name,
            "is_live_api": is_live_api,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": round(call_cost, 6),
            "prompt_snippet": prompt_text[:45] + ("..." if len(prompt_text) > 45 else ""),
        }
        self.history.append(entry)
        if len(self.history) > self.max_history_entries:
            self.history.pop(0)

        return entry

    def reset(self):
        """Resets all session cost and token metrics to zero."""
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.last_call_prompt_tokens = 0
        self.last_call_completion_tokens = 0
        self.last_call_cost_usd = 0.0
        self.last_call_time = None
        self.history.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Structured dictionary for WebSocket streaming and REST endpoints."""
        avg_cost_per_call = (self.total_cost_usd / self.total_calls) if self.total_calls > 0 else 0.0
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "formatted_cost": f"${self.total_cost_usd:.5f}",
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "last_call": {
                "model": self.last_call_model,
                "cost_usd": round(self.last_call_cost_usd, 6),
                "formatted_cost": f"${self.last_call_cost_usd:.5f}",
                "input_tokens": self.last_call_prompt_tokens,
                "output_tokens": self.last_call_completion_tokens,
                "time": self.last_call_time,
            },
            "avg_cost_per_call_usd": round(avg_cost_per_call, 6),
            "history": self.history[-10:], # last 10 entries for UI
        }
