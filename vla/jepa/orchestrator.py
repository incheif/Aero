"""
vla/jepa/orchestrator.py

Re-exports the optimized, verified HierarchicalVLJEPAOrchestrator from vla.h_jepa.
Eliminates CEM jitter, avoids false-positive hold timeouts, and ensures real-time 60 FPS performance.
"""

from vla.h_jepa.orchestrator import HierarchicalVLJEPAOrchestrator

__all__ = ["HierarchicalVLJEPAOrchestrator"]
