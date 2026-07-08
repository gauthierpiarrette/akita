"""Akita: plan, validate, and cost encrypted computation before running it."""

from .embedder import HashingEmbedder, MiniLMEmbedder
from .memory import IntegrityError, Memory, MemoryServer
from .planner import PipelineSpec, Plan, PlanError, plan
from .runtime import run_column_scoring, run_matvec

__all__ = [
    "PipelineSpec",
    "Plan",
    "PlanError",
    "plan",
    "run_matvec",
    "run_column_scoring",
    "IntegrityError",
    "Memory",
    "MemoryServer",
    "HashingEmbedder",
    "MiniLMEmbedder",
]

__version__ = "0.2.0"
