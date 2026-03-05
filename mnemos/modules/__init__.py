"""
mnemos/modules/__init__.py — Public API for the modules package.
"""

from .affective import AffectiveRouter
from .mutable_rag import MutableRAG
from .sleep import SleepDaemon
from .spreading import SpreadingActivation
from .surprisal import SurprisalGate

__all__ = [
    "SurprisalGate",
    "MutableRAG",
    "AffectiveRouter",
    "SleepDaemon",
    "SpreadingActivation",
]
