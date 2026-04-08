"""Clinical Triage Environment — public API."""

from .client import ClinicalTriageEnv
from .models import (
    TriageAction,
    TriageObservation,
    TriageState,
    ClinicalTriageAction,       # alias
    ClinicalTriageObservation,  # alias
)

__all__ = [
    # Primary names
    "TriageAction",
    "TriageObservation",
    "TriageState",
    # Backwards-compat aliases (used by generated client)
    "ClinicalTriageAction",
    "ClinicalTriageObservation",
    # Client
    "ClinicalTriageEnv",
]
