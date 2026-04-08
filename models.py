"""
Data models for the Clinical Triage Environment.

Three core types following the OpenEnv contract:
  - TriageAction     (extends Action)      — what the agent can do
  - TriageObservation (extends Observation) — what the agent sees
  - TriageState       (extends State)       — hidden episode metadata

Aliased as ClinicalTriage* for backwards compatibility with the generated
client.py and __init__.py imports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator, model_validator

# ── Valid constants (mirrors data/specialties.py — duplicated to avoid
#    a cross-package import from models) ────────────────────────────────────────

_VALID_SPECIALTIES = {
    "cardiology", "neurology", "pulmonology", "gastroenterology",
    "orthopedics", "general_surgery", "internal_medicine", "psychiatry",
    "pediatrics", "emergency_medicine", "obstetrics_gynecology", "urology",
    "dermatology", "nephrology", "hematology", "endocrinology",
    "infectious_disease", "trauma_surgery", "vascular_surgery", "neurosurgery",
}

_VALID_ACTION_KINDS = {
    "ASSIGN_TRIAGE",            # Task 1: assign ESI level only
    "ASSIGN_TRIAGE_REFERRAL",   # Task 2: assign ESI + specialty
    "REQUEST_VITAL",            # Task 3: ask for an unrevealed vital
    "REQUEST_EXAM",             # Task 3: ask for an exam finding
    "ORDER_TEST",               # Task 3: order a diagnostic test
    "FINALIZE",                 # Task 3: commit final decision
}

_VALID_TASKS = {"triage_level", "triage_referral", "full_workup"}
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}


# ─────────────────────────────────────────────────────────────────────────────
# ACTION
# ─────────────────────────────────────────────────────────────────────────────

class TriageAction(Action):
    """
    Action the agent can take inside the Clinical Triage environment.

    Task 1 (triage_level):
        kind="ASSIGN_TRIAGE", triage_level=<1-5>

    Task 2 (triage_referral):
        kind="ASSIGN_TRIAGE_REFERRAL", triage_level=<1-5>, specialty=<str>

    Task 3 (full_workup) — multi-step:
        kind="REQUEST_VITAL",  request_target="hr"          # reveal a vital
        kind="REQUEST_EXAM",   request_target="lungs"       # reveal exam finding
        kind="ORDER_TEST",     request_target="ecg"         # order a test
        kind="FINALIZE",       triage_level=<1-5>, specialty=<str>, diagnosis=<str>
    """

    kind: Literal[
        "ASSIGN_TRIAGE",
        "ASSIGN_TRIAGE_REFERRAL",
        "REQUEST_VITAL",
        "REQUEST_EXAM",
        "ORDER_TEST",
        "FINALIZE",
    ] = Field(..., description="Action type — determines which other fields are required")

    # For ASSIGN_TRIAGE / ASSIGN_TRIAGE_REFERRAL / FINALIZE
    triage_level: Optional[int] = Field(
        default=None,
        ge=1, le=5,
        description="ESI triage level (1=most urgent … 5=least urgent)",
    )

    # For ASSIGN_TRIAGE_REFERRAL / FINALIZE
    specialty: Optional[str] = Field(
        default=None,
        description="Referral specialty (must be a member of SPECIALTIES)",
    )

    # For FINALIZE
    diagnosis: Optional[str] = Field(
        default=None,
        description="Working diagnosis string (free text, e.g. 'stemi')",
    )

    # For REQUEST_VITAL / REQUEST_EXAM / ORDER_TEST
    request_target: Optional[str] = Field(
        default=None,
        description="Name of the vital / exam finding / test to reveal",
    )

    # ── Validators ─────────────────────────────────────────────────────────────

    @field_validator("specialty")
    @classmethod
    def _validate_specialty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _VALID_SPECIALTIES:
            raise ValueError(
                f"specialty {v!r} is not valid. Choose from: {sorted(_VALID_SPECIALTIES)}"
            )
        return v

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> "TriageAction":
        kind = self.kind
        if kind == "ASSIGN_TRIAGE":
            if self.triage_level is None:
                raise ValueError("ASSIGN_TRIAGE requires triage_level")
        elif kind == "ASSIGN_TRIAGE_REFERRAL":
            if self.triage_level is None:
                raise ValueError("ASSIGN_TRIAGE_REFERRAL requires triage_level")
            if self.specialty is None:
                raise ValueError("ASSIGN_TRIAGE_REFERRAL requires specialty")
        elif kind in ("REQUEST_VITAL", "REQUEST_EXAM", "ORDER_TEST"):
            if self.request_target is None:
                raise ValueError(f"{kind} requires request_target")
        elif kind == "FINALIZE":
            if self.triage_level is None:
                raise ValueError("FINALIZE requires triage_level")
            if self.specialty is None:
                raise ValueError("FINALIZE requires specialty")
            if self.diagnosis is None:
                raise ValueError("FINALIZE requires diagnosis")
        return self

    def __str__(self) -> str:
        """Compact string for [STEP] log lines."""
        parts = [f"kind={self.kind}"]
        if self.triage_level is not None:
            parts.append(f"esi={self.triage_level}")
        if self.specialty is not None:
            parts.append(f"spec={self.specialty}")
        if self.diagnosis is not None:
            parts.append(f"dx={self.diagnosis}")
        if self.request_target is not None:
            parts.append(f"target={self.request_target}")
        return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION
# ─────────────────────────────────────────────────────────────────────────────

class TriageObservation(Observation):
    """
    What the agent sees at each step.

    Always visible:
        case_id, task_name, age, sex, arrival_mode,
        chief_complaint, history_present_illness,
        past_medical_history, medications, allergies

    Task 1 & 2 — full vitals exposed on reset.
    Task 3     — vitals start as None; revealed one-by-one via REQUEST_VITAL.
    Exam findings and test results start hidden and are revealed on request.
    """

    # ── Episode metadata ───────────────────────────────────────────────────────
    case_id: str = Field(default="", description="Unique ID of the sampled patient case")
    task_name: str = Field(
        default="",
        description="Active task: 'triage_level' | 'triage_referral' | 'full_workup'",
    )
    steps_taken: int = Field(default=0, description="Number of steps consumed so far")
    max_steps: int = Field(default=1, description="Maximum allowed steps for this task")
    tests_ordered: int = Field(default=0, description="Number of diagnostic tests ordered")
    cost_incurred: float = Field(default=0.0, description="Cumulative resource cost (tests)")
    message: str = Field(default="", description="Feedback / hint from the environment")
    available_actions: List[str] = Field(
        default_factory=list,
        description="Valid action kinds for the current step",
    )

    # ── Demographics (always visible) ─────────────────────────────────────────
    age: int = Field(default=0, description="Patient age in years")
    sex: str = Field(default="", description="Patient sex (M / F)")
    arrival_mode: str = Field(default="", description="How the patient arrived")

    # ── Presentation (always visible) ──────────────────────────────────────────
    chief_complaint: str = Field(default="", description="Chief complaint in patient's words")
    history_present_illness: str = Field(
        default="", description="Narrative history of the presenting illness"
    )
    past_medical_history: List[str] = Field(
        default_factory=list, description="Known past medical conditions"
    )
    medications: List[str] = Field(
        default_factory=list, description="Current medications"
    )
    allergies: List[str] = Field(default_factory=list, description="Known allergies")

    # ── Vitals (None if hidden/not-yet-requested in Task 3) ────────────────────
    vitals: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description=(
            "Vital signs. Keys: hr, sbp, dbp, rr, spo2, temp, pain. "
            "Value is None when the vital has not yet been revealed."
        ),
    )

    # ── Task-3 revealed findings (empty until REQUEST_EXAM) ────────────────────
    revealed_exam_findings: Dict[str, str] = Field(
        default_factory=dict,
        description="Exam findings revealed so far (Task 3 only)",
    )

    # ── Task-3 revealed test results (empty until ORDER_TEST) ─────────────────
    revealed_test_results: Dict[str, str] = Field(
        default_factory=dict,
        description="Diagnostic test results revealed so far (Task 3 only)",
    )

    # ── Convenience helpers ────────────────────────────────────────────────────

    def summary(self) -> str:
        """One-line patient summary for LLM prompt building."""
        return (
            f"{self.age}yo {self.sex}, arrived by {self.arrival_mode}. "
            f"CC: {self.chief_complaint}"
        )

    def vitals_str(self) -> str:
        """Human-readable vitals for prompt building."""
        labels = {
            "hr": "HR", "sbp": "SBP", "dbp": "DBP",
            "rr": "RR", "spo2": "SpO2", "temp": "Temp", "pain": "Pain",
        }
        parts = []
        for k, label in labels.items():
            v = self.vitals.get(k)
            parts.append(f"{label}={'?' if v is None else v}")
        return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# STATE  (hidden from agent; used by graders)
# ─────────────────────────────────────────────────────────────────────────────

class TriageState(State):
    """
    Internal episode state — NOT exposed to the agent.

    Holds ground-truth labels used by the graders to compute rewards.
    """

    # Task metadata
    task_name: str = Field(default="", description="Active task name")
    difficulty: str = Field(default="", description="Case difficulty tier")
    case_id: str = Field(default="", description="ID of the sampled case")
    max_steps: int = Field(default=1, description="Step budget for this episode")

    # Grading ground truth
    true_esi_level: int = Field(default=0, description="Ground-truth ESI level (1-5)")
    true_specialty: str = Field(default="", description="Ground-truth referral specialty")
    true_diagnosis: str = Field(default="", description="Ground-truth diagnosis")
    true_diagnosis_category: str = Field(default="", description="Diagnosis category")

    # Task-3 cost tracking
    cumulative_cost: float = Field(default=0.0, description="Cumulative test resource cost")
    tests_ordered: int = Field(default=0, description="Number of tests ordered so far")

    # Episode status
    is_finalized: bool = Field(
        default=False, description="True once the agent has committed a final answer"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compat aliases (generated client + __init__ reference these names)
# ─────────────────────────────────────────────────────────────────────────────
ClinicalTriageAction = TriageAction
ClinicalTriageObservation = TriageObservation


__all__ = [
    "TriageAction",
    "TriageObservation",
    "TriageState",
    "ClinicalTriageAction",      # alias
    "ClinicalTriageObservation", # alias
    "_VALID_SPECIALTIES",
    "_VALID_ACTION_KINDS",
    "_VALID_TASKS",
    "_VALID_DIFFICULTIES",
]
