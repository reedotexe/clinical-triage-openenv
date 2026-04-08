"""
Clinical Triage Environment — Core Logic.

Implements the OpenEnv Environment interface for all 3 tasks:

  Task 1 — triage_level (Easy)
      Single-step: agent sees full vitals, assigns ESI 1–5.
      Max steps: 1.  Action: ASSIGN_TRIAGE.

  Task 2 — triage_referral (Medium)
      Single-step: agent sees full vitals, assigns ESI + referral specialty.
      Max steps: 1.  Action: ASSIGN_TRIAGE_REFERRAL.

  Task 3 — full_workup (Hard)
      Multi-step: agent starts with sparse info, requests vitals/exam/tests,
      then commits with FINALIZE.
      Max steps: 10.  Actions: REQUEST_VITAL | REQUEST_EXAM | ORDER_TEST | FINALIZE.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TriageAction, TriageObservation, TriageState
    from ..data import (
        get_cases,
        get_random_case,
        get_case_by_id,
        PatientCase,
        AVAILABLE_TESTS,
        is_danger_zone,
        compute_task1_reward,
        compute_task2_reward,
        compute_task3_reward,
    )
except ImportError:
    from models import TriageAction, TriageObservation, TriageState
    from data import (
        get_cases,
        get_random_case,
        get_case_by_id,
        PatientCase,
        AVAILABLE_TESTS,
        is_danger_zone,
        compute_task1_reward,
        compute_task2_reward,
        compute_task3_reward,
    )


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

_TASK_CONFIG = {
    "triage_level": {
        "difficulty": "easy",
        "max_steps": 1,
        "vitals_hidden": False,
        "valid_actions": ["ASSIGN_TRIAGE"],
    },
    "triage_referral": {
        "difficulty": "medium",
        "max_steps": 1,
        "vitals_hidden": False,
        "valid_actions": ["ASSIGN_TRIAGE_REFERRAL"],
    },
    "full_workup": {
        "difficulty": "hard",
        "max_steps": 10,
        "vitals_hidden": True,
        "valid_actions": ["REQUEST_VITAL", "REQUEST_EXAM", "ORDER_TEST", "FINALIZE"],
    },
}

_DEFAULT_TASK = "triage_level"

# Task 3: which vitals are hidden at start (agent must request these)
_HIDDEN_VITALS = {"hr", "sbp", "dbp", "rr", "spo2"}
# These are always revealed (temp and pain are obvious to observer)
_ALWAYS_VISIBLE_VITALS = {"temp", "pain"}


# Inlined grader fns removed — now imported from data.graders


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ClinicalTriageEnvironment(Environment):
    """
    Clinical Triage & Medical Decision-Making Environment.

    Three tasks of increasing complexity:
      - triage_level    (Task 1, easy)
      - triage_referral (Task 2, medium)
      - full_workup     (Task 3, hard)

    Supports concurrent WebSocket sessions (each session gets its own instance).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        # Episode state — reset() populates these
        self._state: TriageState = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            task_name="",
        )
        self._case: Optional[PatientCase] = None
        self._task_name: str = ""
        self._revealed_vitals: Dict[str, Optional[float]] = {}
        self._revealed_exam: Dict[str, str] = {}
        self._revealed_tests: Dict[str, str] = {}
        self._rng: random.Random = random.Random()

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, **kwargs: Any) -> TriageObservation:  # type: ignore[override]
        """
        Reset the environment and return the initial observation.

        Keyword args:
            task (str): One of "triage_level", "triage_referral", "full_workup".
                        Defaults to "triage_level".
            seed (int): Optional random seed for reproducibility.
            case_id (str): Optional — force a specific case (testing).
        """
        task = str(kwargs.get("task", _DEFAULT_TASK))
        if task not in _TASK_CONFIG:
            task = _DEFAULT_TASK

        seed = kwargs.get("seed")
        if seed is not None:
            self._rng.seed(int(seed))

        cfg = _TASK_CONFIG[task]

        # Sample (or force) a patient case
        case_id = kwargs.get("case_id")
        if case_id:
            try:
                case = get_case_by_id(str(case_id))
            except KeyError:
                case = get_random_case(cfg["difficulty"], self._rng)
        else:
            case = get_random_case(cfg["difficulty"], self._rng)

        self._case = case
        self._task_name = task
        self._revealed_exam = {}
        self._revealed_tests = {}

        # Vitals exposure
        if cfg["vitals_hidden"]:
            # Task 3: only temp and pain visible upfront
            self._revealed_vitals = {
                k: (v if k in _ALWAYS_VISIBLE_VITALS else None)
                for k, v in case.vitals.items()
            }
        else:
            # Task 1 & 2: all vitals visible
            self._revealed_vitals = dict(case.vitals)

        # Reset internal state
        self._state = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            task_name=task,
            difficulty=cfg["difficulty"],
            case_id=case.case_id,
            max_steps=cfg["max_steps"],
            true_esi_level=case.true_esi_level,
            true_specialty=case.true_specialty,
            true_diagnosis=case.true_diagnosis,
            true_diagnosis_category=case.true_diagnosis_category,
            cumulative_cost=0.0,
            tests_ordered=0,
            is_finalized=False,
        )

        return self._build_observation(
            done=False,
            reward=None,
            message=self._welcome_message(task, case),
        )

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: TriageAction, **kwargs: Any) -> TriageObservation:  # type: ignore[override]
        """Execute one agent action and return the resulting observation."""
        if self._case is None:
            return self._build_observation(
                done=True, reward=0.0,
                message="Environment not initialised — call reset() first.",
            )

        self._state.step_count += 1
        cfg = _TASK_CONFIG[self._task_name]
        kind = action.kind

        # ── Guard: already finished ──────────────────────────────────────────
        if self._state.is_finalized:
            return self._build_observation(
                done=True, reward=0.0,
                message="Episode already ended. Call reset() to start a new episode.",
            )

        # ── Guard: step budget exceeded ───────────────────────────────────────
        if self._state.step_count > cfg["max_steps"]:
            return self._auto_finalize(reason="Step budget exceeded — auto-finalising with penalty.")

        # ── Validate action kind for this task ───────────────────────────────
        if kind not in cfg["valid_actions"]:
            return self._build_observation(
                done=False, reward=-0.05,
                message=(
                    f"Action '{kind}' is not valid for task '{self._task_name}'. "
                    f"Valid actions: {cfg['valid_actions']}"
                ),
            )

        # ── Route to task handler ────────────────────────────────────────────
        if kind == "ASSIGN_TRIAGE":
            return self._handle_assign_triage(action)
        elif kind == "ASSIGN_TRIAGE_REFERRAL":
            return self._handle_assign_triage_referral(action)
        elif kind == "REQUEST_VITAL":
            return self._handle_request_vital(action)
        elif kind == "REQUEST_EXAM":
            return self._handle_request_exam(action)
        elif kind == "ORDER_TEST":
            return self._handle_order_test(action)
        elif kind == "FINALIZE":
            return self._handle_finalize(action)
        else:
            return self._build_observation(
                done=False, reward=-0.05,
                message=f"Unknown action kind: {kind!r}",
            )

    # ── Task 1 handler ────────────────────────────────────────────────────────

    def _handle_assign_triage(self, action: TriageAction) -> TriageObservation:
        case = self._case
        predicted = action.triage_level
        true = case.true_esi_level
        reward = compute_task1_reward(predicted, true, self._case.red_flags)
        self._state.is_finalized = True

        if predicted == true:
            msg = f"Correct! ESI {true} — {self._esi_description(true)}."
        elif predicted < true:
            msg = (
                f"Over-triaged: you assigned ESI {predicted}, true level is ESI {true}. "
                f"Partial credit: {reward:.2f}."
            )
        else:
            msg = (
                f"⚠ Under-triaged: you assigned ESI {predicted}, true level is ESI {true}. "
                f"Under-triage is dangerous! Score: {reward:.2f}."
            )

        return self._build_observation(done=True, reward=reward, message=msg)

    # ── Task 2 handler ────────────────────────────────────────────────────────

    def _handle_assign_triage_referral(self, action: TriageAction) -> TriageObservation:
        case = self._case
        pred_esi = action.triage_level
        pred_spec = action.specialty
        true_esi = case.true_esi_level
        true_spec = case.true_specialty

        reward = compute_task2_reward(pred_esi, true_esi, pred_spec, true_spec, self._case.red_flags)
        self._state.is_finalized = True

        esi_ok = "✓" if pred_esi == true_esi else f"✗ (true={true_esi})"
        spec_ok = "✓" if pred_spec == true_spec else f"✗ (true={true_spec})"
        msg = (
            f"Result — ESI: {pred_esi} {esi_ok}  |  "
            f"Specialty: {pred_spec} {spec_ok}  |  Score: {reward:.2f}"
        )
        return self._build_observation(done=True, reward=reward, message=msg)

    # ── Task 3 handlers ───────────────────────────────────────────────────────

    def _handle_request_vital(self, action: TriageAction) -> TriageObservation:
        target = action.request_target.lower().strip()
        case = self._case

        if target not in case.vitals:
            known = list(case.vitals.keys())
            return self._build_observation(
                done=False, reward=0.0,
                message=f"Unknown vital '{target}'. Available vitals: {known}",
            )
        if self._revealed_vitals.get(target) is not None:
            return self._build_observation(
                done=False, reward=0.0,
                message=f"Vital '{target}' is already revealed: {self._revealed_vitals[target]}",
            )

        self._revealed_vitals[target] = case.vitals[target]
        return self._build_observation(
            done=False, reward=0.0,
            message=f"Vital revealed — {target.upper()}: {case.vitals[target]}",
        )

    def _handle_request_exam(self, action: TriageAction) -> TriageObservation:
        target = action.request_target.lower().strip()
        case = self._case

        if target not in case.exam_findings:
            available = list(case.exam_findings.keys())
            return self._build_observation(
                done=False, reward=0.0,
                message=f"No exam finding for '{target}'. Available: {available}",
            )
        if target in self._revealed_exam:
            return self._build_observation(
                done=False, reward=0.0,
                message=f"Exam finding '{target}' already revealed: {self._revealed_exam[target]}",
            )

        self._revealed_exam[target] = case.exam_findings[target]
        return self._build_observation(
            done=False, reward=0.0,
            message=f"Exam finding — {target}: {case.exam_findings[target]}",
        )

    def _handle_order_test(self, action: TriageAction) -> TriageObservation:
        target = action.request_target.lower().strip()
        case = self._case

        if target not in case.available_tests:
            available = list(case.available_tests.keys())
            return self._build_observation(
                done=False, reward=0.0,
                message=f"Test '{target}' not available for this patient. Available: {available}",
            )
        if target in self._revealed_tests:
            return self._build_observation(
                done=False, reward=0.0,
                message=f"Test '{target}' already ordered: {self._revealed_tests[target]}",
            )

        # Increment cost
        test_cost = AVAILABLE_TESTS.get(target, {}).get("cost", 1)
        self._state.cumulative_cost += test_cost
        self._state.tests_ordered += 1
        self._revealed_tests[target] = case.available_tests[target]

        return self._build_observation(
            done=False, reward=0.0,
            message=f"Test result — {target}: {case.available_tests[target]}  [cost +{test_cost}]",
        )

    def _handle_finalize(self, action: TriageAction) -> TriageObservation:
        case = self._case
        pred_esi  = action.triage_level
        pred_spec = action.specialty
        pred_dx   = action.diagnosis

        reward = compute_task3_reward(
            predicted_esi=pred_esi,        true_esi=case.true_esi_level,
            predicted_specialty=pred_spec, true_specialty=case.true_specialty,
            predicted_diagnosis=pred_dx,   true_diagnosis=case.true_diagnosis,
            true_diagnosis_category=case.true_diagnosis_category,
            tests_ordered=self._state.tests_ordered,
            steps_taken=self._state.step_count,
            max_steps=self._state.max_steps,
            red_flags=case.red_flags,
        )
        self._state.is_finalized = True

        esi_ok  = "✓" if pred_esi == case.true_esi_level else f"✗ (true={case.true_esi_level})"
        spec_ok = "✓" if pred_spec == case.true_specialty else f"✗ (true={case.true_specialty})"
        dx_ok   = "✓" if pred_dx.lower().replace(" ","_") == case.true_diagnosis else f"✗ (true={case.true_diagnosis})"
        msg = (
            f"FINALIZED — ESI: {pred_esi} {esi_ok}  |  "
            f"Specialty: {pred_spec} {spec_ok}  |  "
            f"Diagnosis: {pred_dx} {dx_ok}  |  "
            f"Tests: {self._state.tests_ordered}  |  "
            f"Score: {reward:.2f}"
        )
        return self._build_observation(done=True, reward=reward, message=msg)

    def _auto_finalize(self, reason: str) -> TriageObservation:
        """Called when step budget is exceeded — scores whatever was revealed."""
        self._state.is_finalized = True
        return self._build_observation(
            done=True, reward=0.0,
            message=f"{reason} Score: 0.00 (no answer committed in time).",
        )

    # ── state property ────────────────────────────────────────────────────────

    @property
    def state(self) -> TriageState:
        return self._state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_observation(
        self,
        done: bool,
        reward: Optional[float],
        message: str = "",
    ) -> TriageObservation:
        """Construct a TriageObservation from current episode state."""
        case = self._case
        cfg = _TASK_CONFIG.get(self._task_name, _TASK_CONFIG[_DEFAULT_TASK])

        if case is None:
            return TriageObservation(
                done=done, reward=reward, message=message,
                available_actions=list(cfg["valid_actions"]),
            )

        return TriageObservation(
            # Observation base fields
            done=done,
            reward=reward,
            # Episode metadata
            case_id=case.case_id,
            task_name=self._task_name,
            steps_taken=self._state.step_count,
            max_steps=self._state.max_steps,
            tests_ordered=self._state.tests_ordered,
            cost_incurred=self._state.cumulative_cost,
            message=message,
            available_actions=list(cfg["valid_actions"]) if not done else [],
            # Demographics
            age=case.age,
            sex=case.sex,
            arrival_mode=case.arrival_mode,
            # Presentation
            chief_complaint=case.chief_complaint,
            history_present_illness=case.history_present_illness,
            past_medical_history=list(case.past_medical_history),
            medications=list(case.medications),
            allergies=list(case.allergies),
            # Vitals (may include None for hidden values in Task 3)
            vitals=dict(self._revealed_vitals),
            # Task-3 revealed data
            revealed_exam_findings=dict(self._revealed_exam),
            revealed_test_results=dict(self._revealed_tests),
        )

    @staticmethod
    def _welcome_message(task: str, case: PatientCase) -> str:
        if task == "triage_level":
            return (
                "Task 1 — Triage Level: Review the patient and assign an ESI level (1–5). "
                "Action: ASSIGN_TRIAGE with triage_level."
            )
        elif task == "triage_referral":
            return (
                "Task 2 — Triage + Referral: Assign ESI level AND the appropriate referral specialty. "
                "Action: ASSIGN_TRIAGE_REFERRAL with triage_level and specialty."
            )
        else:
            return (
                "Task 3 — Full Workup: You have limited information. "
                "Request vitals (REQUEST_VITAL), exam findings (REQUEST_EXAM), "
                "or order tests (ORDER_TEST) before committing with FINALIZE. "
                f"Budget: {_TASK_CONFIG['full_workup']['max_steps']} steps."
            )

    @staticmethod
    def _esi_description(level: int) -> str:
        return {
            1: "Immediate life-saving intervention required",
            2: "High-risk / danger-zone vitals",
            3: "Stable, multiple resources required",
            4: "Stable, one resource required",
            5: "No resources required",
        }.get(level, f"ESI {level}")
