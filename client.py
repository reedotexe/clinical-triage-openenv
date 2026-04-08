"""
Clinical Triage Environment Client.

Wraps the running server over WebSocket for all three tasks:
  - Task 1: triage_level     (ASSIGN_TRIAGE)
  - Task 2: triage_referral  (ASSIGN_TRIAGE_REFERRAL)
  - Task 3: full_workup      (REQUEST_VITAL/EXAM/TEST → FINALIZE)

Usage:
    with ClinicalTriageEnv(base_url="http://localhost:8000") as client:
        obs = client.reset(task="triage_level", seed=42)
        result = client.step(TriageAction(kind="ASSIGN_TRIAGE", triage_level=2))
        print(result.observation.message, result.reward)

    # Task 3 multi-step workflow:
    with ClinicalTriageEnv(base_url="http://localhost:8000") as client:
        obs = client.reset(task="full_workup", seed=7)
        r = client.step(TriageAction(kind="REQUEST_VITAL", request_target="hr"))
        r = client.step(TriageAction(kind="ORDER_TEST",    request_target="ecg"))
        r = client.step(TriageAction(kind="FINALIZE",
                                     triage_level=2,
                                     specialty="cardiology",
                                     diagnosis="nstemi"))
        print(r.reward)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import TriageAction, TriageObservation, TriageState
except ImportError:
    from models import TriageAction, TriageObservation, TriageState  # type: ignore[no-redef]


class ClinicalTriageEnv(
    EnvClient[TriageAction, TriageObservation, TriageState]
):
    """
    WebSocket client for the Clinical Triage Environment server.

    Maintains a persistent connection, enabling efficient multi-step
    interactions with low latency across all three task tiers.
    """

    # ── Serialise action → JSON payload ──────────────────────────────────────

    def _step_payload(self, action: TriageAction) -> Dict[str, Any]:
        """Convert TriageAction to a JSON-serialisable dict for the server."""
        payload: Dict[str, Any] = {"kind": action.kind}

        if action.triage_level is not None:
            payload["triage_level"] = action.triage_level
        if action.specialty is not None:
            payload["specialty"] = action.specialty
        if action.diagnosis is not None:
            payload["diagnosis"] = action.diagnosis
        if action.request_target is not None:
            payload["request_target"] = action.request_target
        # Forward any extra metadata from the base Action
        if action.metadata:
            payload["metadata"] = action.metadata

        return payload

    # ── Deserialise server response → StepResult ─────────────────────────────

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TriageObservation]:
        """Parse the server's JSON response into a StepResult[TriageObservation]."""
        obs_data: Dict[str, Any] = payload.get("observation", {})

        observation = TriageObservation(
            # OpenEnv base fields
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),

            # Episode info
            case_id=obs_data.get("case_id", ""),
            task_name=obs_data.get("task_name", ""),
            steps_taken=obs_data.get("steps_taken", 0),
            max_steps=obs_data.get("max_steps", 1),
            tests_ordered=obs_data.get("tests_ordered", 0),
            cost_incurred=obs_data.get("cost_incurred", 0.0),
            message=obs_data.get("message", ""),
            available_actions=obs_data.get("available_actions", []),

            # Demographics
            age=obs_data.get("age", 0),
            sex=obs_data.get("sex", ""),
            arrival_mode=obs_data.get("arrival_mode", ""),

            # Presentation
            chief_complaint=obs_data.get("chief_complaint", ""),
            history_present_illness=obs_data.get("history_present_illness", ""),
            past_medical_history=obs_data.get("past_medical_history", []),
            medications=obs_data.get("medications", []),
            allergies=obs_data.get("allergies", []),

            # Vitals (may contain None values for hidden vitals in Task 3)
            vitals=obs_data.get("vitals", {}),

            # Task-3 progressive reveal
            revealed_exam_findings=obs_data.get("revealed_exam_findings", {}),
            revealed_test_results=obs_data.get("revealed_test_results", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    # ── Deserialise server state ──────────────────────────────────────────────

    def _parse_state(self, payload: Dict[str, Any]) -> TriageState:
        """Parse the server's /state response into a TriageState object."""
        return TriageState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", ""),
            difficulty=payload.get("difficulty", ""),
            case_id=payload.get("case_id", ""),
            max_steps=payload.get("max_steps", 1),
            true_esi_level=payload.get("true_esi_level", 0),
            true_specialty=payload.get("true_specialty", ""),
            true_diagnosis=payload.get("true_diagnosis", ""),
            true_diagnosis_category=payload.get("true_diagnosis_category", ""),
            cumulative_cost=payload.get("cumulative_cost", 0.0),
            tests_ordered=payload.get("tests_ordered", 0),
            is_finalized=payload.get("is_finalized", False),
        )

    # ── Convenience helpers for multi-step Task 3 ─────────────────────────────

    def request_vital(self, vital: str) -> StepResult[TriageObservation]:
        """Convenience shorthand for REQUEST_VITAL."""
        return self.step(TriageAction(kind="REQUEST_VITAL", request_target=vital))

    def request_exam(self, finding: str) -> StepResult[TriageObservation]:
        """Convenience shorthand for REQUEST_EXAM."""
        return self.step(TriageAction(kind="REQUEST_EXAM", request_target=finding))

    def order_test(self, test: str) -> StepResult[TriageObservation]:
        """Convenience shorthand for ORDER_TEST."""
        return self.step(TriageAction(kind="ORDER_TEST", request_target=test))

    def finalize(
        self,
        triage_level: int,
        specialty: str,
        diagnosis: str,
    ) -> StepResult[TriageObservation]:
        """Convenience shorthand for FINALIZE."""
        return self.step(TriageAction(
            kind="FINALIZE",
            triage_level=triage_level,
            specialty=specialty,
            diagnosis=diagnosis,
        ))

    def assign_triage(self, triage_level: int) -> StepResult[TriageObservation]:
        """Convenience shorthand for Task 1 ASSIGN_TRIAGE."""
        return self.step(TriageAction(kind="ASSIGN_TRIAGE", triage_level=triage_level))

    def assign_triage_referral(
        self,
        triage_level: int,
        specialty: str,
    ) -> StepResult[TriageObservation]:
        """Convenience shorthand for Task 2 ASSIGN_TRIAGE_REFERRAL."""
        return self.step(TriageAction(
            kind="ASSIGN_TRIAGE_REFERRAL",
            triage_level=triage_level,
            specialty=specialty,
        ))
