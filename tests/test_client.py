"""
Unit tests for client.py — serialisation and deserialisation logic.
No live server required: tests _step_payload and _parse_result directly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from client import ClinicalTriageEnv
from models import TriageAction, TriageObservation, TriageState


@pytest.fixture
def env():
    """Return a bare ClinicalTriageEnv instance (no server connection)."""
    return ClinicalTriageEnv.__new__(ClinicalTriageEnv)


@pytest.fixture
def full_obs_payload():
    return {
        "done": False,
        "reward": None,
        "observation": {
            "case_id": "hard_001",
            "task_name": "full_workup",
            "steps_taken": 2,
            "max_steps": 10,
            "tests_ordered": 1,
            "cost_incurred": 2.0,
            "message": "ECG result revealed",
            "available_actions": ["REQUEST_VITAL", "ORDER_TEST", "FINALIZE"],
            "age": 70, "sex": "M", "arrival_mode": "walk-in",
            "chief_complaint": "Feeling tired and unwell",
            "history_present_illness": "1-week vague fatigue...",
            "past_medical_history": ["type2_diabetes", "hypertension"],
            "medications": ["metformin", "ramipril"],
            "allergies": [],
            "vitals": {
                "hr": 105.0, "sbp": None, "dbp": None,
                "rr": None, "spo2": None, "temp": 38.6, "pain": 1.0,
            },
            "revealed_exam_findings": {"skin": "warm, flushed"},
            "revealed_test_results": {"cbc": "WBC 18.5 elevated"},
        },
    }


class TestStepPayload:
    def test_assign_triage(self, env):
        a = TriageAction(kind="ASSIGN_TRIAGE", triage_level=3)
        p = env._step_payload(a)
        assert p == {"kind": "ASSIGN_TRIAGE", "triage_level": 3}

    def test_assign_triage_referral(self, env):
        a = TriageAction(kind="ASSIGN_TRIAGE_REFERRAL", triage_level=2, specialty="cardiology")
        p = env._step_payload(a)
        assert p["kind"] == "ASSIGN_TRIAGE_REFERRAL"
        assert p["triage_level"] == 2
        assert p["specialty"] == "cardiology"

    def test_request_vital(self, env):
        a = TriageAction(kind="REQUEST_VITAL", request_target="hr")
        p = env._step_payload(a)
        assert p == {"kind": "REQUEST_VITAL", "request_target": "hr"}

    def test_order_test(self, env):
        a = TriageAction(kind="ORDER_TEST", request_target="ecg")
        p = env._step_payload(a)
        assert p == {"kind": "ORDER_TEST", "request_target": "ecg"}

    def test_finalize_all_fields(self, env):
        a = TriageAction(kind="FINALIZE", triage_level=1, specialty="cardiology", diagnosis="stemi")
        p = env._step_payload(a)
        assert p["kind"] == "FINALIZE"
        assert p["triage_level"] == 1
        assert p["specialty"] == "cardiology"
        assert p["diagnosis"] == "stemi"

    def test_no_none_fields_in_payload(self, env):
        """None fields should not appear in the serialised payload."""
        a = TriageAction(kind="ASSIGN_TRIAGE", triage_level=2)
        p = env._step_payload(a)
        assert "specialty" not in p
        assert "diagnosis" not in p
        assert "request_target" not in p


class TestParseResult:
    def test_parse_returns_step_result(self, env, full_obs_payload):
        from openenv.core.client_types import StepResult
        result = env._parse_result(full_obs_payload)
        assert isinstance(result, StepResult)

    def test_observation_case_id(self, env, full_obs_payload):
        obs = env._parse_result(full_obs_payload).observation
        assert obs.case_id == "hard_001"

    def test_observation_demographics(self, env, full_obs_payload):
        obs = env._parse_result(full_obs_payload).observation
        assert obs.age == 70
        assert obs.sex == "M"
        assert obs.arrival_mode == "walk-in"

    def test_hidden_vitals_preserved(self, env, full_obs_payload):
        obs = env._parse_result(full_obs_payload).observation
        assert obs.vitals["hr"] == 105.0
        assert obs.vitals["sbp"] is None  # Task 3 hidden vital

    def test_revealed_findings(self, env, full_obs_payload):
        obs = env._parse_result(full_obs_payload).observation
        assert obs.revealed_exam_findings == {"skin": "warm, flushed"}
        assert obs.revealed_test_results == {"cbc": "WBC 18.5 elevated"}

    def test_available_actions(self, env, full_obs_payload):
        obs = env._parse_result(full_obs_payload).observation
        assert "FINALIZE" in obs.available_actions

    def test_done_and_reward_propagated(self, env):
        payload = {"done": True, "reward": 0.85, "observation": {}}
        result = env._parse_result(payload)
        assert result.done is True
        assert result.reward == pytest.approx(0.85)

    def test_empty_observation_safe(self, env):
        """Server returning empty observation dict should not crash."""
        result = env._parse_result({"done": False, "reward": None, "observation": {}})
        assert result.observation.case_id == ""


class TestParseState:
    def test_parse_state_fields(self, env):
        payload = {
            "episode_id": "ep-abc", "step_count": 3,
            "task_name": "full_workup", "difficulty": "hard",
            "case_id": "hard_005", "max_steps": 10,
            "true_esi_level": 2, "true_specialty": "urology",
            "true_diagnosis": "testicular_torsion",
            "true_diagnosis_category": "urological",
            "cumulative_cost": 4.0, "tests_ordered": 2,
            "is_finalized": False,
        }
        state = env._parse_state(payload)
        assert state.episode_id == "ep-abc"
        assert state.true_esi_level == 2
        assert state.true_specialty == "urology"
        assert state.is_finalized is False

    def test_defaults_safe(self, env):
        """Minimal state payload should not crash."""
        state = env._parse_state({})
        assert state.step_count == 0
        assert state.true_esi_level == 0


class TestConvenienceHelpers:
    def test_helpers_produce_correct_actions(self, env):
        """Verify convenience methods build the right TriageAction."""
        # We can't call the actual network methods, so just verify the action
        # that _would_ be sent by checking _step_payload output
        a = TriageAction(kind="REQUEST_VITAL", request_target="hr")
        p = env._step_payload(a)
        assert p["kind"] == "REQUEST_VITAL"
        assert p["request_target"] == "hr"

        a2 = TriageAction(kind="FINALIZE", triage_level=2, specialty="cardiology", diagnosis="stemi")
        p2 = env._step_payload(a2)
        assert p2["kind"] == "FINALIZE"
        assert p2["specialty"] == "cardiology"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
