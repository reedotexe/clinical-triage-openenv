"""
Unit tests for data/graders.py — all grader functions.

Run from project root:
    cd clinical_triage
    python -m pytest tests/test_graders.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from data.graders import (
    grade_triage_level,
    grade_referral,
    grade_diagnosis,
    grade_efficiency,
    red_flag_modifier,
    compute_task1_reward,
    compute_task2_reward,
    compute_task3_reward,
)


# ──────────────────────────────────────────────────────────────────────────────
# grade_triage_level
# ──────────────────────────────────────────────────────────────────────────────

class TestGradeTriageLevel:
    def test_exact_match_each_level(self):
        for esi in range(1, 6):
            assert grade_triage_level(esi, esi) == 1.0

    def test_under_triage_by_1(self):
        # predicted=3, true=2 → under by 1
        score = grade_triage_level(3, 2)
        assert score == 0.3

    def test_under_triage_by_2_plus(self):
        # predicted=3, true=1 → under by 2
        score = grade_triage_level(3, 1)
        assert score == 0.0

    def test_over_triage_by_1(self):
        # predicted=2, true=3 → over by 1
        score = grade_triage_level(2, 3)
        assert score == 0.5

    def test_over_triage_by_2(self):
        # predicted=1, true=3 → over by 2
        score = grade_triage_level(1, 3)
        assert score == 0.25

    def test_under_triage_worse_than_over(self):
        # under-triage by 1 (0.3) should score worse than over-triage by 1 (0.5)
        assert grade_triage_level(3, 2) < grade_triage_level(1, 2)

    def test_out_of_bounds_returns_zero(self):
        assert grade_triage_level(0, 3) == 0.0
        assert grade_triage_level(3, 6) == 0.0

    def test_returns_float(self):
        assert isinstance(grade_triage_level(2, 2), float)


# ──────────────────────────────────────────────────────────────────────────────
# grade_referral
# ──────────────────────────────────────────────────────────────────────────────

class TestGradeReferral:
    def test_exact_match(self):
        assert grade_referral("cardiology", "cardiology") == 1.0

    def test_case_insensitive(self):
        assert grade_referral("Cardiology", "cardiology") == 1.0

    def test_adjacent_specialty_gets_partial_credit(self):
        # cardiology and internal_medicine are adjacent
        score = grade_referral("internal_medicine", "cardiology")
        assert 0.0 < score < 1.0

    def test_unrelated_specialty_scores_zero(self):
        score = grade_referral("dermatology", "neurosurgery")
        assert score == 0.0

    def test_empty_strings_return_zero(self):
        assert grade_referral("", "cardiology") == 0.0
        assert grade_referral("cardiology", "") == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# grade_diagnosis
# ──────────────────────────────────────────────────────────────────────────────

class TestGradeDiagnosis:
    def test_exact_match(self):
        assert grade_diagnosis("stemi", "stemi", "cardiac") == 1.0

    def test_normalises_spaces_and_hyphens(self):
        assert grade_diagnosis("acute myocardial infarction", "acute_myocardial_infarction", "cardiac") == 1.0

    def test_keyword_overlap_gets_half_credit(self):
        # 'myocardial' token overlaps
        score = grade_diagnosis("myocardial_contusion", "myocardial_infarction", "cardiac")
        assert score == 0.5

    def test_category_match_gets_half_credit(self):
        # 'cardiac' in predicted
        score = grade_diagnosis("cardiac_arrest", "stemi", "cardiac")
        assert score == 0.5

    def test_completely_wrong_returns_zero(self):
        assert grade_diagnosis("appendicitis", "stemi", "cardiac") == 0.0

    def test_empty_strings_return_zero(self):
        assert grade_diagnosis("", "stemi", "cardiac") == 0.0
        assert grade_diagnosis("stemi", "", "cardiac") == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# grade_efficiency
# ──────────────────────────────────────────────────────────────────────────────

class TestGradeEfficiency:
    def test_perfect_efficiency(self):
        # 3 tests, 6 steps → no penalty
        assert grade_efficiency(3, 6, 10) == 1.0

    def test_one_extra_test(self):
        score = grade_efficiency(4, 6, 10)
        assert score == pytest.approx(0.9, abs=1e-4)

    def test_one_extra_step(self):
        score = grade_efficiency(3, 7, 10)
        assert score == pytest.approx(0.95, abs=1e-4)

    def test_many_tests_and_steps(self):
        score = grade_efficiency(10, 15, 10)
        assert score == 0.0   # capped at 0

    def test_within_optimal_is_full_score(self):
        # Under optimal still 1.0
        assert grade_efficiency(1, 3, 10) == 1.0

    def test_always_non_negative(self):
        for t in range(0, 15):
            for s in range(0, 15):
                assert grade_efficiency(t, s, 10) >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# red_flag_modifier
# ──────────────────────────────────────────────────────────────────────────────

class TestRedFlagModifier:
    def test_no_red_flags_returns_zero(self):
        assert red_flag_modifier(3, 1, []) == 0.0

    def test_critical_case_severe_undertriage_gets_penalty(self):
        # true_esi=1, predicted=3 → under by 2 on critical case
        mod = red_flag_modifier(3, 1, ["sepsis", "high_lactate"])
        assert mod == -0.20

    def test_critical_case_exact_match_gets_bonus(self):
        mod = red_flag_modifier(1, 1, ["sepsis"])
        assert mod == +0.05

    def test_non_critical_case_no_modifier(self):
        # true_esi=3, even if under-triaged
        mod = red_flag_modifier(5, 3, ["some_flag"])
        assert mod == 0.0

    def test_critical_under_by_1_no_severe_penalty(self):
        # under by 1 but not by 2
        mod = red_flag_modifier(2, 1, ["flag"])
        # should NOT trigger the severe penalty
        assert mod != -0.20


# ──────────────────────────────────────────────────────────────────────────────
# Composite reward functions
# ──────────────────────────────────────────────────────────────────────────────

class TestCompositeRewards:

    # Task 1
    def test_task1_perfect(self):
        r = compute_task1_reward(2, 2)
        assert r == 1.0

    def test_task1_perfect_with_red_flags(self):
        r = compute_task1_reward(1, 1, ["sepsis"])
        assert r == 1.0  # capped at 1.0 even with +0.05 bonus

    def test_task1_critical_miss_gets_extra_penalty(self):
        r = compute_task1_reward(3, 1, ["sepsis"])
        # base=0.0 for under by 2, modifier=-0.20 → capped at 0.0
        assert r == 0.0

    def test_task1_score_in_range(self):
        for p in range(1, 6):
            for t in range(1, 6):
                r = compute_task1_reward(p, t)
                assert 0.0 <= r <= 1.0

    # Task 2
    def test_task2_perfect(self):
        r = compute_task2_reward(2, 2, "cardiology", "cardiology")
        assert r == 1.0

    def test_task2_wrong_specialty_partial(self):
        r = compute_task2_reward(2, 2, "internal_medicine", "cardiology")
        # ESI perfect (0.6*1), specialty partial (0.4*0.5)
        assert 0.6 <= r < 1.0

    def test_task2_score_in_range(self):
        for p in range(1, 6):
            for t in range(1, 6):
                r = compute_task2_reward(p, t, "cardiology", "cardiology")
                assert 0.0 <= r <= 1.0

    # Task 3
    def test_task3_perfect(self):
        r = compute_task3_reward(
            predicted_esi=2, true_esi=2,
            predicted_specialty="cardiology", true_specialty="cardiology",
            predicted_diagnosis="nstemi", true_diagnosis="nstemi",
            true_diagnosis_category="cardiac",
            tests_ordered=2, steps_taken=5, max_steps=10,
        )
        assert r == 1.0

    def test_task3_all_wrong(self):
        r = compute_task3_reward(
            predicted_esi=5, true_esi=1,
            predicted_specialty="dermatology", true_specialty="cardiology",
            predicted_diagnosis="sunburn", true_diagnosis="stemi",
            true_diagnosis_category="cardiac",
            tests_ordered=0, steps_taken=1, max_steps=10,
        )
        # ESI=0, spec=0, dx=0, eff=1.0 still → 0.15 floor
        assert 0.0 <= r <= 0.20

    def test_task3_score_always_bounded(self):
        import itertools
        for esi_p, esi_t in itertools.product(range(1, 6), range(1, 6)):
            r = compute_task3_reward(
                predicted_esi=esi_p, true_esi=esi_t,
                predicted_specialty="cardiology", true_specialty="cardiology",
                predicted_diagnosis="stemi", true_diagnosis="stemi",
                true_diagnosis_category="cardiac",
                tests_ordered=3, steps_taken=6, max_steps=10,
            )
            assert 0.0 <= r <= 1.0, f"Out of range: esi_p={esi_p}, esi_t={esi_t} → {r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
