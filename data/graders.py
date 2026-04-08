"""
Deterministic Graders for the Clinical Triage Environment.

All grader functions return a score in [0.0, 1.0].

Composite rewards per task:
  Task 1: 100% ESI accuracy
  Task 2: 60% ESI + 40% specialty
  Task 3: 35% ESI + 25% specialty + 25% diagnosis + 15% efficiency
"""

from __future__ import annotations

from typing import Optional

from .specialties import specialty_distance


# ─────────────────────────────────────────────────────────────────────────────
# 1. ESI Triage Level Grader
# ─────────────────────────────────────────────────────────────────────────────

def grade_triage_level(predicted: int, true: int) -> float:
    """
    Asymmetric ESI grading.

    Under-triage (predicted > true) is clinically dangerous — a high-acuity
    patient sent to the waiting room can deteriorate or die.  Therefore
    under-triage is penalised more heavily than over-triage.

    Score table:
        Exact match              → 1.0
        Over-triage by 1 level   → 0.5
        Over-triage by 2+ levels → 0.25
        Under-triage by 1 level  → 0.3
        Under-triage by 2+ levels→ 0.0

    Args:
        predicted: Agent's assigned ESI level (1–5).
        true:      Ground-truth ESI level (1–5).

    Returns:
        Float score in [0.0, 1.0].
    """
    if not (1 <= predicted <= 5) or not (1 <= true <= 5):
        return 0.0
    if predicted == true:
        return 1.0
    diff = predicted - true   # positive → under-triage, negative → over-triage
    if diff > 0:              # under-triage (more dangerous)
        return 0.3 if diff == 1 else 0.0
    else:                     # over-triage
        return 0.5 if abs(diff) == 1 else 0.25


# ─────────────────────────────────────────────────────────────────────────────
# 2. Referral Specialty Grader
# ─────────────────────────────────────────────────────────────────────────────

def grade_referral(predicted: str, true: str) -> float:
    """
    Specialty referral grading with clinical adjacency partial credit.

    Score:
        Exact match              → 1.0
        Clinically adjacent spec → 0.5
        Unrelated specialty      → 0.0

    Args:
        predicted: Agent's predicted specialty string.
        true:      Ground-truth specialty string.

    Returns:
        Float score in [0.0, 1.0].
    """
    if not predicted or not true:
        return 0.0
    return specialty_distance(predicted.lower().strip(), true.lower().strip())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Diagnosis Grader
# ─────────────────────────────────────────────────────────────────────────────

def grade_diagnosis(
    predicted: str,
    true: str,
    true_category: str,
) -> float:
    """
    Working-diagnosis grading with keyword and category partial credit.

    Score:
        Exact string match (normalised)       → 1.0
        Overlapping keywords OR category hit  → 0.5
        Completely wrong                      → 0.0

    Args:
        predicted:      Agent's working diagnosis (free text, e.g. "stemi").
        true:           Ground-truth diagnosis string.
        true_category:  Ground-truth category (e.g. "cardiac").

    Returns:
        Float score in [0.0, 1.0].
    """
    if not predicted or not true:
        return 0.0

    p = predicted.lower().replace(" ", "_").replace("-", "_")
    t = true.lower().replace(" ", "_").replace("-", "_")
    tc = true_category.lower()

    if p == t:
        return 1.0

    # Keyword overlap (shared tokens between predicted and true)
    p_tokens = set(p.split("_"))
    t_tokens = set(t.split("_"))
    _STOP = {"the", "and", "of", "with", "in", "on", "a", "an"}
    overlap = (p_tokens - _STOP) & (t_tokens - _STOP)

    if overlap or tc in p:
        return 0.5

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Efficiency Grader  (Task 3 only)
# ─────────────────────────────────────────────────────────────────────────────

def grade_efficiency(
    tests_ordered: int,
    steps_taken: int,
    max_steps: int,
    optimal_tests: int = 3,
    optimal_steps: int = 6,
) -> float:
    """
    Reward agents who reach a decision efficiently.

    Penalties:
        Each test beyond optimal_tests  → -0.10
        Each step beyond optimal_steps  → -0.05

    Args:
        tests_ordered:  Number of diagnostic tests the agent ordered.
        steps_taken:    Total steps consumed in the episode.
        max_steps:      Episode step budget.
        optimal_tests:  Target maximum tests (default 3).
        optimal_steps:  Target maximum steps (default 6).

    Returns:
        Float score in [0.0, 1.0].
    """
    test_penalty = max(0, tests_ordered - optimal_tests) * 0.10
    step_penalty = max(0, steps_taken - optimal_steps) * 0.05
    return round(max(0.0, 1.0 - test_penalty - step_penalty), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Red-Flag Safety Check  (bonus/penalty modifier)
# ─────────────────────────────────────────────────────────────────────────────

def red_flag_modifier(
    predicted_esi: int,
    true_esi: int,
    red_flags: list,
) -> float:
    """
    Additional safety modifier applied on top of the base ESI score.

    If the true case is ESI 1 or 2 (safety-critical) AND the agent
    under-triages by 2+ levels, apply an extra penalty of -0.2.

    If the agent correctly identifies a high-acuity case with known red flags,
    apply a small bonus of +0.05 (capped so total stays ≤ 1.0).

    Args:
        predicted_esi: Agent's assigned ESI level.
        true_esi:      Ground-truth ESI level.
        red_flags:     List of red flag strings for this case.

    Returns:
        Float modifier in [-0.2, +0.05] to add to the base ESI score.
    """
    if not red_flags:
        return 0.0

    is_critical = true_esi <= 2
    under_triage_severe = (predicted_esi - true_esi) >= 2

    if is_critical and under_triage_severe:
        return -0.20   # dangerous miss on a critical patient

    if is_critical and predicted_esi == true_esi:
        return +0.05   # correctly caught a critical patient

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Score Clamping  (hackathon requires strictly open interval (0, 1))
# ─────────────────────────────────────────────────────────────────────────────

_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def _clamp(score: float) -> float:
    """Clamp to strictly open interval (0, 1) as required by the hackathon validator."""
    return round(max(_SCORE_MIN, min(_SCORE_MAX, score)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Composite Reward Functions (one per task)
# ─────────────────────────────────────────────────────────────────────────────

def compute_task1_reward(
    predicted_esi: int,
    true_esi: int,
    red_flags: Optional[list] = None,
) -> float:
    """
    Task 1 — Triage Level Only.

    Weights:
        100% ESI accuracy + red-flag safety modifier

    Returns: score in [0.0, 1.0]
    """
    base = grade_triage_level(predicted_esi, true_esi)
    modifier = red_flag_modifier(predicted_esi, true_esi, red_flags or [])
    return _clamp(base + modifier)


def compute_task2_reward(
    predicted_esi: int,
    true_esi: int,
    predicted_specialty: str,
    true_specialty: str,
    red_flags: Optional[list] = None,
) -> float:
    """
    Task 2 — Triage Level + Referral Specialty.

    Weights:
        60% ESI accuracy
        40% specialty accuracy
        + red-flag safety modifier on the ESI component

    Returns: score in [0.0, 1.0]
    """
    esi_score  = grade_triage_level(predicted_esi, true_esi)
    spec_score = grade_referral(predicted_specialty, true_specialty)
    modifier   = red_flag_modifier(predicted_esi, true_esi, red_flags or [])

    raw = 0.60 * esi_score + 0.40 * spec_score + modifier
    return _clamp(raw)


def compute_task3_reward(
    predicted_esi: int,
    true_esi: int,
    predicted_specialty: str,
    true_specialty: str,
    predicted_diagnosis: str,
    true_diagnosis: str,
    true_diagnosis_category: str,
    tests_ordered: int,
    steps_taken: int,
    max_steps: int,
    red_flags: Optional[list] = None,
) -> float:
    """
    Task 3 — Full Workup (multi-step).

    Weights:
        35% ESI accuracy
        25% specialty accuracy
        25% diagnosis accuracy
        15% efficiency (tests + steps)
        + red-flag safety modifier on the ESI component

    Returns: score in [0.0, 1.0]
    """
    esi_score  = grade_triage_level(predicted_esi, true_esi)
    spec_score = grade_referral(predicted_specialty, true_specialty)
    dx_score   = grade_diagnosis(predicted_diagnosis, true_diagnosis, true_diagnosis_category)
    eff_score  = grade_efficiency(tests_ordered, steps_taken, max_steps)
    modifier   = red_flag_modifier(predicted_esi, true_esi, red_flags or [])

    raw = (
        0.35 * esi_score
        + 0.25 * spec_score
        + 0.25 * dx_score
        + 0.15 * eff_score
        + modifier
    )
    return _clamp(raw)
