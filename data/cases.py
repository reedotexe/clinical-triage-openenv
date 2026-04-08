"""
Synthetic Patient Case Bank for Clinical Triage Environment.

100 handcrafted cases across 3 difficulty levels:
  - 30 Easy  : obvious presentations (clear ESI, direct triage)
  - 30 Medium: require reasoning for BOTH triage + specialty
  - 40 Hard  : sparse initial info; agent must request vitals/tests (Task 3)

All cases are purely synthetic — no real patient data.
"""

from __future__ import annotations
import random
from typing import Dict, List, Literal, Optional

from .cases_easy import PatientCase, EASY_CASES as _EASY_BASE
from .cases_medium import MEDIUM_CASES as _MEDIUM_BASE
from .cases_hard import HARD_CASES as _HARD_BASE
from .cases_extra import EXTRA_EASY_CASES, EXTRA_MEDIUM_CASES, EXTRA_HARD_CASES

# Merged difficulty pools (30 / 30 / 40 = 100 total)
EASY_CASES: List[PatientCase] = _EASY_BASE + EXTRA_EASY_CASES
MEDIUM_CASES: List[PatientCase] = _MEDIUM_BASE + EXTRA_MEDIUM_CASES
HARD_CASES: List[PatientCase] = _HARD_BASE + EXTRA_HARD_CASES

# Re-export PatientCase so importers only need `from data.cases import ...`
__all__ = [
    "PatientCase",
    "EASY_CASES",
    "MEDIUM_CASES",
    "HARD_CASES",
    "ALL_CASES",
    "get_cases",
    "get_case_by_id",
    "get_random_case",
    "validate_cases",
]

# ── Combined case pool ────────────────────────────────────────────────────────

ALL_CASES: List[PatientCase] = EASY_CASES + MEDIUM_CASES + HARD_CASES

_CASE_BY_ID: Dict[str, PatientCase] = {c.case_id: c for c in ALL_CASES}


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_cases(difficulty: Literal["easy", "medium", "hard"]) -> List[PatientCase]:
    """Return all cases for the given difficulty tier."""
    mapping = {"easy": EASY_CASES, "medium": MEDIUM_CASES, "hard": HARD_CASES}
    if difficulty not in mapping:
        raise ValueError(f"difficulty must be one of {list(mapping.keys())}, got {difficulty!r}")
    return list(mapping[difficulty])


def get_case_by_id(case_id: str) -> PatientCase:
    """Return a specific case by ID. Raises KeyError if not found."""
    if case_id not in _CASE_BY_ID:
        raise KeyError(f"No case with id={case_id!r}. Valid IDs: {sorted(_CASE_BY_ID)}")
    return _CASE_BY_ID[case_id]


def get_random_case(
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None,
    rng: Optional[random.Random] = None,
) -> PatientCase:
    """
    Return a random case.

    Args:
        difficulty: If given, sample only from that tier.
        rng: Optional seeded Random for reproducibility.
    """
    pool = get_cases(difficulty) if difficulty else ALL_CASES
    return (rng or random).choice(pool)


# ── Validation ─────────────────────────────────────────────────────────────────

_VALID_DISPOSITIONS = {"admit_icu", "admit_ward", "discharge", "cath_lab", "or"}
_VALID_ESI = {1, 2, 3, 4, 5}
_VALID_SEX = {"M", "F"}
_VALID_ARRIVAL = {"walk-in", "ambulance", "wheelchair"}

# Physiological plausibility bounds
_VITAL_BOUNDS = {
    "hr":   (0, 300),
    "sbp":  (0, 300),
    "dbp":  (0, 250),
    "rr":   (0, 70),
    "spo2": (0, 100),
    "temp": (30.0, 44.0),
    "pain": (0, 10),
}


def validate_cases(cases: Optional[List[PatientCase]] = None) -> Dict[str, List[str]]:
    """
    Validate all (or provided) cases against structural and clinical rules.

    Returns a dict mapping case_id → list of error strings.
    An empty dict means all cases are valid.
    """
    if cases is None:
        cases = ALL_CASES

    errors: Dict[str, List[str]] = {}

    ids_seen = set()
    for case in cases:
        errs: List[str] = []

        # Structural checks
        if case.case_id in ids_seen:
            errs.append(f"Duplicate case_id: {case.case_id!r}")
        ids_seen.add(case.case_id)

        if case.true_esi_level not in _VALID_ESI:
            errs.append(f"Invalid ESI level: {case.true_esi_level}")

        if case.sex not in _VALID_SEX:
            errs.append(f"Invalid sex: {case.sex!r}")

        if case.arrival_mode not in _VALID_ARRIVAL:
            errs.append(f"Invalid arrival_mode: {case.arrival_mode!r}")

        if case.true_disposition not in _VALID_DISPOSITIONS:
            errs.append(f"Invalid disposition: {case.true_disposition!r}")

        if not (0 <= case.age <= 120):
            errs.append(f"Implausible age: {case.age}")

        # Vital sign bounds
        for key, (lo, hi) in _VITAL_BOUNDS.items():
            val = case.vitals.get(key)
            if val is not None and not (lo <= val <= hi):
                errs.append(f"Vital {key}={val} out of bounds [{lo}, {hi}]")

        # ESI 1 cases should have at least one red flag
        if case.true_esi_level == 1 and not case.red_flags:
            errs.append("ESI-1 case should list at least one red flag")

        if errs:
            errors[case.case_id] = errs

    return errors


# ── Quick stats on load ────────────────────────────────────────────────────────

def _case_stats() -> str:
    from collections import Counter
    esi_counts = Counter(c.true_esi_level for c in ALL_CASES)
    diff_counts = Counter(c.difficulty for c in ALL_CASES)
    spec_counts = Counter(c.true_specialty for c in ALL_CASES)
    lines = [
        f"Total cases: {len(ALL_CASES)}",
        f"By difficulty: {dict(diff_counts)}",
        f"By ESI level: {dict(sorted(esi_counts.items()))}",
        f"Distinct specialties: {len(spec_counts)}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(_case_stats())
    print()
    errs = validate_cases()
    if errs:
        print(f"VALIDATION ERRORS ({len(errs)} cases):")
        for cid, msgs in errs.items():
            for m in msgs:
                print(f"  [{cid}] {m}")
    else:
        print("All cases valid ✓")
