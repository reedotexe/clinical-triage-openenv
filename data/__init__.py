"""
Clinical Triage Data Package.

Exports:
  - PatientCase + case helpers
  - Specialty constants + distance function
  - Grader functions (grade_triage_level, grade_referral, etc.)
  - Composite reward functions (compute_task1/2/3_reward)
"""

from .cases import (
    PatientCase,
    get_cases,
    get_case_by_id,
    get_random_case,
    validate_cases,
    ALL_CASES,
    EASY_CASES,
    MEDIUM_CASES,
    HARD_CASES,
)
from .specialties import (
    SPECIALTIES,
    ADJACENT_SPECIALTIES,
    AVAILABLE_TESTS,
    ESI_DANGER_ZONE,
    is_danger_zone,
    specialty_distance,
)
from .graders import (
    grade_triage_level,
    grade_referral,
    grade_diagnosis,
    grade_efficiency,
    red_flag_modifier,
    compute_task1_reward,
    compute_task2_reward,
    compute_task3_reward,
)

__all__ = [
    # Cases
    "PatientCase", "get_cases", "get_case_by_id", "get_random_case",
    "validate_cases", "ALL_CASES", "EASY_CASES", "MEDIUM_CASES", "HARD_CASES",
    # Specialties
    "SPECIALTIES", "ADJACENT_SPECIALTIES", "AVAILABLE_TESTS",
    "ESI_DANGER_ZONE", "is_danger_zone", "specialty_distance",
    # Graders
    "grade_triage_level", "grade_referral", "grade_diagnosis",
    "grade_efficiency", "red_flag_modifier",
    "compute_task1_reward", "compute_task2_reward", "compute_task3_reward",
]
