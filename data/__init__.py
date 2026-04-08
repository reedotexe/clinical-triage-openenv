"""
Clinical Triage Data Package.

Exports:
  - PatientCase: dataclass for a single synthetic patient
  - get_cases(difficulty): returns list of cases for a given difficulty
  - ALL_CASES, EASY_CASES, MEDIUM_CASES, HARD_CASES
  - SPECIALTIES, ADJACENT_SPECIALTIES, AVAILABLE_TESTS, ESI_DANGER_ZONE
  - is_danger_zone, specialty_distance
"""

from .cases import PatientCase, get_cases, get_case_by_id, get_random_case, validate_cases, ALL_CASES, EASY_CASES, MEDIUM_CASES, HARD_CASES
from .specialties import (
    SPECIALTIES,
    ADJACENT_SPECIALTIES,
    AVAILABLE_TESTS,
    ESI_DANGER_ZONE,
    is_danger_zone,
    specialty_distance,
)

__all__ = [
    "PatientCase",
    "get_cases",
    "get_case_by_id",
    "get_random_case",
    "validate_cases",
    "ALL_CASES",
    "EASY_CASES",
    "MEDIUM_CASES",
    "HARD_CASES",
    "SPECIALTIES",
    "ADJACENT_SPECIALTIES",
    "AVAILABLE_TESTS",
    "ESI_DANGER_ZONE",
    "is_danger_zone",
    "specialty_distance",
]
