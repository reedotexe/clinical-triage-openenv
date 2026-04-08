"""
Specialty mappings, test catalog, and ESI vital-sign danger zone thresholds.
"""

from __future__ import annotations
from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# Valid referral specialties
# ---------------------------------------------------------------------------
SPECIALTIES: List[str] = [
    "cardiology",
    "neurology",
    "pulmonology",
    "gastroenterology",
    "orthopedics",
    "general_surgery",
    "internal_medicine",
    "psychiatry",
    "pediatrics",
    "emergency_medicine",
    "obstetrics_gynecology",
    "urology",
    "dermatology",
    "nephrology",
    "hematology",
    "endocrinology",
    "infectious_disease",
    "trauma_surgery",
    "vascular_surgery",
    "neurosurgery",
]

# ---------------------------------------------------------------------------
# Clinically adjacent specialties (partial credit on referral grading)
# If you predict an adjacent specialty you get 0.5 instead of 0.0
# ---------------------------------------------------------------------------
ADJACENT_SPECIALTIES: Dict[str, Set[str]] = {
    "cardiology":           {"internal_medicine", "emergency_medicine", "vascular_surgery"},
    "neurology":            {"internal_medicine", "emergency_medicine", "neurosurgery"},
    "pulmonology":          {"internal_medicine", "emergency_medicine", "infectious_disease"},
    "gastroenterology":     {"internal_medicine", "general_surgery"},
    "orthopedics":          {"trauma_surgery", "emergency_medicine"},
    "general_surgery":      {"trauma_surgery", "gastroenterology", "vascular_surgery"},
    "internal_medicine":    {"emergency_medicine", "infectious_disease"},
    "psychiatry":           {"emergency_medicine", "internal_medicine"},
    "pediatrics":           {"emergency_medicine", "internal_medicine"},
    "emergency_medicine":   {"internal_medicine", "trauma_surgery"},
    "obstetrics_gynecology":{"internal_medicine", "emergency_medicine", "urology"},
    "urology":              {"obstetrics_gynecology", "nephrology", "general_surgery"},
    "dermatology":          {"internal_medicine", "infectious_disease"},
    "nephrology":           {"internal_medicine", "urology"},
    "hematology":           {"internal_medicine", "oncology"},
    "endocrinology":        {"internal_medicine"},
    "infectious_disease":   {"internal_medicine", "pulmonology"},
    "trauma_surgery":       {"general_surgery", "orthopedics", "emergency_medicine"},
    "vascular_surgery":     {"cardiology", "general_surgery"},
    "neurosurgery":         {"neurology", "trauma_surgery"},
}


def specialty_distance(predicted: str, true: str) -> float:
    """
    Return a similarity score [0.0–1.0] between two specialty strings.
    1.0 = exact match, 0.5 = adjacent, 0.0 = unrelated
    """
    if predicted == true:
        return 1.0
    if predicted in ADJACENT_SPECIALTIES.get(true, set()):
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Available diagnostic tests for Task 3 (multi-step workup)
# cost = number of resource units consumed
# ---------------------------------------------------------------------------
AVAILABLE_TESTS: Dict[str, Dict] = {
    "ecg":            {"cost": 1, "time": "fast",   "category": "cardiac"},
    "troponin":       {"cost": 1, "time": "fast",   "category": "cardiac"},
    "chest_xray":     {"cost": 2, "time": "medium", "category": "respiratory"},
    "ct_head":        {"cost": 3, "time": "medium", "category": "neurological"},
    "ct_chest":       {"cost": 3, "time": "medium", "category": "respiratory"},
    "ct_abdomen":     {"cost": 3, "time": "medium", "category": "abdominal"},
    "cbc":            {"cost": 1, "time": "fast",   "category": "general"},
    "bmp":            {"cost": 1, "time": "fast",   "category": "general"},
    "lft":            {"cost": 1, "time": "fast",   "category": "hepatic"},
    "lipase":         {"cost": 1, "time": "fast",   "category": "abdominal"},
    "urinalysis":     {"cost": 1, "time": "fast",   "category": "renal"},
    "urine_culture":  {"cost": 1, "time": "slow",   "category": "renal"},
    "blood_cultures": {"cost": 2, "time": "slow",   "category": "infectious"},
    "d_dimer":        {"cost": 1, "time": "fast",   "category": "coagulation"},
    "procalcitonin":  {"cost": 1, "time": "fast",   "category": "infectious"},
    "lactate":        {"cost": 1, "time": "fast",   "category": "sepsis"},
    "glucose":        {"cost": 1, "time": "fast",   "category": "metabolic"},
    "hba1c":          {"cost": 1, "time": "fast",   "category": "metabolic"},
    "thyroid_panel":  {"cost": 1, "time": "fast",   "category": "endocrine"},
    "coag_panel":     {"cost": 1, "time": "fast",   "category": "coagulation"},
    "abdominal_us":   {"cost": 2, "time": "medium", "category": "abdominal"},
    "echo":           {"cost": 2, "time": "medium", "category": "cardiac"},
    "mri_brain":      {"cost": 4, "time": "slow",   "category": "neurological"},
    "lumbar_puncture":{"cost": 2, "time": "slow",   "category": "neurological"},
}

# ---------------------------------------------------------------------------
# ESI Vital Sign Danger Zone Thresholds
# Used to determine if a patient should be upgraded from ESI 3 → ESI 2
# ---------------------------------------------------------------------------
ESI_DANGER_ZONE = {
    "hr":   {"low": 50,   "high": 100},   # bpm
    "rr":   {"low": 12,   "high": 20},    # breaths/min
    "sbp":  {"low": 90,   "high": None},  # mmHg — only low is danger
    "spo2": {"low": 92,   "high": None},  # % — only low is danger
    "temp": {"low": None, "high": 38.3},  # °C — only high is danger
}


def is_danger_zone(vitals: dict) -> bool:
    """
    Return True if any vital is in the ESI danger zone.
    Used to decide if a would-be ESI-3 patient should be upgraded to ESI-2.
    """
    for key, thresholds in ESI_DANGER_ZONE.items():
        value = vitals.get(key)
        if value is None:
            continue
        if thresholds["low"] is not None and value < thresholds["low"]:
            return True
        if thresholds["high"] is not None and value > thresholds["high"]:
            return True
    return False
