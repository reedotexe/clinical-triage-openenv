"""
Inference script for the Clinical Triage OpenEnv environment.

Runs an LLM agent (via HF router) against all 3 tasks and emits the
required hackathon log format to stdout.

Required environment variable:
    HF_TOKEN       — Hugging Face user token (used as the API key)

Optional environment variables:
    ENV_BASE_URL   — Running server URL (default: http://localhost:8000)
    API_BASE_URL   — LLM API base URL   (default: https://router.huggingface.co/v1)
    MODEL_NAME     — LLM model name     (default: Qwen/Qwen2.5-72B-Instruct)

Log format (hackathon spec):
    [START] task=<name> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from clinical_triage.client import ClinicalTriageEnv
    from clinical_triage.models import TriageAction, TriageObservation
except ImportError:
    from client import ClinicalTriageEnv  # type: ignore[no-redef]
    from models import TriageAction, TriageObservation  # type: ignore[no-redef]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME: str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL") or "http://localhost:8000"
BENCHMARK_NAME: str = "clinical_triage"
SUCCESS_THRESHOLD: float = 0.5
MAX_STEPS: int = 12  # buffer above task max (task 3 = 10 steps)
TASKS: List[str] = ["triage_level", "triage_referral", "full_workup"]

_VALID_SPECIALTIES = [
    "cardiology", "neurology", "pulmonology", "gastroenterology",
    "orthopedics", "general_surgery", "internal_medicine", "psychiatry",
    "pediatrics", "emergency_medicine", "obstetrics_gynecology", "urology",
    "dermatology", "nephrology", "hematology", "endocrinology",
    "infectious_disease", "trauma_surgery", "vascular_surgery", "neurosurgery",
]


# ─────────────────────────────────────────────────────────────────────────────
# Logging  (exact hackathon format)
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: TriageAction,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    action_dict = {
        k: v for k, v in action.model_dump().items()
        if v is not None and k != "metadata"
    }
    action_str = json.dumps(action_dict, separators=(",", ":"))
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM client
# ─────────────────────────────────────────────────────────────────────────────

def build_llm_client() -> Optional[OpenAI]:
    if not HF_TOKEN:
        print(
            "[WARN] HF_TOKEN not set — running deterministic fallback policy",
            flush=True,
        )
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

_TASK1_SYSTEM = """\
You are an experienced emergency department triage nurse.
Given the patient presentation, assign the correct ESI (Emergency Severity Index) level.

ESI scale:
  1 = Immediate life-saving intervention required (cardiac arrest, respiratory failure, unresponsive)
  2 = High-risk situation, severe pain, or danger-zone vitals:
        HR >100 or <50  |  SpO2 <92%  |  SBP <90  |  RR >24 or <10  |  Pain >=8
        Stroke symptoms, major trauma, altered mental status, active suicidal plan
  3 = Stable but requires 2+ diagnostic resources (labs, imaging, IV)
  4 = Stable, requires only 1 resource
  5 = No resources needed (simple wound check, prescription refill)

Before assigning ESI, mentally run through:
  Step 1: Is there an immediate life threat? (ESI 1)
  Step 2: Is there a high-risk situation OR danger-zone vitals OR severe pain? (ESI 2)
  Step 3: How many resources will this patient need? (determines ESI 3/4/5)

Respond with ONLY a single valid JSON object — no markdown, no explanation:
{"kind": "ASSIGN_TRIAGE", "triage_level": <1-5>}"""

_TASK2_SYSTEM = """\
You are an emergency department triage nurse and specialist coordinator.
Assign the ESI triage level AND the most appropriate referral specialty.

ESI scale:
  1 = Immediate life-saving intervention required
  2 = High-risk, severe pain, or danger-zone vitals
  3 = Stable, needs 2+ resources
  4 = Stable, needs 1 resource
  5 = No resources needed

Specialty routing guide (pick the PRIMARY specialty):
  Chest pain / palpitations / MI / heart failure     -> cardiology
  Stroke / seizure / neurological deficit / headache -> neurology
  Respiratory distress / COPD / PE / pneumonia       -> pulmonology
  Abdominal pain / GI bleed / liver / bowel          -> gastroenterology
  Fracture / joint / sprain / back pain              -> orthopedics
  Penetrating / blunt polytrauma / burn              -> trauma_surgery
  Vascular (aortic / mesenteric / limb ischaemia)    -> vascular_surgery
  Intracranial bleed / neurosurgical                 -> neurosurgery
  Psychiatric / suicidal / psychosis                 -> psychiatry
  Paediatric presentations (age <16)                 -> pediatrics
  Pregnancy / gynaecological                         -> obstetrics_gynecology
  Urinary / renal / kidney stone                     -> urology
  Diabetes / thyroid / adrenal / metabolic           -> endocrinology
  Sepsis / infection (when source unclear)           -> infectious_disease
  Other / unclear                                    -> emergency_medicine

Valid specialties (use EXACTLY one of these strings):
  cardiology, neurology, pulmonology, gastroenterology, orthopedics,
  general_surgery, internal_medicine, psychiatry, pediatrics, emergency_medicine,
  obstetrics_gynecology, urology, dermatology, nephrology, hematology, endocrinology,
  infectious_disease, trauma_surgery, vascular_surgery, neurosurgery

Respond with ONLY a single valid JSON object — no markdown, no explanation:
{"kind": "ASSIGN_TRIAGE_REFERRAL", "triage_level": <1-5>, "specialty": "<specialty>"}"""

_TASK3_SYSTEM = """\
You are an emergency physician conducting a full clinical workup.
You start with minimal information. Gather critical data efficiently, then commit a final decision.

Available actions (respond with exactly ONE JSON object per turn, no markdown):

Request a vital sign:
  {"kind": "REQUEST_VITAL", "request_target": "<vital>"}
  Vitals: hr, sbp, dbp, rr, spo2

Request a physical exam finding:
  {"kind": "REQUEST_EXAM", "request_target": "<body_part>"}

Order a diagnostic test (COSTS efficiency points — use targeted tests only):
  {"kind": "ORDER_TEST", "request_target": "<test>"}
  Tests: ecg, troponin, chest_xray, ct_head, ct_chest, ct_abdomen, cbc, bmp,
         lft, lipase, urinalysis, d_dimer, lactate, blood_cultures, echo

Commit your final decision (do this as soon as you have enough information):
  {"kind": "FINALIZE", "triage_level": <1-5>, "specialty": "<specialty>", "diagnosis": "<dx>"}

THREE-PHASE STRATEGY:
  PHASE 1 — Haemodynamic check (steps 1-2): Always request hr and sbp first.
  PHASE 2 — Targeted investigation (steps 3-5): Order 1-2 tests guided by the chief complaint:
      Chest pain / palpitations     -> ecg, troponin
      Dyspnoea / respiratory        -> chest_xray, d_dimer
      Headache / neuro deficit      -> ct_head
      Abdominal pain                -> cbc, bmp, lipase or lft
      Fever / infection             -> cbc, lactate, blood_cultures
      Collapse / altered mental     -> glucose (via bmp), ecg
  PHASE 3 — Finalize (step 6 or earlier): FINALIZE with your best triage level, specialty, diagnosis.

RULES:
  - Never request more than 3 tests total.
  - Never exceed 8 steps — FINALIZE by step 8 at the latest.
  - Be decisive: 4-6 steps is ideal for most cases.

Respond with ONLY a single valid JSON object — no markdown, no explanation."""

_SYSTEM_PROMPTS: Dict[str, str] = {
    "triage_level": _TASK1_SYSTEM,
    "triage_referral": _TASK2_SYSTEM,
    "full_workup": _TASK3_SYSTEM,
}


# ─────────────────────────────────────────────────────────────────────────────
# Observation → text for LLM
# ─────────────────────────────────────────────────────────────────────────────

def _obs_to_text(obs: TriageObservation) -> str:
    lines: List[str] = [
        f"Patient: {obs.age}y {obs.sex}, arrival: {obs.arrival_mode}",
        f"Chief complaint: {obs.chief_complaint}",
        f"History: {obs.history_present_illness}",
    ]
    if obs.past_medical_history:
        lines.append(f"Past medical history: {', '.join(obs.past_medical_history)}")
    if obs.medications:
        lines.append(f"Medications: {', '.join(obs.medications)}")
    if obs.allergies:
        lines.append(f"Allergies: {', '.join(obs.allergies)}")

    vitals_known = {k: v for k, v in obs.vitals.items() if v is not None}
    vitals_hidden = [k for k, v in obs.vitals.items() if v is None]
    if vitals_known:
        lines.append(f"Vitals (known): {', '.join(f'{k}={v}' for k,v in vitals_known.items())}")
    if vitals_hidden:
        lines.append(f"Vitals (not yet requested): {', '.join(vitals_hidden)}")

    if obs.revealed_exam_findings:
        findings = "; ".join(f"{k}: {v}" for k, v in obs.revealed_exam_findings.items())
        lines.append(f"Exam findings: {findings}")
    if obs.revealed_test_results:
        results = "; ".join(f"{k}: {v}" for k, v in obs.revealed_test_results.items())
        lines.append(f"Test results: {results}")

    if obs.max_steps and obs.max_steps > 1:
        remaining = obs.max_steps - obs.steps_taken
        lines.append(f"Progress: step {obs.steps_taken}/{obs.max_steps} ({remaining} steps remaining)")
    if obs.message:
        lines.append(f"[env] {obs.message}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Action parsing
# ─────────────────────────────────────────────────────────────────────────────

# Common LLM specialty outputs → canonical specialty names
_SPECIALTY_ALIASES: Dict[str, str] = {
    "surgery": "general_surgery",
    "general surgery": "general_surgery",
    "trauma": "trauma_surgery",
    "trauma surgery": "trauma_surgery",
    "vascular": "vascular_surgery",
    "vascular surgery": "vascular_surgery",
    "neuro": "neurology",
    "neurosurg": "neurosurgery",
    "ob/gyn": "obstetrics_gynecology",
    "obgyn": "obstetrics_gynecology",
    "ob_gyn": "obstetrics_gynecology",
    "gynecology": "obstetrics_gynecology",
    "gi": "gastroenterology",
    "gastro": "gastroenterology",
    "id": "infectious_disease",
    "ortho": "orthopedics",
    "orthopedic": "orthopedics",
    "psych": "psychiatry",
    "pulm": "pulmonology",
    "respiratory": "pulmonology",
    "cardio": "cardiology",
    "cardiac": "cardiology",
    "em": "emergency_medicine",
    "er": "emergency_medicine",
    "ed": "emergency_medicine",
    "emergency": "emergency_medicine",
    "renal": "nephrology",
    "endo": "endocrinology",
    "derm": "dermatology",
    "peds": "pediatrics",
    "hem": "hematology",
    "heme": "hematology",
    "uro": "urology",
}


def _normalize_specialty(s: Optional[str]) -> Optional[str]:
    """Map common LLM output variants to canonical specialty names."""
    if not s:
        return s
    s_lower = s.lower().strip().replace("-", "_").replace(" ", "_")
    if s_lower in {sp.lower() for sp in [
        "cardiology", "neurology", "pulmonology", "gastroenterology",
        "orthopedics", "general_surgery", "internal_medicine", "psychiatry",
        "pediatrics", "emergency_medicine", "obstetrics_gynecology", "urology",
        "dermatology", "nephrology", "hematology", "endocrinology",
        "infectious_disease", "trauma_surgery", "vascular_surgery", "neurosurgery",
    ]}:
        return s_lower
    return _SPECIALTY_ALIASES.get(s.lower().strip(), s_lower)


def _parse_action(raw: str) -> Optional[TriageAction]:
    """Extract a TriageAction from LLM output. Returns None on failure."""
    if not raw:
        return None
    candidates = [raw]
    lb, rb = raw.find("{"), raw.rfind("}")
    if lb != -1 and rb > lb:
        candidates.append(raw[lb: rb + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if "specialty" in data and data["specialty"]:
                data["specialty"] = _normalize_specialty(data["specialty"])
            return TriageAction.model_validate(data)
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic fallback policy  (used when LLM unavailable or parse fails)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_esi(obs: TriageObservation) -> int:
    v = obs.vitals
    hr = v.get("hr")
    sbp = v.get("sbp")
    spo2 = v.get("spo2")
    rr = v.get("rr")
    pain = v.get("pain")

    if hr and (hr > 120 or hr < 45):
        return 2
    if sbp and sbp < 90:
        return 2
    if spo2 and spo2 < 92:
        return 2
    if rr and (rr > 24 or rr < 10):
        return 2
    if pain and pain >= 8:
        return 2

    cc = (obs.chief_complaint + " " + obs.history_present_illness).lower()
    critical_kw = [
        "cardiac arrest", "not breathing", "unresponsive", "anaphylaxis",
        "stroke", "facial droop", "slurred speech", "crushing chest",
        "suicidal", "overdose", "major trauma", "penetrating",
    ]
    if any(kw in cc for kw in critical_kw):
        return 2

    moderate_kw = [
        "chest pain", "shortness of breath", "abdominal pain", "severe headache",
        "fever", "weakness", "nausea", "vomiting", "dizzy", "syncope",
    ]
    if any(kw in cc for kw in moderate_kw):
        return 3

    return 3


def _estimate_specialty(obs: TriageObservation) -> str:
    text = (
        obs.chief_complaint + " " + obs.history_present_illness
        + " " + " ".join(obs.past_medical_history)
    ).lower()

    rules: List[Tuple[List[str], str]] = [
        (["chest pain", "heart", "cardiac", "palpitation", "stemi", "nstemi", "mi "], "cardiology"),
        (["stroke", "facial droop", "slurred speech", "numbness", "seizure", "worst headache"], "neurology"),
        (["shortness of breath", "dyspnea", "asthma", "copd", "pneumonia", "respiratory"], "pulmonology"),
        (["abdominal pain", "nausea vomiting", "diarrhea", "bowel", "liver", "pancreatitis"], "gastroenterology"),
        (["fracture", "joint pain", "sprain", "orthopedic", "back pain", "fall"], "orthopedics"),
        (["trauma", "laceration", "stabbing", "gunshot", "major injury"], "trauma_surgery"),
        (["suicidal", "psychiatric", "hallucin", "psychosis", "depression"], "psychiatry"),
        (["urinary", "kidney", "renal", "dysuria", "hematuria", "flank"], "urology"),
        (["sepsis", "infection", "fever", "bacteremia", "abscess"], "infectious_disease"),
        (["pregnancy", "obstetric", "gynecol", "vaginal"], "obstetrics_gynecology"),
    ]
    for keywords, specialty in rules:
        if any(kw in text for kw in keywords):
            return specialty
    return "emergency_medicine"


def _targeted_test(obs: TriageObservation) -> Optional[str]:
    """Return the single most useful test based on chief complaint keywords."""
    text = (obs.chief_complaint + " " + obs.history_present_illness).lower()
    if any(k in text for k in ["chest pain", "palpitat", "heart", "cardiac"]):
        used = set(obs.revealed_test_results or {})
        return "ecg" if "ecg" not in used else ("troponin" if "troponin" not in used else None)
    if any(k in text for k in ["breath", "dyspnoe", "copd", "asthma", "respiratory"]):
        used = set(obs.revealed_test_results or {})
        return "chest_xray" if "chest_xray" not in used else None
    if any(k in text for k in ["headache", "neuro", "stroke", "seizure", "confusion", "unconscious"]):
        used = set(obs.revealed_test_results or {})
        return "ct_head" if "ct_head" not in used else None
    if any(k in text for k in ["abdominal", "nausea", "vomit", "diarrh", "bowel", "liver", "pancrea"]):
        used = set(obs.revealed_test_results or {})
        return "bmp" if "bmp" not in used else ("cbc" if "cbc" not in used else None)
    if any(k in text for k in ["fever", "sepsis", "infect", "bactere"]):
        used = set(obs.revealed_test_results or {})
        return "lactate" if "lactate" not in used else ("cbc" if "cbc" not in used else None)
    return None


def _fallback_task3(obs: TriageObservation) -> TriageAction:
    """3-phase strategy: hr -> sbp -> 1 targeted test -> FINALIZE."""
    # Phase 1: haemodynamic vitals
    for vital in ["hr", "sbp"]:
        if obs.vitals.get(vital) is None:
            return TriageAction(kind="REQUEST_VITAL", request_target=vital)
    # Phase 2: one targeted test (if under step 5)
    steps = obs.steps_taken or 0
    if steps < 5 and not obs.revealed_test_results:
        test = _targeted_test(obs)
        if test:
            return TriageAction(kind="ORDER_TEST", request_target=test)
    # Phase 3: finalize
    cc = (obs.chief_complaint + " " + obs.history_present_illness).lower()
    diagnosis = "unspecified"
    if "chest pain" in cc or "cardiac" in cc:
        diagnosis = "acute_coronary_syndrome"
    elif "breath" in cc or "respiratory" in cc:
        diagnosis = "respiratory_failure"
    elif "headache" in cc or "seizure" in cc:
        diagnosis = "neurological_emergency"
    elif "abdominal" in cc:
        diagnosis = "acute_abdomen"
    elif "fever" in cc or "sepsis" in cc:
        diagnosis = "sepsis"
    return TriageAction(
        kind="FINALIZE",
        triage_level=_estimate_esi(obs),
        specialty=_estimate_specialty(obs),
        diagnosis=diagnosis,
    )


def _fallback_action(obs: TriageObservation, task: str) -> TriageAction:
    if task == "triage_level":
        return TriageAction(kind="ASSIGN_TRIAGE", triage_level=_estimate_esi(obs))
    elif task == "triage_referral":
        return TriageAction(
            kind="ASSIGN_TRIAGE_REFERRAL",
            triage_level=_estimate_esi(obs),
            specialty=_estimate_specialty(obs),
        )
    else:
        return _fallback_task3(obs)


# ─────────────────────────────────────────────────────────────────────────────
# Action selection  (LLM with fallback)
# ─────────────────────────────────────────────────────────────────────────────

def choose_action(
    obs: TriageObservation,
    task: str,
    llm: Optional[OpenAI],
    conversation: List[Dict[str, Any]],
) -> Tuple[TriageAction, Optional[str]]:
    """Return (action, error_string_or_None). Falls back to policy on any failure."""
    if llm is None:
        return _fallback_action(obs, task), None

    user_msg = _obs_to_text(obs)
    conversation.append({"role": "user", "content": user_msg})

    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=256,
            messages=conversation,
        )
        raw = (response.choices[0].message.content or "").strip()
        conversation.append({"role": "assistant", "content": raw})

        action = _parse_action(raw)
        if action is not None:
            return action, None

        err = f"parse_error:{raw[:60]}"
    except Exception as exc:
        err = f"llm_error:{str(exc)[:60]}"
        conversation.append({"role": "assistant", "content": ""})

    return _fallback_action(obs, task), err


# ─────────────────────────────────────────────────────────────────────────────
# Task runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(
    env: ClinicalTriageEnv,
    task: str,
    llm: Optional[OpenAI],
) -> Dict[str, Any]:
    """Run one full episode for the given task. Returns summary dict."""
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    exc: Optional[Exception] = None

    log_start(task=task, env=BENCHMARK_NAME, model=MODEL_NAME)

    conversation: List[Dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPTS[task]}
    ]

    try:
        result = env.reset(task=task)
        obs: TriageObservation = (
            result.observation if hasattr(result, "observation") else result
        )

        for step_idx in range(1, MAX_STEPS + 1):
            if getattr(result, "done", False):
                break

            action, err = choose_action(obs, task, llm, conversation)
            result = env.step(action)
            obs = result.observation if hasattr(result, "observation") else result

            reward = float(result.reward if result.reward is not None else 0.0)
            rewards.append(reward)
            steps_taken = step_idx
            log_step(step_idx, action, reward, bool(result.done), err)

            if result.done:
                break

        score = rewards[-1] if rewards else 0.0
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        exc = e
        score = rewards[-1] if rewards else 0.0
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    if exc:
        print(f"[ERROR] task={task} exception={exc}", flush=True)

    return {
        "task": task,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "exception": exc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    llm = build_llm_client()
    first_exception: Optional[Exception] = None

    with ClinicalTriageEnv(base_url=ENV_BASE_URL).sync() as env:
        for task in TASKS:
            result = run_task(env, task, llm)
            if first_exception is None and result["exception"] is not None:
                first_exception = result["exception"]

    if first_exception is not None:
        sys.exit(1)


if __name__ == "__main__":
    main()
