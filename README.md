---
title: Clinical Triage Environment
emoji: 🏥
colorFrom: red
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - healthcare
  - triage
  - reinforcement-learning
---

# 🏥 Clinical Triage Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) reinforcement-learning environment that simulates emergency-department clinical triage. An AI agent reviews patient presentations and must correctly assign ESI triage levels, determine specialist referrals, and conduct diagnostic workups — exactly as a trained triage nurse or emergency physician would.

Built for the **Meta PyTorch OpenEnv Hackathon x Scaler School of Technology**.

---

## Tasks

The environment supports three progressively harder tasks:

| Task | Description | Max Steps | Key Metrics |
|------|-------------|-----------|-------------|
| **Task 1 — Triage Level** | Assign ESI level (1–5) to a fully-observed patient | 1 | Asymmetric ESI scoring |
| **Task 2 — Triage + Referral** | Assign ESI level AND the correct specialist referral | 1 | ESI + specialty adjacency scoring |
| **Task 3 — Full Workup** | Gather vitals/tests efficiently, then commit a full diagnosis | 10 | ESI + specialty + diagnosis + efficiency |

---

## Reward Design

All task scores are bounded to **[0.0, 1.0]**.

### Component Weights

| Task | ESI weight | Specialty weight | Diagnosis weight | Efficiency weight |
|------|-----------|-----------------|-----------------|------------------|
| Task 1 | 1.00 | — | — | — |
| Task 2 | 0.60 | 0.40 | — | — |
| Task 3 | 0.30 | 0.30 | 0.25 | 0.15 |

### Scoring Details

- **ESI grading** — Exact match = 1.0. Under-triage penalised 2× more than over-triage (a patient who needs ESI-1 but gets ESI-3 could die waiting). Adjacent level = 0.5.
- **Specialty grading** — Exact = 1.0, clinically adjacent specialty (e.g. `internal_medicine` for a `cardiology` case) = 0.5, unrelated = 0.0.
- **Diagnosis grading** — Exact/synonym match = 1.0, keyword overlap or same category = 0.5, miss = 0.0.
- **Efficiency penalty** — Each test beyond the optimal set deducts 0.05; each step beyond optimal deducts 0.02. Bounded so it never makes the total negative.
- **Red-flag modifier** — +0.1 bonus for correctly identifying ESI-1/2 patients; −0.2 penalty for severe under-triage of critical cases.

### ESI Reference Table

| ESI | Category | Typical Presentations |
|-----|----------|-----------------------|
| 1 | Immediate | Cardiac arrest, respiratory failure, unresponsive, massive haemorrhage |
| 2 | Emergent | Stroke, STEMI, severe sepsis, aortic dissection, active suicidal plan |
| 3 | Urgent | Pneumonia, moderate pain, dehydration, stable fracture |
| 4 | Less Urgent | Minor laceration, mild infection, ankle sprain |
| 5 | Non-Urgent | Prescription refill, conjunctivitis, minor rash |

### Worked Examples

**Task 2, seed=5** — Jaundice, dark urine, painless (ESI 3, gastroenterology):
```
Agent: ASSIGN_TRIAGE_REFERRAL(triage_level=3, specialty="gastroenterology")
ESI score:       1.0  (exact match)
Specialty score: 1.0  (exact match)
Red-flag bonus:  0.0
Final reward:    0.6*1.0 + 0.4*1.0 = 1.0
```

**Task 3, seed=7** — Dysphagia (ESI 3, general_surgery) — agent used 4 steps:
```
Agent: hr -> sbp -> ORDER_TEST(ecg) -> FINALIZE(level=3, specialty="gastroenterology", diagnosis="foreign_body")
ESI score:       1.0  (exact match)
Specialty score: 0.5  (adjacent: general_surgery ↔ gastroenterology)
Diagnosis score: 0.5  (keyword match)
Efficiency:      0.9  (1 extra step penalty)
Final reward:    0.3*1.0 + 0.3*0.5 + 0.25*0.5 + 0.15*0.9 = 0.3 + 0.15 + 0.125 + 0.135 = 0.71
```

### Agent Strategy Tips for RL Training

- **Task 1 & 2**: Single-step decisions. Use ESI criteria strictly — vitals in danger zones always indicate ESI ≤ 2. The asymmetric penalty means a conservative bias (triage higher) is safer than under-triaging.
- **Task 3**: Reward GRPO/PPO by shaping partial rewards per step (specialty revealed by ECG results, etc). The efficiency penalty creates a natural exploration-exploitation tradeoff: gather just enough information to be confident.
- **Curriculum learning**: Train on `triage_level` first, then `triage_referral`, then `full_workup`. The ESI component is shared across all tasks.
- **Negative reward avoidance**: The −0.2 red-flag penalty is the harshest signal — teach your agent to never under-triage ESI-1 patients.

---

## Quick Start

```python
from clinical_triage import ClinicalTriageEnv, TriageAction

# Task 1 — single-step triage
with ClinicalTriageEnv(base_url="https://reedhamo-clinical-triage.hf.space").sync() as env:
    result = env.reset(task="triage_level")
    obs = result.observation
    print(obs.chief_complaint)
    result = env.step(TriageAction(kind="ASSIGN_TRIAGE", triage_level=2))
    print(f"Reward: {result.reward}  Done: {result.done}")

# Task 3 — multi-step workup
with ClinicalTriageEnv(base_url="https://reedhamo-clinical-triage.hf.space").sync() as env:
    result = env.reset(task="full_workup")
    result = env.step(TriageAction(kind="REQUEST_VITAL", request_target="hr"))
    result = env.step(TriageAction(kind="ORDER_TEST",    request_target="ecg"))
    result = env.step(TriageAction(kind="FINALIZE",
                                   triage_level=2,
                                   specialty="cardiology",
                                   diagnosis="stemi"))
    print(f"Final score: {result.reward}")
```

---

## Action Space

```python
# Task 1
{"kind": "ASSIGN_TRIAGE", "triage_level": <1-5>}

# Task 2
{"kind": "ASSIGN_TRIAGE_REFERRAL", "triage_level": <1-5>, "specialty": "<specialty>"}

# Task 3 — gather information
{"kind": "REQUEST_VITAL",  "request_target": "hr"}      # hr, sbp, dbp, rr, spo2
{"kind": "REQUEST_EXAM",   "request_target": "<body_part>"}
{"kind": "ORDER_TEST",     "request_target": "ecg"}     # ecg, troponin, cbc, ...

# Task 3 — commit decision
{"kind": "FINALIZE", "triage_level": <1-5>, "specialty": "<specialty>", "diagnosis": "<dx>"}
```

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `chief_complaint` | str | Patient's primary complaint |
| `history_present_illness` | str | Narrative history |
| `past_medical_history` | list[str] | Known conditions |
| `medications` | list[str] | Current medications |
| `vitals` | dict | hr, sbp, dbp, rr, spo2, temp, pain (may be hidden in Task 3) |
| `available_actions` | list[str] | Valid action kinds at current step |
| `steps_taken` / `max_steps` | int | Progress counter |
| `revealed_exam_findings` | dict | Task-3 progressive reveal |
| `revealed_test_results` | dict | Task-3 progressive reveal |
| `message` | str | Feedback from environment |

---

## Case Bank

100 synthetic patient cases validated by clinical criteria:
- **30 easy** — clear presentations, full vitals
- **30 medium** — ambiguous presentations, partial vitals
- **40 hard** — sparse initial info, requires active information gathering

Covers cardiology, neurology, pulmonology, GI, orthopedics, psychiatry, trauma, and more.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/info` | GET | Case bank stats, task descriptions, reward weights |
| `/reset` | POST | Start new episode `{"task": "triage_level", "seed": 42}` |
| `/step` | POST | Submit action `{"action": {"kind": "ASSIGN_TRIAGE", "triage_level": 2}}` |
| `/state` | GET | Current internal state |
| `/schema` | GET | Action JSON schema |
| `/ws` | WS | Persistent WebSocket for low-latency multi-step episodes |
| `/docs` | GET | Swagger UI |

---

## Baseline Scores

Using `Qwen/Qwen2.5-72B-Instruct` via HF Inference Router (seed=42):

| Task | Score | Steps | Notes |
|------|-------|-------|-------|
| **triage_level** | **1.000** | 1 | Perfect ESI assignment |
| **triage_referral** | **1.000** | 1 | Perfect ESI + specialty |
| **full_workup** | **0.925** | 6 | HR → SBP → test → FINALIZE |
| **Average** | **0.975** | — | |

---

## Running Inference

```bash
HF_TOKEN=<your_token> \
ENV_BASE_URL=https://reedhamo-clinical-triage.hf.space \
python inference.py
```

Expected output format:
```
[START] task=triage_level env=clinical_triage model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action={"kind":"ASSIGN_TRIAGE","triage_level":2} reward=1.00 done=true error=null
[END]   success=true steps=1 score=1.000 rewards=1.00
```

---

## Project Structure

```
clinical_triage/
├── Dockerfile                          # Container (project root, required by hackathon)
├── requirements.txt                    # pip dependencies
├── openenv.yaml                        # OpenEnv manifest
├── inference.py                        # LLM agent inference script
├── __init__.py                         # Public API exports
├── models.py                           # TriageAction, TriageObservation, TriageState
├── client.py                           # ClinicalTriageEnv WebSocket client
├── data/
│   ├── cases.py                        # PatientCase dataclass + case bank
│   ├── cases_easy/medium/hard.py       # 60 synthetic cases
│   ├── graders.py                      # All reward / grading functions
│   └── specialties.py                  # Specialty lists, adjacency map
├── server/
│   ├── app.py                          # FastAPI app (create_app + /health + /info)
│   └── clinical_triage_environment.py  # Core Environment logic
└── tests/
    ├── test_graders.py                 # 45 grader unit tests
    └── test_client.py                  # 12 client serialization tests
```

---

## Team

- **Team**: Chocolate
- **Lead**: Reedham (reedham.sh@gmail.com)
- **Member**: Meharpreet Kaur
- **Hackathon**: Meta PyTorch OpenEnv Hackathon × Scaler School of Technology
