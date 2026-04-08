"""
FastAPI application for the Clinical Triage Environment.

Exposes the ClinicalTriageEnvironment over HTTP and WebSocket endpoints,
compatible with EnvClient, following the OpenEnv server standard.

Standard endpoints (provided by create_app):
    POST /reset      — Reset the environment (accepts task, seed, case_id)
    POST /step       — Execute an action
    GET  /state      — Get current episode state
    GET  /schema     — Action + observation JSON schemas
    WS   /ws         — WebSocket for persistent sessions

Additional endpoints added here:
    GET  /health     — Lightweight liveness probe for Docker/k8s
    GET  /info       — Environment metadata (tasks, case stats, specialties)

Usage:
    # Development (inside clinical_triage/):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Via uv:
    uv run --project . server

    # Direct:
    python -m server.app
    python -m server.app --port 8001
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install with:\n    uv sync\n"
    ) from e

# ------------------------------------------------------------------
# Import our environment and models (package-relative or bare-module
# fallback for running with  python -m server.app inside clinical_triage/)
# ------------------------------------------------------------------
try:
    from ..models import TriageAction, TriageObservation, TriageState
    from ..data import ALL_CASES, SPECIALTIES
    from .clinical_triage_environment import ClinicalTriageEnvironment
except (ImportError, ModuleNotFoundError):
    from models import TriageAction, TriageObservation, TriageState   # type: ignore
    from data import ALL_CASES, SPECIALTIES                           # type: ignore
    from server.clinical_triage_environment import ClinicalTriageEnvironment  # type: ignore


# ------------------------------------------------------------------
# Core FastAPI app (all standard OpenEnv endpoints)
# ------------------------------------------------------------------
_MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_ENVS", "4"))

app = create_app(
    ClinicalTriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="clinical_triage",
    max_concurrent_envs=_MAX_CONCURRENT,
)


# ------------------------------------------------------------------
# Additional custom endpoints
# ------------------------------------------------------------------
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["info"])


@router.get("/", summary="Landing page", include_in_schema=False)
async def root() -> Dict[str, Any]:
    """Root endpoint — returns environment overview instead of 404."""
    return {
        "name": "clinical_triage",
        "version": "1.0.0",
        "description": "Clinical Triage & Medical Decision-Making Environment for RL training. This space is our submission for Scaler's Meta Pythorch Openenv Hackathon. Reedham and Mehar (Team Chocolate)",
        "framework": "OpenEnv (Meta PyTorch Hackathon 2026)",
        "tasks": ["triage_level", "triage_referral", "full_workup"],
        "endpoints": {
            "health":   "GET  /health",
            "info":     "GET  /info",
            "schema":   "GET  /schema",
            "metadata": "GET  /metadata",
            "reset":    "POST /reset",
            "step":     "POST /step",
            "state":    "GET  /state",
            "docs":     "GET  /docs",
            "ws":       "WS   /ws",
        },
        "space_url": "https://reedhamo-clinical-triage.hf.space",
        "status": "running",
    }


@router.get("/health", summary="Liveness probe")
async def health() -> Dict[str, Any]:
    """
    Lightweight health check for Docker and orchestration platforms.
    Returns HTTP 200 when the server is ready to accept requests.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": "clinical_triage",
        "max_concurrent_envs": _MAX_CONCURRENT,
    }


@router.get("/info", summary="Environment metadata")
async def info() -> Dict[str, Any]:
    """
    Return metadata about the clinical triage environment:
    tasks, case bank stats, and supported specialties.
    Useful for building inference scripts and dashboards.
    """
    from collections import Counter

    esi_dist = dict(sorted(Counter(c.true_esi_level for c in ALL_CASES).items()))
    diff_dist = dict(Counter(c.difficulty for c in ALL_CASES))

    return {
        "env_name": "clinical_triage",
        "version": "1.0.0",
        "description": (
            "A clinical triage and medical decision-making environment "
            "for RL/LLM agent training and evaluation."
        ),
        "tasks": {
            "triage_level": {
                "description": "Assign ESI triage level (1–5). Single-step.",
                "max_steps": 1,
                "difficulty": "easy",
                "valid_actions": ["ASSIGN_TRIAGE"],
            },
            "triage_referral": {
                "description": "Assign ESI level + referral specialty. Single-step.",
                "max_steps": 1,
                "difficulty": "medium",
                "valid_actions": ["ASSIGN_TRIAGE_REFERRAL"],
            },
            "full_workup": {
                "description": (
                    "Multi-step workup: request vitals/exam/tests then FINALIZE."
                ),
                "max_steps": 10,
                "difficulty": "hard",
                "valid_actions": [
                    "REQUEST_VITAL", "REQUEST_EXAM", "ORDER_TEST", "FINALIZE"
                ],
            },
        },
        "case_bank": {
            "total": len(ALL_CASES),
            "by_difficulty": diff_dist,
            "by_esi_level": esi_dist,
        },
        "specialties": sorted(SPECIALTIES),
        "reward_weights": {
            "task1": {"esi": 1.0},
            "task2": {"esi": 0.6, "specialty": 0.4},
            "task3": {"esi": 0.35, "specialty": 0.25, "diagnosis": 0.25, "efficiency": 0.15},
        },
        "reset_params": {
            "task": "triage_level | triage_referral | full_workup",
            "seed": "int (optional) — for reproducibility",
            "case_id": "str (optional) — force a specific case",
        },
    }


# Register additional routes onto the app
app.include_router(router)


# ------------------------------------------------------------------
# Entry points
# ------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Start the server with uvicorn.

    Can be invoked via:
        uv run --project . server
        python -m server.app
        python -m server.app --port 8001
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clinical Triage Environment Server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000)")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
