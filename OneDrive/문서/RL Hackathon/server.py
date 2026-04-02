"""
MediGuard-AI — FastAPI Server for HuggingFace Spaces / OpenEnv validation.

Exposes the MediGuardEnv as REST endpoints:
  POST /reset   — reset environment, returns first observation
  POST /step    — take an action, returns (obs, reward, done, info)
  GET  /state   — get current environment state
  GET  /health  — health check
  GET  /        — landing page with environment info
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from mediguard_env import MediGuardEnv

# ------------------------------------------------------------------ #
#  FastAPI app                                                        #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="MediGuard-AI Environment",
    description="OpenEnv-compliant ICU patient monitoring RL environment",
    version="1.0.0",
)

# ------------------------------------------------------------------ #
#  Request / Response models                                          #
# ------------------------------------------------------------------ #

class ResetRequest(BaseModel):
    task: str = Field(default="suppression", description="Task name: suppression, deterioration, triage")
    seed: int = Field(default=42, description="Random seed for reproducibility")


class StepRequest(BaseModel):
    action: Union[int, List[int]] = Field(..., description="Action: int for single-patient, List[int] for triage")


class StepResponse(BaseModel):
    observation: Union[Dict, List[Dict]]
    reward: float
    done: bool
    info: Dict


class StateResponse(BaseModel):
    step: int
    task: str
    patient_type: Union[str, List[str]]
    done: bool
    current_activity: Union[int, List[int]]
    deterioration_severity: Union[float, List[float]]


# ------------------------------------------------------------------ #
#  Environment instance (one per server for simplicity)               #
# ------------------------------------------------------------------ #

_env: Optional[MediGuardEnv] = None


# ------------------------------------------------------------------ #
#  Endpoints                                                          #
# ------------------------------------------------------------------ #

@app.get("/")
async def root():
    """Landing page with environment metadata."""
    return {
        "name": "MediGuard-AI",
        "description": "AI-powered ICU patient monitoring RL environment",
        "version": "1.0.0",
        "tasks": ["suppression", "deterioration", "triage"],
        "action_space": {"type": "discrete", "n": 3, "labels": {0: "Ignore", 1: "Verify", 2: "Alert"}},
        "episode_length": 360,
        "endpoints": {
            "POST /reset": "Reset environment with task and seed",
            "POST /step":  "Take an action, get (obs, reward, done, info)",
            "GET /state":  "Get current environment state",
            "GET /health": "Health check",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment and return the first observation."""
    global _env

    if request.task not in ("suppression", "deterioration", "triage"):
        raise HTTPException(status_code=400, detail=f"Unknown task: {request.task}")

    _env = MediGuardEnv(task=request.task, seed=request.seed)
    obs = _env.reset()

    return {"observation": obs, "info": {"task": request.task, "seed": request.seed}}


@app.post("/step")
async def step(request: StepRequest):
    """Execute one step in the environment."""
    global _env

    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
        obs, reward, done, info = _env.step(request.action)
    except (AssertionError, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
async def state():
    """Return the current environment state."""
    global _env

    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    return _env.state()
