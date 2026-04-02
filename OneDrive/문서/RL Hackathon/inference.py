"""
MediGuard-AI — Baseline Inference Script

Runs a rule-based agent against all 3 hackathon tasks and prints
structured logs that the automated scoring pipeline reads.

Environment variables:
  API_BASE_URL — LLM endpoint (default: HuggingFace router)
  MODEL_NAME   — model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN / API_KEY — authentication token

Usage:
  python inference.py
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, List, Union

from openai import OpenAI
from mediguard_env import MediGuardEnv

# ------------------------------------------------------------------ #
#  Configuration from environment variables                          #
# ------------------------------------------------------------------ #

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# OpenAI client — mandatory per hackathon rules.
# The baseline uses rule-based logic, but the client is ready for
# LLM-based agents. Replace baseline_agent() to use this client.
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

# Action constants
IGNORE = 0
VERIFY = 1
ALERT  = 2


# ------------------------------------------------------------------ #
#  Baseline rule-based agent                                         #
# ------------------------------------------------------------------ #

def baseline_agent(obs: Dict) -> int:
    """
    Rule-based policy that mimics a trained PPO agent.

    Given a single-patient observation dict, returns an action int:
      0 = Ignore, 1 = Verify, 2 = Alert
    """
    activity = obs["activity"]
    delta    = obs["baseline_delta"]
    hours    = obs["hours_observed"]
    spo2     = obs["spo2"]           # normalized 0-1
    hr       = obs["heart_rate"]     # normalized 0-1

    # Still learning baseline — be cautious
    if hours < 1.0:
        return VERIFY

    # Ambulating — elevated HR is expected, don't over-react
    if activity == 2:
        return IGNORE

    # High deviation while resting → alert
    if delta > 0.6 and activity == 0:
        return ALERT

    # Oxygen dangerously low (normalized <0.3 ≈ raw SpO2 < ~91%)
    if spo2 < 0.3:
        return ALERT

    # Moderate deviation while resting → verify
    if delta > 0.35 and activity == 0:
        return VERIFY

    return IGNORE


def triage_agent(obs_list: List[Dict]) -> List[int]:
    """Apply baseline_agent independently to each patient in triage mode."""
    return [baseline_agent(obs) for obs in obs_list]


# ------------------------------------------------------------------ #
#  Logging helpers                                                   #
# ------------------------------------------------------------------ #

def log_start(task: str, model: str):
    print(f"[START] task={task} env=mediguard model={model}", flush=True)


def log_step(step: int, action, reward: float, done: bool, error=None):
    """Print a [STEP] line in the mandatory format."""
    # Format action — triage uses comma-separated ints
    if isinstance(action, (list, tuple)):
        action_str = ",".join(str(a) for a in action)
    else:
        action_str = str(action)

    done_str = "true" if done else "false"
    error_str = "null" if error is None else str(error)

    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)


# ------------------------------------------------------------------ #
#  Main episode runner                                               #
# ------------------------------------------------------------------ #

def run_episode(task: str, seed: int = 42):
    """Run a single episode for the given task and print structured logs."""
    log_start(task, MODEL_NAME)

    rewards: List[float] = []
    steps = 0
    success = False

    try:
        env = MediGuardEnv(task=task, seed=seed)
        obs = env.reset()
        done = False

        while not done:
            # Choose action
            if task == "triage":
                action = triage_agent(obs)
            else:
                action = baseline_agent(obs)

            # Step
            obs, reward, done, info = env.step(action)
            steps = info["step"]
            rewards.append(reward)

            log_step(steps, action, reward, done, error=None)

        # Determine success: mean reward > 0.4
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        success = mean_reward > 0.4

    except Exception as exc:
        # Log the failing step and then end
        steps += 1
        log_step(steps, 0, 0.0, True, error=str(exc))
        success = False

    log_end(success, steps, rewards)
    return rewards


# ------------------------------------------------------------------ #
#  Entry point                                                       #
# ------------------------------------------------------------------ #

def main():
    tasks = ["suppression", "deterioration", "triage"]
    all_rewards = {}

    for task in tasks:
        all_rewards[task] = run_episode(task, seed=42)

    # Summary (not part of scored output, but useful for humans)
    print()
    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    for task in tasks:
        rews = all_rewards[task]
        mean_r = sum(rews) / len(rews) if rews else 0.0
        print(f"  {task:15s}  steps={len(rews):4d}  mean_reward={mean_r:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
