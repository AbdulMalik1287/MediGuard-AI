"""
MediGuard-AI Environment — OpenEnv-compliant RL environment.

Wraps PatientSimulator to produce normalized observations, compute rewards,
and support 3 hackathon tasks: suppression, deterioration, triage.

OpenEnv spec: reset() → obs, step(action) → (obs, reward, done, info), state() → dict
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field
from patient_simulator import PatientSimulator


# ------------------------------------------------------------------ #
#  Pydantic models (required by OpenEnv spec)                        #
# ------------------------------------------------------------------ #

class ObservationModel(BaseModel):
    """Schema for a single-patient observation dict."""
    heart_rate: float = Field(..., ge=0.0, le=1.0, description="Normalized heart rate")
    systolic_bp: float = Field(..., ge=0.0, le=1.0, description="Normalized systolic BP")
    diastolic_bp: float = Field(..., ge=0.0, le=1.0, description="Normalized diastolic BP")
    spo2: float = Field(..., ge=0.0, le=1.0, description="Normalized SpO2")
    respiratory_rate: float = Field(..., ge=0.0, le=1.0, description="Normalized respiratory rate")
    temperature: float = Field(..., ge=0.0, le=1.0, description="Normalized temperature")
    baseline_delta: float = Field(..., ge=0.0, le=1.0, description="Rolling deviation from personal baseline")
    hours_observed: float = Field(..., ge=0.0, description="Hours elapsed (step / 60)")
    activity: int = Field(..., ge=0, le=4, description="Current activity code")
    vitals_history: list = Field(..., description="Last 10 timesteps of normalized vitals [10][6]")


class ActionModel(BaseModel):
    """Schema for an action.
    
    For suppression / deterioration: a single int in {0,1,2}.
    For triage: a list of 4 ints, each in {0,1,2}.
    """
    action: Union[int, List[int]] = Field(
        ...,
        description="0=Ignore, 1=Verify, 2=Alert. List[int] of length 4 for triage."
    )


# ------------------------------------------------------------------ #
#  Constants                                                         #
# ------------------------------------------------------------------ #

# Normalization ranges: (min, max)
NORM_RANGES = {
    "heart_rate":       (30.0, 200.0),
    "systolic_bp":      (60.0, 220.0),
    "diastolic_bp":     (30.0, 140.0),
    "spo2":             (70.0, 100.0),
    "respiratory_rate": (5.0,  40.0),
    "temperature":      (34.0, 42.0),
}

# Ordered list of vital keys — used for consistent array indexing
VITAL_KEYS = ["heart_rate", "systolic_bp", "diastolic_bp", "spo2", "respiratory_rate", "temperature"]

# Task → patient_type mapping
TASK_PATIENT_MAP = {
    "suppression":   "hypertensive",
    "deterioration": "deteriorating",
}

# Triage patient configs: (patient_type, seed_offset)
TRIAGE_PATIENTS = [
    ("healthy",        0),
    ("post_op",        1),
    ("deteriorating",  2),
    ("healthy",        3),
]

EPISODE_LENGTH = 360
NUM_ACTIONS = 3          # Discrete(3): 0=Ignore, 1=Verify, 2=Alert
HISTORY_LEN = 10         # Number of past timesteps kept in vitals_history


# ------------------------------------------------------------------ #
#  Helper: per-patient observation tracker                            #
# ------------------------------------------------------------------ #

class _PatientTracker:
    """Maintains rolling baseline and vitals history for one patient."""

    def __init__(self, sim: PatientSimulator):
        self.sim = sim
        self.vitals_history: deque = deque(maxlen=HISTORY_LEN)
        # Rolling accumulators — running sum and count for each vital
        self._running_sum = np.zeros(len(VITAL_KEYS), dtype=np.float64)
        self._running_count = 0

    def reset(self, sim: PatientSimulator):
        self.sim = sim
        self.vitals_history.clear()
        self._running_sum[:] = 0.0
        self._running_count = 0

    # ---- normalisation ---- #
    @staticmethod
    def _normalize(raw: Dict[str, float]) -> np.ndarray:
        """Normalize raw vitals dict → numpy array in [0, 1]."""
        arr = np.empty(len(VITAL_KEYS), dtype=np.float64)
        for i, key in enumerate(VITAL_KEYS):
            lo, hi = NORM_RANGES[key]
            arr[i] = np.clip((raw[key] - lo) / (hi - lo), 0.0, 1.0)
        return arr

    # ---- observation building ---- #
    def build_observation(self, step: int) -> Dict:
        """Read current vitals from the simulator and return an obs dict."""
        raw_vitals = self.sim.get_vitals()
        norm = self._normalize(raw_vitals)

        # Update rolling baseline
        self._running_sum += norm
        self._running_count += 1
        rolling_mean = self._running_sum / self._running_count

        # Baseline delta: mean absolute deviation across all 6 vitals
        baseline_delta = float(np.clip(np.mean(np.abs(norm - rolling_mean)), 0.0, 1.0))

        # Update vitals history
        self.vitals_history.append(norm.tolist())

        # Pad history to HISTORY_LEN with zeros
        padded_history = [[0.0] * len(VITAL_KEYS)] * (HISTORY_LEN - len(self.vitals_history))
        padded_history += list(self.vitals_history)

        obs = {
            "heart_rate":       float(norm[0]),
            "systolic_bp":      float(norm[1]),
            "diastolic_bp":     float(norm[2]),
            "spo2":             float(norm[3]),
            "respiratory_rate": float(norm[4]),
            "temperature":      float(norm[5]),
            "baseline_delta":   baseline_delta,
            "hours_observed":   step / 60.0,
            "activity":         int(self.sim.get_activity()),
            "vitals_history":   padded_history,
        }
        return obs


# ------------------------------------------------------------------ #
#  MediGuardEnv                                                      #
# ------------------------------------------------------------------ #

class MediGuardEnv:
    """
    OpenEnv-compliant RL environment for MediGuard-AI.

    Supports three tasks:
      - suppression:   single hypertensive patient, learn to suppress false alarms
      - deterioration: single deteriorating patient, detect sepsis drift
      - triage:        4 concurrent patients, prioritise limited attention

    Actions are Discrete(3) per patient:
      0 = Ignore   1 = Verify   2 = Alert

    Episode length: 360 steps.
    """

    def __init__(self, task: str = "suppression", seed: int = 42):
        assert task in ("suppression", "deterioration", "triage"), (
            f"Unknown task '{task}'. Must be one of: suppression, deterioration, triage."
        )
        self._task = task
        self._seed = seed
        self._step = 0

        # Will be populated in reset()
        self._trackers: List[_PatientTracker] = []
        self._is_triage = (task == "triage")

        # Initialise by calling reset
        self.reset()

    # -------------------------------------------------------------- #
    #  Public API                                                     #
    # -------------------------------------------------------------- #

    def reset(self) -> Union[Dict, List[Dict]]:
        """Reset environment to initial state and return first observation."""
        self._step = 0

        if self._is_triage:
            sims = [
                PatientSimulator(patient_type=pt, seed=self._seed + offset)
                for pt, offset in TRIAGE_PATIENTS
            ]
        else:
            patient_type = TASK_PATIENT_MAP[self._task]
            sims = [PatientSimulator(patient_type=patient_type, seed=self._seed)]

        # Build / reset trackers
        self._trackers = [_PatientTracker(sim) for sim in sims]

        # Advance simulators one tick to get the first set of readings
        for tr in self._trackers:
            tr.sim.tick()

        # Build initial observations
        obs_list = [tr.build_observation(self._step) for tr in self._trackers]

        return obs_list if self._is_triage else obs_list[0]

    def step(self, action: Union[int, List[int]]):
        """
        Execute one environment step.

        Args:
            action: int for single-patient tasks, List[int] of length 4 for triage.

        Returns:
            (observation, reward, done, info)
        """
        # Validate action shape
        if self._is_triage:
            assert isinstance(action, (list, tuple)) and len(action) == len(self._trackers), (
                f"Triage task requires a list of {len(self._trackers)} actions, got {action}"
            )
            actions = list(action)
        else:
            assert isinstance(action, (int, np.integer)), (
                f"Single-patient task requires an int action, got {type(action)}"
            )
            actions = [int(action)]

        # 1. Tick all simulators
        for tr in self._trackers:
            tr.sim.tick()

        # 2. Build observations
        obs_list = [tr.build_observation(self._step) for tr in self._trackers]

        # 3. Compute reward
        obs_for_reward = obs_list if self._is_triage else obs_list[0]
        action_for_reward = actions if self._is_triage else actions[0]
        reward = self._compute_reward(action_for_reward, obs_for_reward)

        # 4. Increment step and check termination
        self._step += 1
        done = self._step >= EPISODE_LENGTH

        # 5. Build info dict
        if self._is_triage:
            patient_types = [pt for pt, _ in TRIAGE_PATIENTS]
        else:
            patient_types = TASK_PATIENT_MAP[self._task]

        info = {
            "step": self._step,
            "patient_type": patient_types,
            "task": self._task,
        }

        obs_out = obs_list if self._is_triage else obs_list[0]
        return obs_out, reward, done, info

    def state(self) -> Dict:
        """Return current environment state for debugging / grading."""
        if self._is_triage:
            patient_type = [pt for pt, _ in TRIAGE_PATIENTS]
            det_severity = [tr.sim.get_state().get("deterioration_severity", 0.0)
                            for tr in self._trackers]
            current_activity = [tr.sim.get_activity() for tr in self._trackers]
        else:
            patient_type = TASK_PATIENT_MAP[self._task]
            det_severity = self._trackers[0].sim.get_state().get("deterioration_severity", 0.0)
            current_activity = self._trackers[0].sim.get_activity()

        return {
            "step": self._step,
            "task": self._task,
            "patient_type": patient_type,
            "done": self._step >= EPISODE_LENGTH,
            "current_activity": current_activity,
            "deterioration_severity": det_severity,
        }

    # -------------------------------------------------------------- #
    #  Reward function (STUB)                                        #
    # -------------------------------------------------------------- #

    def _compute_reward(self, action, obs) -> float:
        # TODO: teammate will implement full reward function here
        # Temporary logic to keep env runnable
        return 0.5

    # -------------------------------------------------------------- #
    #  Graders (STUBS)                                               #
    # -------------------------------------------------------------- #

    def false_alarm_rate_grader(self) -> float:
        # TODO: teammate will implement false_alarm_rate_grader here
        return 0.0

    def deterioration_grader(self) -> float:
        # TODO: teammate will implement deterioration_grader here
        return 0.0

    def triage_grader(self) -> float:
        # TODO: teammate will implement triage_grader here
        return 0.0


# ------------------------------------------------------------------ #
#  Smoke test                                                        #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 65)
    print("MediGuardEnv  —  Smoke Test")
    print("=" * 65)

    for task in ("suppression", "deterioration", "triage"):
        print(f"\n{'─' * 65}")
        print(f"Task: {task}")
        print(f"{'─' * 65}")

        env = MediGuardEnv(task=task, seed=42)
        obs = env.reset()

        for s in range(1, 6):
            if task == "triage":
                action = [1, 0, 1, 0]     # sample multi-patient action
            else:
                action = 1                 # Verify

            obs, reward, done, info = env.step(action)

            if task == "triage":
                # Print summary for patient 0
                p0 = obs[0]
                print(
                    f"  step {s} | reward={reward:.2f} | done={done} | "
                    f"P0 HR={p0['heart_rate']:.3f} SpO2={p0['spo2']:.3f} "
                    f"delta={p0['baseline_delta']:.3f} act={p0['activity']}"
                )
            else:
                print(
                    f"  step {s} | reward={reward:.2f} | done={done} | "
                    f"HR={obs['heart_rate']:.3f} SpO2={obs['spo2']:.3f} "
                    f"delta={obs['baseline_delta']:.3f} act={obs['activity']}"
                )

        st = env.state()
        print(f"  state → step={st['step']}, done={st['done']}, "
              f"deterioration_severity={st['deterioration_severity']}")

    print(f"\n{'=' * 65}")
    print("✅  All smoke tests passed.")
    print(f"{'=' * 65}")
