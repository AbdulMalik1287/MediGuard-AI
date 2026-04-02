from dataclasses import dataclass
from typing import List
from enum import Enum


# --- These enums define your action and ground truth types ---

class Action(Enum):
    ALERT          = "alert"
    VERIFY         = "verify"
    IGNORE         = "ignore"
    SWITCH_PATIENT = "switch_patient"


class PatientCondition(Enum):
    EMERGENCY     = "emergency"      # needs immediate action
    BORDERLINE    = "borderline"     # elevated but not critical
    STABLE        = "stable"         # all good
    DRUG_MASKED   = "drug_masked"    # emergency hidden by medication effect


# --- This dataclass tracks the history of one episode ---
#reward function needs history to compute the alarm fatigue modifier

@dataclass
class StepRecord:
    step:          int
    action:        Action
    condition:     PatientCondition
    reward:        float


class RewardFunction:
    """
    Computes per-step reward for MediGuard agent.
    
    Usage:
        rf = RewardFunction()
        reward = rf.compute(action, condition, step_number)
    
    Call rf.reset() at the start of each episode.
    """

    # --- Base reward table ---
    # Rows = actions, Cols = patient conditions
    BASE_REWARDS = {
        Action.ALERT: {
            PatientCondition.EMERGENCY:   +1.0,
            PatientCondition.BORDERLINE:  +0.2,   # not wrong, but cautious
            PatientCondition.STABLE:      -0.5,   # false alarm
            PatientCondition.DRUG_MASKED: +1.0,   # correctly caught masked emergency
        },
        Action.VERIFY: {
            PatientCondition.EMERGENCY:   -0.8,   # too slow
            PatientCondition.BORDERLINE:  +0.4,   # exactly right
            PatientCondition.STABLE:      +0.1,   # slightly overcautious but ok
            PatientCondition.DRUG_MASKED: -0.5,   # still too slow for masked emergency
        },
        Action.IGNORE: {
            PatientCondition.EMERGENCY:   -2.0,   # worst outcome — never do this
            PatientCondition.BORDERLINE:  -0.1,   # slightly risky
            PatientCondition.STABLE:      +0.2,   # correct, quiet
            PatientCondition.DRUG_MASKED: -2.0,   # missed masked emergency = same as missing real one
        },
        Action.SWITCH_PATIENT: {
            PatientCondition.EMERGENCY:   -0.3,   # abandoning someone critical
            PatientCondition.BORDERLINE:  +0.0,   # neutral
            PatientCondition.STABLE:      +0.1,   # fine to move on
            PatientCondition.DRUG_MASKED: -0.3,   # abandoning masked emergency
        },
    }

    # How many recent steps to look back for alarm fatigue
    FATIGUE_WINDOW     = 30
    FATIGUE_THRESHOLD  = 5     # if more than this many ALERTs in window...
    FATIGUE_MULTIPLIER = 0.6   # ...multiply ALERT reward by this

    # When personalization bonus kicks in (step number)
    BASELINE_LEARNED_AFTER = 200
    PERSONALIZATION_BONUS  = 0.2

    def __init__(self):
        self.history: List[StepRecord] = []

    def reset(self):
        """Call this at the start of every new episode."""
        self.history = []

    def compute(
        self,
        action:           Action,
        condition:        PatientCondition,
        step_number:      int,
        is_patient_normal_for_them: bool = False,
        # ^ True when vitals look abnormal by textbook but are
        #   normal for this specific patient. Passed in by the
        #   environment based on the baseline tracker.
    ) -> float:

        # Step 1: Get the base reward from the table
        base = self.BASE_REWARDS[action][condition]

        # Step 2: Apply alarm fatigue modifier
        # Count how many ALERTs happened in the last FATIGUE_WINDOW steps
        recent_steps   = self.history[-self.FATIGUE_WINDOW:]
        recent_alerts  = sum(1 for s in recent_steps if s.action == Action.ALERT)

        fatigue_modifier = 1.0   # default: no modification
        if action == Action.ALERT and recent_alerts >= self.FATIGUE_THRESHOLD:
            fatigue_modifier = self.FATIGUE_MULTIPLIER
            # Note: this only reduces reward, never makes it negative
            # A correct ALERT on real emergency still gets +0.6 even when fatigued
            # because missing emergencies is always worse than alarm fatigue

        reward = base * fatigue_modifier

        # Step 3: Personalization bonus
        # After the baseline learning period, reward the agent for correctly
        # ignoring things that look abnormal by textbook but are normal for
        # this specific patient
        if (
            step_number > self.BASELINE_LEARNED_AFTER
            and action == Action.IGNORE
            and is_patient_normal_for_them
            and condition == PatientCondition.STABLE
        ):
            reward += self.PERSONALIZATION_BONUS

        # Step 4: Record this step in history
        record = StepRecord(
            step=step_number,
            action=action,
            condition=condition,
            reward=reward,
        )
        self.history.append(record)

        return reward

    def get_episode_stats(self) -> dict:
        """
        Returns a summary of the episode.
        Used by the graders — call this at episode end.
        """
        total_steps    = len(self.history)
        total_reward   = sum(s.reward for s in self.history)

        alerts         = [s for s in self.history if s.action == Action.ALERT]
        ignores        = [s for s in self.history if s.action == Action.IGNORE]

        true_positives  = sum(1 for s in alerts  if s.condition in (PatientCondition.EMERGENCY, PatientCondition.DRUG_MASKED))
        false_positives = sum(1 for s in alerts  if s.condition == PatientCondition.STABLE)
        false_negatives = sum(1 for s in ignores if s.condition in (PatientCondition.EMERGENCY, PatientCondition.DRUG_MASKED))
        true_negatives  = sum(1 for s in ignores if s.condition == PatientCondition.STABLE)

        return {
            "total_steps":    total_steps,
            "total_reward":   total_reward,
            "true_positives":  true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives":  true_negatives,
        }
