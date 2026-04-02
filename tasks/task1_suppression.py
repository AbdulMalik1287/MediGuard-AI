import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.reward_function import RewardFunction, Action, PatientCondition


# ── Task Definition ──────────────────────────────────────────────────────────

TASK1_CONFIG = {
    "task_id":      "task1_suppression",
    "difficulty":   "easy",
    "description":  "Reduce false alarms for a stable patient with known high baseline BP.",
    "max_steps":    200,
    "patient_type": "hypertensive_stable",
}


# ── Grader ────────────────────────────────────────────────────────────────────

class false_alarm_rate_grader:
    """
    Scores 1.0 when false alarm rate drops below 5%.
    Partial credit proportional to reduction from naive baseline.
    
    A naive threshold agent fires alarms ~60% of the time on this patient
    because their BP of 150 constantly crosses the textbook threshold of 140.
    The RL agent must learn to suppress these.
    """

    # A naive threshold agent produces this many false alarms per 100 steps
    # Used as the baseline to compute partial credit
    NAIVE_FALSE_ALARM_RATE = 0.60

    # Below this rate → full score
    TARGET_FALSE_ALARM_RATE = 0.05

    def score(self, stats: dict, history: list) -> float:
        total_steps     = stats["total_steps"]
        false_positives = stats["false_positives"]

        if total_steps == 0:
            return 0.0

        actual_rate = false_positives / total_steps

        # Full score if below target
        if actual_rate <= self.TARGET_FALSE_ALARM_RATE:
            return 1.0

        # Zero score if as bad as naive agent or worse
        if actual_rate >= self.NAIVE_FALSE_ALARM_RATE:
            return 0.0

        # Partial credit: linear interpolation between naive and target
        # e.g. if naive=0.60, target=0.05, actual=0.30
        # improvement = (0.60 - 0.30) / (0.60 - 0.05) = 0.545
        improvement = (self.NAIVE_FALSE_ALARM_RATE - actual_rate)
        possible    = (self.NAIVE_FALSE_ALARM_RATE - self.TARGET_FALSE_ALARM_RATE)

        score = improvement / possible

        # Safety check: if it suppressed false alarms but also missed
        # real emergencies, penalize
        fn_penalty = min(0.3, stats["false_negatives"] * 0.1)

        return max(0.0, min(1.0, score - fn_penalty))
