import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.reward_function import RewardFunction, Action, PatientCondition


TASK2_CONFIG = {
    "task_id":      "task2_deterioration",
    "difficulty":   "medium",
    "description":  "Detect slow-onset sepsis over 6 hours before it becomes critical.",
    "max_steps":    360,   # 6 hours at 1 step per minute
    "patient_type": "sepsis_onset",
}


class deterioration_grader:
    """
    Scores 1.0 for:  VERIFY before step 180 (hour 3) AND ALERT before step 300 (hour 5)
    Partial credit:  late detection still scores, but lower
    Zero:            missed the event entirely
    
    This grader specifically rewards the two-phase response:
    1. Soft concern early (VERIFY) — "something might be happening"
    2. Definitive alarm later (ALERT) — "this is real"
    """

    # Step numbers (1 step = 1 minute of simulated time)
    EARLY_VERIFY_DEADLINE  = 180   # hour 3
    ALERT_DEADLINE_FULL    = 300   # hour 5 — full score
    ALERT_DEADLINE_PARTIAL = 360   # hour 6 — partial score

    def score(self, stats: dict, history: list) -> float:

        # Find the first VERIFY and first ALERT during a deteriorating condition
        first_verify_step = None
        first_alert_step  = None

        for record in history:
            if record.condition == PatientCondition.EMERGENCY:
                if record.action == Action.VERIFY and first_verify_step is None:
                    first_verify_step = record.step
                if record.action == Action.ALERT and first_alert_step is None:
                    first_alert_step = record.step

        # Case 1: Missed event entirely — no ALERT ever
        if first_alert_step is None:
            return 0.0

        # Case 2: Compute alert component (0.0 – 0.7)
        if first_alert_step <= self.ALERT_DEADLINE_FULL:
            alert_score = 0.7   # caught it in time
        elif first_alert_step <= self.ALERT_DEADLINE_PARTIAL:
            # Partial credit — linear decay from 0.7 to 0.3
            steps_late  = first_alert_step - self.ALERT_DEADLINE_FULL
            max_late    = self.ALERT_DEADLINE_PARTIAL - self.ALERT_DEADLINE_FULL
            alert_score = 0.7 - (0.4 * steps_late / max_late)
        else:
            alert_score = 0.1   # very late but technically caught it

        # Case 3: Early VERIFY bonus (0.0 – 0.3)
        if first_verify_step and first_verify_step <= self.EARLY_VERIFY_DEADLINE:
            verify_bonus = 0.3
        elif first_verify_step:
            verify_bonus = 0.1   # verified but late
        else:
            verify_bonus = 0.0   # never verified, went straight to alert

        score = alert_score + verify_bonus
        return max(0.0, min(1.0, score))
