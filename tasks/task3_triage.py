import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.reward_function import RewardFunction, Action, PatientCondition


TASK3_CONFIG = {
    "task_id":      "task3_triage",
    "difficulty":   "hard",
    "description":  "Manage alarms for 4 patients with limited nurse capacity.",
    "max_steps":    400,
    "patient_type": "multi_patient",
    "n_patients":   4,
}


class triage_grader:
    """
    Three components, weighted:
    
    50% — Sensitivity + Specificity (F1) across all patients
    30% — Did it catch the activity-masked emergency?
           (patient exercising with high HR — is it exercise or cardiac event?)
    20% — Triage priority: did it attend to critical patients over stable ones?
    
    This is genuinely hard because:
    - One patient has high HR from exercise → should IGNORE
    - One patient has high HR from cardiac event → should ALERT
    - The vitals look identical, only activity context differentiates them
    """

    def score(self, stats: dict, history: list) -> float:

        # Component 1: F1 score (50%)
        f1 = self._compute_f1(stats)

        # Component 2: Activity-masked detection (30%)
        # Your teammate needs to tag DRUG_MASKED records in history
        # for the exercising-but-actually-cardiac patient
        masked_score = self._compute_masked_detection(history)

        # Component 3: Triage priority (20%)
        triage_score = self._compute_triage_priority(history)

        final = (0.5 * f1) + (0.3 * masked_score) + (0.2 * triage_score)
        return max(0.0, min(1.0, final))

    def _compute_f1(self, stats: dict) -> float:
        tp = stats["true_positives"]
        fn = stats["false_negatives"]
        fp = stats["false_positives"]
        tn = stats["true_negatives"]

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if sensitivity + specificity == 0:
            return 0.0
        return 2 * (sensitivity * specificity) / (sensitivity + specificity)

    def _compute_masked_detection(self, history: list) -> float:
        masked_steps  = [s for s in history if s.condition == PatientCondition.DRUG_MASKED]
        masked_alerts = [s for s in masked_steps if s.action == Action.ALERT]

        if len(masked_steps) == 0:
            return 1.0   # no masked patient in this episode
        return 1.0 if len(masked_alerts) > 0 else 0.0

    def _compute_triage_priority(self, history: list) -> float:
        switches = [s for s in history if s.action == Action.SWITCH_PATIENT]
        if len(switches) == 0:
            return 0.5

        good = sum(1 for s in switches if s.condition == PatientCondition.STABLE)
        bad  = sum(1 for s in switches if s.condition in (
            PatientCondition.EMERGENCY, PatientCondition.DRUG_MASKED
        ))
        total = good + bad
        return good / total if total > 0 else 0.5
```

---

## The Import Chain (Share This With Your Teammates)

This is how all files connect:
```
env/mediguard_env.py          ← your teammate's file
    imports → tasks/reward_function.py     ← YOUR file
    
tasks/task1_suppression.py    ← YOUR file
    imports → tasks/reward_function.py     ← YOUR file

tasks/task2_deterioration.py  ← YOUR file  
    imports → tasks/reward_function.py     ← YOUR file

tasks/task3_triage.py         ← YOUR file
    imports → tasks/reward_function.py     ← YOUR file

baseline/run_baseline.py      ← third teammate's file
    imports → tasks/task1_suppression.false_alarm_rate_grader
    imports → tasks/task2_deterioration.deterioration_grader
    imports → tasks/task3_triage.triage_grader
