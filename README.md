# MediGuard-AI — OpenEnv Environment

**Meta PyTorch OpenEnv Hackathon x SST | India AI Hackathon 2026**

> *"We don't ask if 130 is high. We ask if 130 is high for YOU, today."*

---

Imagine a hospital patient hooked up to monitors. The monitor beeps every time heart rate goes above 100. But maybe that's normal for this patient. Nurses hear 200 beeps a day and start ignoring them. Then a real emergency happens — and nobody responds.

We're building an *AI that learns each patient* and only beeps when it actually matters.

---
## What This Is

A fully spec-compliant **OpenEnv environment** for intelligent hospital alarm management. An RL agent learns each patient's personal vital sign baseline and makes context-aware alert decisions — replacing noisy binary threshold alarms with personalized, adaptive intelligence.

This is a **real-world task**, not a game or toy. It simulates ICU-grade patient monitoring with stochastic vitals, camera activity context, and medication feedback loops.

---

## OpenEnv Spec Compliance

| Requirement | Status |
|---|---|
| Typed observation + action models | - |
| `step()` / `reset()` / `state()` API | - |
| `openenv.yaml` config file | - |
| Minimum 3 tasks with agent graders | - |
| Reward: 0.0 – 1.0 with partial signals | - |
| Baseline inference script | - |
| Hugging Face Spaces deployment | - |
| Working Dockerfile | - |

---

## Environment Description

The agent monitors a single ICU patient. It observes a stream of vital signs (HR, BP, SpO2, RR, Temp) and camera-inferred activity context (eating, resting, ambulating, falling). Each timestep, it decides:

- 🔴 **Alert** — real emergency, notify staff immediately
- 🟡 **Verify** — soft nudge, nurse checks when free
- 🟢 **Ignore** — suppress alarm, all within expected range

The patient's baseline is learned over the first 72 simulation hours. What counts as "dangerous" is personal — a resting HR of 110 is alarming; a post-meal HR of 110 is not.

---

## Observation Space

```python
{
  # Vitals (float32, normalized 0–1)
  "heart_rate":        float,   # bpm, range 30–200
  "systolic_bp":       float,   # mmHg, range 60–220
  "diastolic_bp":      float,   # mmHg, range 30–140
  "spo2":              float,   # %, range 70–100
  "respiratory_rate":  float,   # breaths/min, range 5–40
  "temperature":       float,   # °C, range 34–42

  # Personalization
  "baseline_delta":    float,   # deviation from learned personal baseline
  "hours_observed":    float,   # hours of baseline data collected so far

  # Camera context (one-hot encoded)
  "activity":          int,     # 0=resting, 1=eating, 2=ambulating, 3=distressed, 4=falling

  # History (last 10 timesteps, each vital)
  "vitals_history":    float[10][6]
}
```

## Action Space

```python
Discrete(3)
# 0 → Ignore   (suppress alarm)
# 1 → Verify   (silent nudge to nurse)
# 2 → Alert    (immediate alarm)
```

---

## Tasks

### Task 1 — Alarm Suppression (Easy) `score: 0.0–1.0`

**Goal**: Reduce false alarms for a stable patient with known high baseline BP.

The patient has chronic hypertension. BP of 150 is normal for them. A naive threshold agent fires constantly. The RL agent must learn to ignore non-events.

**Grader**: `false_alarm_rate_grader` — scores 1.0 when false alarm rate drops below 5%, partial credit proportional to reduction from baseline.

---

### Task 2 — Early Deterioration Detection (Medium) `score: 0.0–1.0`

**Goal**: Catch a slow-onset sepsis event over 6 hours before it becomes critical.

Vitals drift gradually. No single reading crosses a hard threshold. The agent must learn to Verify early, then Alert as the pattern becomes undeniable.

**Grader**: `deterioration_grader` — scores 1.0 for Verify before hour 3 and Alert before hour 5. Partial credit for late detection. Zero for missed event.

---

### Task 3 — Activity-Aware Multi-Patient Triage (Hard) `score: 0.0–1.0`

**Goal**: Manage alarms for 4 patients simultaneously with limited nurse capacity.

One patient is exercising (high HR expected). One is post-op (low BP expected). One is genuinely deteriorating. One is stable. The agent must route attention correctly using camera context.

**Grader**: `triage_grader` — scores on (correct prioritization × nurse response time × false alarm count). Partial credit at each decision checkpoint.

---

## Reward Function

The reward at each timestep is a weighted sum designed to prevent alarm fatigue while preserving safety:

```
R(t) = w1 * true_positive_signal
      - w2 * false_alarm_penalty
      + w3 * early_detection_bonus
      - w4 * missed_event_penalty
      + w5 * baseline_learning_progress

Default weights: w1=1.0, w2=0.3, w3=0.5, w4=2.0, w5=0.1
```

**Partial progress signals are always on.** The agent gets small positive rewards for improving its baseline estimate, even before it makes any alert decisions. Missed critical events carry the heaviest penalty (`w4=2.0`).

---

## Project Structure

```
MediGuard-AI/
├── openenv.yaml                  # OpenEnv spec config
├── Dockerfile                    # Deployment container
├── requirements.txt
│
├── env/
│   ├── mediguard_env.py          # Core environment: step() / reset() / state()
│   ├── models.py                 # Typed observation + action models (Pydantic)
│   ├── patient_simulator.py      # Stochastic vitals + activity generator
│   └── baseline_tracker.py      # Personal baseline learning module
│
├── tasks/
│   ├── task1_suppression.py      # Easy task + false_alarm_rate_grader
│   ├── task2_deterioration.py    # Medium task + deterioration_grader
│   └── task3_triage.py           # Hard task + triage_grader
│
├── agents/
│   ├── alert_agent/              # PPO/DQN agent for 3-action decisions
│   └── dosing_agent/             # Medication adjustment agent
│
├── baseline/
│   └── run_baseline.py           # Reproducible baseline inference script
│
├── app/
│   ├── dashboard/app.py          # Gradio monitoring dashboard
│   └── api/main.py               # FastAPI serving endpoint
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_baseline_simulation.ipynb
    ├── 03_rl_training.ipynb
    └── 04_demo.ipynb
```

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA 12.1 (RTX 4060 or better recommended)
- Miniconda

### Install

```bash
git clone https://github.com/SutikshanUpman/MediGuard-AI.git
cd MediGuard-AI

conda create -n mediguard python=3.11
conda activate mediguard

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Run Baseline

```bash
python baseline/run_baseline.py --task all --seed 42
```

Expected output:
```
Task 1 (Suppression):    score = 0.71
Task 2 (Deterioration):  score = 0.54
Task 3 (Triage):         score = 0.38
```

### Docker

```bash
docker build -t mediguard-env .
docker run -p 7860:7860 mediguard-env
```

---

## openenv.yaml

```yaml
name: MediGuard-AI
version: 1.0.0
description: >
  ICU patient monitoring environment for intelligent alarm management.
  Agent observes vitals + camera context and decides: Alert / Verify / Ignore.

observation_space: mediguard_env.ObservationModel
action_space: Discrete(3)

tasks:
  - id: task1_suppression
    difficulty: easy
    grader: tasks.task1_suppression.false_alarm_rate_grader
  - id: task2_deterioration
    difficulty: medium
    grader: tasks.task2_deterioration.deterioration_grader
  - id: task3_triage
    difficulty: hard
    grader: tasks.task3_triage.triage_grader

reward:
  range: [0.0, 1.0]
  partial_signals: true

deployment:
  platform: huggingface-spaces
  framework: gradio
  dockerfile: Dockerfile
```

---

## Why RL for This?

Every patient is different. Vital sign consequences are delayed. Conditions are non-stationary. This is exactly what RL is built for:

- **Personalization** — learns individual patient patterns, not textbook thresholds
- **Delayed rewards** — handles time-lagged consequences (sepsis develops over hours)
- **Non-stationarity** — adapts as patient condition evolves post-surgery, post-medication
- **Multi-objective** — balances safety (catch real events) vs. fatigue (suppress noise)

---

## Deployment

Live on Hugging Face Spaces: `huggingface.co/spaces/SutikshanUpman/MediGuard-AI`

---

**Built with ❤️ for better healthcare**
