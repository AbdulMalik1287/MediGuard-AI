# 🏥 MediGuard-AI

**An AI-powered ICU patient monitoring system built for the Meta PyTorch OpenEnv Hackathon 2026.**

MediGuard-AI uses reinforcement learning to teach an AI agent when to raise alarms about ICU patients — learning to catch real emergencies while ignoring false alarms.

---

## 🧠 The Simple Explanation

Imagine you're a nurse watching heart monitors for 4 ICU patients at once. Every minute, you see their heart rate, blood pressure, oxygen levels, etc. You have three choices:

| Action | What it means |
|--------|--------------|
| **Ignore** | "Everything looks fine, carry on." |
| **Verify** | "Something seems off, let me take a closer look." |
| **Alert** | "This is serious — get a doctor NOW!" |

The challenge? Some patients naturally have high blood pressure (it's their normal). Some are walking around (so of course their heart rate is up). A bad AI would panic at every spike. A good AI learns each patient's **personal baseline** and only alerts when something is *genuinely* wrong.

**MediGuard-AI trains an RL agent to make these decisions smartly across 3 scenarios:**

1. **Suppression** (Easy) — A patient with chronically high blood pressure. The AI must learn that 150/95 is *normal for them* and stop crying wolf.
2. **Deterioration** (Medium) — A patient slowly developing sepsis over 6 hours. Vitals drift so gradually a human might miss it. The AI must catch it early.
3. **Triage** (Hard) — 4 patients at once. The AI must figure out who actually needs attention and who is fine.

---

## 📁 Project Structure

```
RL Hackathon/
├── patient_simulator.py   ← Generates realistic patient vital signs (Made By Sutikshan, main helping code in his branch)
├── mediguard_env.py       ← RL environment (the "game" the AI plays)
├── inference.py           ← Baseline inference script with structured logs
├── server.py              ← FastAPI server for HF Space deployment
├── openenv.yaml           ← OpenEnv spec metadata
├── Dockerfile             ← Container build for HF Spaces
├── requirements.txt       ← Python dependencies
└── README.md              ← You are here
```

---

## 📄 File 1: `patient_simulator.py`

### In Plain English
This is a **virtual patient**. It generates realistic vital signs (heart rate, blood pressure, oxygen, etc.) that change over time — just like a real ICU patient. It simulates different patient types: healthy, high blood pressure, slowly getting sicker, post-surgery recovery, and unstable.

### Technical Details

| Component | Description |
|-----------|-------------|
| **Class** | `PatientSimulator` |
| **Input** | `patient_type` (str), `seed` (int) |
| **Output** | Vital signs dict with 6 keys |

**Key methods:**

```python
sim = PatientSimulator(patient_type="healthy", seed=42)

sim.get_vitals()    # → dict with heart_rate, systolic_bp, diastolic_bp, spo2, respiratory_rate, temperature
sim.get_activity()  # → int: 0=resting, 1=eating, 2=ambulating, 3=distressed, 4=falling
sim.tick()          # Advances time by 1 step, updates all vitals
sim.get_state()     # → full debug state dict
sim.reset()         # Resets to initial conditions
```

**Patient types and their clinical profiles:**

| Type | Baseline BP | Clinical Scenario |
|------|------------|-------------------|
| `healthy` | 120/80 | Normal vitals, small random noise |
| `hypertensive` | 150/95 | Chronically elevated BP (their normal) |
| `deteriorating` | 118/78 → declining | Slow sepsis drift: temp↑ HR↑ BP↓ SpO2↓ over 360 steps |
| `post_op` | 100/65 | Low BP, recovering from surgery |
| `unstable` | 125/82 | Random spikes and drops (10% chance per step) |

**How vitals are generated each tick:**
1. Fresh baseline values + Gaussian noise
2. Activity effects applied (walking raises HR, etc.)
3. Deterioration drift applied (for deteriorating/unstable types)
4. Smoothing: 70% previous + 30% new (prevents unrealistic jumps)
5. Clipped to physiological limits (HR: 30–200, SpO2: 70–100, etc.)

---

## 📄 File 2: `mediguard_env.py`

### In Plain English
This is the **game board**. It wraps the patient simulator into a standard RL environment. Every "turn," the AI agent sees the patient's vitals and decides: Ignore, Verify, or Alert. The environment scores the decision and moves time forward. After 360 turns (representing 6 hours), the episode ends.

It also tracks the patient's **personal baseline** — a running average of their vitals — so the AI can tell "this is unusual *for this specific patient*" rather than just "this number is high."

### Technical Details

**Class:** `MediGuardEnv` — OpenEnv-compliant with 3 public methods.

```python
env = MediGuardEnv(task="suppression", seed=42)

obs = env.reset()                          # → observation dict
obs, reward, done, info = env.step(action) # → (obs, reward, done, info)
state = env.state()                        # → debug state dict
```

**Pydantic Models (OpenEnv spec):**
- `ObservationModel` — validates observation dict schema
- `ActionModel` — validates action schema (int or List[int])

**Observation Space — 10 fields per patient:**

| Key | Type | Description |
|-----|------|-------------|
| `heart_rate` | float [0,1] | Normalized HR. Raw range: 30–200 bpm |
| `systolic_bp` | float [0,1] | Normalized systolic BP. Raw: 60–220 mmHg |
| `diastolic_bp` | float [0,1] | Normalized diastolic BP. Raw: 30–140 mmHg |
| `spo2` | float [0,1] | Normalized oxygen saturation. Raw: 70–100% |
| `respiratory_rate` | float [0,1] | Normalized resp rate. Raw: 5–40 breaths/min |
| `temperature` | float [0,1] | Normalized temp. Raw: 34–42°C |
| `baseline_delta` | float [0,1] | Mean abs deviation from personal rolling baseline |
| `hours_observed` | float | `step / 60.0` — how long we've been watching |
| `activity` | int {0–4} | What the patient is doing right now |
| `vitals_history` | list [10][6] | Last 10 timesteps of normalized vitals (zero-padded) |

**Normalization formula:**
```
normalized = clip((raw - min) / (max - min), 0, 1)
```

**Baseline Delta computation:**
```
For each vital: track running_mean of normalized values over all steps so far
baseline_delta = mean(|current_normalized - running_mean|) across all 6 vitals
Clipped to [0, 1]
```

**Action Space:** `Discrete(3)` — `0`=Ignore, `1`=Verify, `2`=Alert

**Task configurations:**

| Task | # Patients | Patient Type(s) | Action Shape | Obs Shape |
|------|-----------|-----------------|-------------|-----------|
| `suppression` | 1 | hypertensive | `int` | `dict` |
| `deterioration` | 1 | deteriorating | `int` | `dict` |
| `triage` | 4 | healthy, post_op, deteriorating, healthy | `List[int]` len 4 | `List[dict]` len 4 |

**Episode length:** 360 steps (done=True when `step >= 360`)

---

## 📄 File 3: `inference.py`

### In Plain English
This is the **test run**. It plays the game using a simple rule-based strategy (no neural network needed). It runs all 3 tasks, and for each one, it prints detailed logs in a strict format that the hackathon's automated scoring system reads.

The rule-based agent thinks like a cautious nurse:
- *First hour?* → Always verify (still learning the patient's baseline)
- *Patient is walking?* → Ignore (elevated vitals are expected)
- *Big deviation while resting?* → Alert!
- *Oxygen dropping dangerously?* → Alert!
- *Moderate deviation while resting?* → Verify (check on them)
- *Otherwise?* → Ignore

### Technical Details

**Environment variables (read at startup):**
```
API_BASE_URL  — LLM endpoint (default: https://router.huggingface.co/v1)
MODEL_NAME    — model ID (default: Qwen/Qwen2.5-72B-Instruct)
HF_TOKEN / API_KEY — auth token
```

**OpenAI client is initialized** per hackathon rules, ready for LLM-based agents.

**Baseline agent decision tree:**
```python
if hours < 1.0:           return VERIFY    # still learning baseline
if activity == 2:          return IGNORE    # ambulating — high HR expected
if delta > 0.6 and rest:   return ALERT     # big deviation at rest
if spo2 < 0.3:             return ALERT     # oxygen critically low
if delta > 0.35 and rest:  return VERIFY    # moderate deviation at rest
return IGNORE
```

**Mandatory log format (one line per event, no deviations):**
```
[START] task=suppression env=mediguard model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=1 reward=0.50 done=false error=null
...
[STEP] step=360 action=0 reward=0.50 done=true error=null
[END] success=true steps=360 rewards=0.50,0.50,...,0.50
```

For triage, actions are comma-separated: `action=1,0,2,0`

**Success criterion:** `mean(rewards) > 0.4`

---

## 📄 File 4: `server.py`

### In Plain English
This turns the environment into a **web server** that can run on HuggingFace Spaces. The hackathon's automated validator pings your server — it needs to respond to REST calls like `/reset` and `/step`.

### Technical Details
FastAPI server exposing the environment's 3 methods as HTTP endpoints:

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/` | GET | Landing page with env metadata |
| `/health` | GET | Health check (returns `{"status": "ok"}`) |
| `/reset` | POST | Reset env with `{"task": "suppression", "seed": 42}` |
| `/step` | POST | Take action with `{"action": 1}` or `{"action": [1,0,2,0]}` |
| `/state` | GET | Get current environment state |

---

## 📄 File 5: `openenv.yaml`

### In Plain English
A metadata file that tells the OpenEnv framework what our environment is, what tasks it has, and where to find the code.

### What it contains
- Environment name, description, version
- Module and class references (`mediguard_env.MediGuardEnv`)
- All 3 tasks with descriptions, difficulty levels, and grader names
- Action and observation space schemas

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Smoke-test the environment (5 steps per task)

```bash
python mediguard_env.py
```

### 3. Run the full baseline inference (360 steps × 3 tasks)

```bash
python inference.py
```

### 4. Run the server locally

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

Then test: `curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{}"`

### 5. Docker build & run

```bash
docker build -t mediguard-ai .
docker run -p 7860:7860 mediguard-ai
```

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    inference.py                          │
│  ┌────────────┐    ┌───────────────────────────────┐    │
│  │  Baseline   │───▶│       MediGuardEnv            │    │
│  │   Agent     │◀───│  (mediguard_env.py)           │    │
│  │             │    │                               │    │
│  │ Rules:      │    │  • Normalizes vitals 0–1      │    │
│  │ • delta>0.6 │    │  • Tracks personal baseline   │    │
│  │   → ALERT   │    │  • Keeps 10-step history      │    │
│  │ • spo2<0.3  │    │  • Computes reward (STUB)     │    │
│  │   → ALERT   │    │  • Manages 360-step episodes  │    │
│  └────────────┘    │                               │    │
│                     │  ┌───────────────────────┐    │    │
│  ┌────────────┐    │  │  PatientSimulator     │    │    │
│  │  OpenAI    │    │  │  (patient_simulator.py)│    │    │
│  │  Client    │    │  │                       │    │    │
│  │ (ready for │    │  │  Generates vitals     │    │    │
│  │  LLM agent)│    │  │  per timestep         │    │    │
│  └────────────┘    │  └───────────────────────┘    │    │
│                     └───────────────────────────────┘    │
│                                                          │
│  Output: [START] / [STEP] / [END] logs                   │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  server.py (FastAPI)          │  openenv.yaml           │
│  POST /reset, /step           │  Metadata + task defs   │
│  GET  /state, /health         │                         │
│  Served via Dockerfile        │                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔴 Things To Be Done By Abdul Bhai

> **⚠️ CRITICAL: These stubs MUST be replaced before submission. Graders that always return the same score = automatic disqualification.**

### 1. Reward Function — `mediguard_env.py` line ~195

**Where:** `mediguard_env.py` → method `_compute_reward(self, action, obs)`

**Current stub:**
```python
def _compute_reward(self, action, obs) -> float:
    # TODO: teammate will implement full reward function here
    # Temporary logic to keep env runnable
    return 0.5
```

**What it needs to do:**
- Return a float reward based on the action taken and current observation
- Provide **varying signal** — NOT a constant! (constant reward = disqualification)
- Reward partial progress (e.g., correctly ignoring a stable patient, catching early deterioration)
- Penalize clearly wrong actions (e.g., alerting on a healthy resting patient, ignoring dropping SpO2)
- Score should be between 0.0 and 1.0

**Suggested reward logic:**
- **Ignore + stable patient** → small positive reward (correct)
- **Ignore + deteriorating patient** → penalty (missed it)
- **Alert + genuinely abnormal vitals** → large positive reward
- **Alert + normal vitals** → penalty (false alarm)
- **Verify** → small cost (it takes nurse time) but no penalty for being cautious

**Inputs available:**
- `action` — int (0/1/2) or list of ints for triage
- `obs` — observation dict with all 10 fields (delta, spo2, activity, etc.)

---

### 2. False Alarm Rate Grader — `mediguard_env.py` line ~201

**Where:** `mediguard_env.py` → method `false_alarm_rate_grader(self)`

**Current stub:**
```python
def false_alarm_rate_grader(self) -> float:
    # TODO: teammate will implement false_alarm_rate_grader here
    return 0.0
```

**What it needs to do:**
- Score 0.0–1.0 for the **suppression** task
- Measure: how well did the agent suppress false alarms on the hypertensive patient?
- 1.0 = zero false alarms, correctly identified all normal readings
- 0.0 = alerted on everything (panicked)
- You'll need to track alert counts during the episode — add counters in `step()`

---

### 3. Deterioration Grader — `mediguard_env.py` line ~205

**Where:** `mediguard_env.py` → method `deterioration_grader(self)`

**Current stub:**
```python
def deterioration_grader(self) -> float:
    # TODO: teammate will implement deterioration_grader here
    return 0.0
```

**What it needs to do:**
- Score 0.0–1.0 for the **deterioration** task
- Measure: how early and accurately did the agent detect sepsis onset?
- 1.0 = alerted before severity reached 0.5
- 0.0 = never alerted or alerted way too late
- Use `sim.get_state()["deterioration_severity"]` for timing

---

### 4. Triage Grader — `mediguard_env.py` line ~209

**Where:** `mediguard_env.py` → method `triage_grader(self)`

**Current stub:**
```python
def triage_grader(self) -> float:
    # TODO: teammate will implement triage_grader here
    return 0.0
```

**What it needs to do:**
- Score 0.0–1.0 for the **triage** task
- Measure: did the agent correctly prioritize the 4 patients?
- Patient 2 (deteriorating) should get the most alerts
- Patients 0, 3 (healthy) should get the fewest
- Patient 1 (post_op) somewhere in between

---

### Quick Find: Search for `# TODO: teammate` in `mediguard_env.py` to find all 4 stubs.

---

## ✅ Hackathon Checklist

| Requirement | Status |
|------------|--------|
| Real-world task simulation (ICU monitoring) | ✅ Done |
| OpenEnv spec: typed Pydantic models | ✅ Done |
| OpenEnv spec: `step()` / `reset()` / `state()` | ✅ Done |
| `openenv.yaml` with metadata | ✅ Done |
| 3 tasks (easy → medium → hard) | ✅ Done |
| Agent graders (0.0–1.0 scores) | ⏳ **Abdul Bhai** |
| Meaningful reward function (varying signal) | ⏳ **Abdul Bhai** |
| Baseline `inference.py` with reproducible scores | ✅ Done |
| Structured stdout: `[START]` / `[STEP]` / `[END]` | ✅ Done |
| OpenAI client for LLM calls | ✅ Done |
| Environment variables: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` | ✅ Done |
| `Dockerfile` builds | ✅ Done |
| FastAPI server with `/reset` endpoint | ✅ Done |
| `requirements.txt` | ✅ Done |
| Runs on vcpu=2, 8GB RAM, no GPU | ✅ Done |
| Completes in under 20 minutes | ✅ Done |
| `seed=42` reproducible | ✅ Done |
| Deploy to HuggingFace Spaces | 🔜 After Abdul Bhai's changes |

---

## 📋 Constraints & Requirements

- **Hardware:** vcpu=2, 8GB RAM — no GPU required
- **Runtime:** Must complete all 3 tasks in under 20 minutes
- **Reproducibility:** `seed=42` must produce identical output every run
- **Python version:** 3.9+
- **Dependencies:** `numpy`, `pydantic`, `openai`, `fastapi`, `uvicorn`
