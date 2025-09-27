Here’s a **documentation-style implementation plan** tailored for your project **therapist-ai-core**, using `virtualenv` instead of conda, with everything structured and reproducible.

---

# 📘 Therapist AI Core — Implementation Plan

## 1. Project Overview

**Goal:**
Build an AI agent that:

1. Chats with patients, asking structured follow-up questions to collect required data.
2. At the end, generates a JSON report matching a strict schema.
3. Stores both the chat transcript and final report in the same format as training data, so they can be reused for further fine-tuning.

**Architecture:**

- **One fine-tuned LLM (LoRA SFT)** handles both chat and report generation.
- **Two prompts (“modes”)**:

  - _Chat Mode_: collects missing info (slot filling).
  - _Report Mode_: generates strict schema-valid JSON from collected data.

- **Terminal UI** for live conversation/testing.

---

## 2. Setup

### Virtual Environment

```bash
# Create virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install torch transformers datasets accelerate peft trl bitsandbytes jsonschema rapidfuzz lm-format-enforcer
```

---

## 3. Data Format

### Training / Validation Data (`data/train.jsonl`)

Each line = one case.

```json
{
  "id": "ex-001",
  "messages": [
    { "role": "system", "content": "You are a clinical intake assistant..." },
    { "role": "user", "content": "My throat hurts for 3 days." },
    { "role": "assistant", "content": "How old are you?" },
    { "role": "user", "content": "57, female." },
    { "role": "assistant", "content": "<READY_TO_REPORT>" }
  ],
  "report_json": "{\"patient_id\":\"P001\",\"visit_date\":\"2025-09-14\",\"visit_time\":\"11:15\",\"age\":57,\"sex\":\"F\"}"
}
```

- `messages`: conversation turns.
- `report_json`: gold canonical JSON (stringified, minified).

### JSON Schema (`data/schema.json`)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "VisitSummary",
  "type": "object",
  "properties": {
    "patient_id": { "type": "string" },
    "visit_date": { "type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$" },
    "visit_time": { "type": "string", "pattern": "^\\d{2}:\\d{2}$" },
    "age": { "type": "integer", "minimum": 0, "maximum": 120 },
    "sex": { "type": "string", "enum": ["M", "F", "X"] },
    "complaints": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "symptom": { "type": "string" },
          "details": { "type": "string" },
          "priority": { "type": "integer" }
        },
        "required": ["symptom", "details", "priority"]
      }
    },
    "history_of_present_illness": { "type": "string" },
    "medical_history": { "type": "object" },
    "status_praesens": { "type": "object" },
    "skin_and_mucosa": { "type": "object" },
    "respiratory_system": { "type": "object" },
    "cardiovascular_system": { "type": "object" },
    "digestive_system": { "type": "object" },
    "urinary_system": { "type": "object" },
    "status_localis": { "type": "string" }
  },
  "required": [
    "patient_id",
    "visit_date",
    "visit_time",
    "age",
    "sex",
    "complaints"
  ]
}
```

---

## 4. Source Code Plan

### 📂 `src/`

#### `train.py`

- Fine-tunes the base LLM (e.g., Llama-3.1-Instruct) using **LoRA SFT**.
- Training objective: model learns both _good conversation turns_ and _final JSON style_.

#### `terminal_chat.py`

- Terminal-based chat interface.
- Flow:

  1. Start with chat mode (system prompt).
  2. Collect answers, update a `case_state` dictionary.
  3. If user types `/done` or model says `<READY_TO_REPORT>`, switch to report mode.
  4. Generate final JSON.
  5. Save `{messages, report_json}` in `outputs/sessions/` as JSONL (same format as training data).

#### `report_gen.py`

- Library function to generate a JSON report given a `case_state`.
- Uses **report system prompt** and schema validation.

#### `completeness.py`

- Defines required fields and checks missing values in `case_state`.
- Used to decide which follow-up question to ask.

#### `utils.py`

- JSON helpers (canonicalization, schema load).
- Session saver (append transcript + report to file).

#### `eval.py`

- Evaluates model on validation set.
- Metrics:

  - JSON valid %
  - Schema valid %
  - Exact match %
  - Narrative similarity (e.g., for free-text fields)

---

## 5. Workflow

### Step 1: Collect Data

- Annotate sample conversations → gold reports.
- Save as `train.jsonl` / `valid.jsonl`.

### Step 2: Train

```bash
python src/train.py
```

- Produces LoRA adapter in `outputs/lora/`.

### Step 3: Interactive Chat

```bash
python src/terminal_chat.py
```

- Chat with AI in terminal.
- Type `/done` when finished.
- Outputs a validated JSON report.
- Saves transcript + report in `outputs/sessions/latest.jsonl`.

### Step 4: Evaluation

```bash
python src/eval.py
```

- Reports metrics on validation set.

---

## 6. Deliverables

- **Fine-tuned model adapter**: `outputs/lora/`
- **Terminal UI** for intake simulation: `src/terminal_chat.py`
- **Session logs** (usable as new training data): `outputs/sessions/*.jsonl`
- **Evaluation script**: metrics for quality check

---

✅ With this setup:

- One model handles both conversation & report generation.
- Training/testing formats are consistent.
- Every real conversation can be recycled into training data to improve performance.
