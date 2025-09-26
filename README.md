# 🩺 Primary Care Physician AI — Documentation

## 1) Project Overview

**Goal**
Primary Care Physician AI (PCP-AI) is an intake assistant that:

1. **Conversation Mode (intake):**

   - Chats with a patient to collect required clinical details (demographics, chief complaints, history, ROS-style sections, vitals if provided, etc.).
   - Asks concise, targeted follow-ups to fill missing “slots” from a predefined schema.

2. **Report Generation Mode:**

   - When intake is complete, produces a **strict, schema-valid JSON visit summary** (compatible with downstream systems).
   - Saves the **full chat transcript + final report** in the same format used for training—so each real session can become future training data.

**One model, two modes**
A single fine-tuned LLM (via **LoRA SFT**) handles both: (a) intake conversation and (b) structured JSON report. Mode is controlled by prompts and the app’s “case state.”

**Clinical positioning**
This is an **intake and documentation helper**, not a diagnostic or treatment engine. The agent **collects** reported information and formats it; it must not diagnose, prescribe, or replace clinician judgment.

---

## 2) Conceptual Flow

1. **Start** the terminal UI (`src/terminal_chat.py`).
2. PCP-AI **asks questions** based on what fields are still missing in the schema (slot-filling).
3. **Patient answers**; the app **extracts facts** into an internal `case_state` (your source of truth).
4. When required fields are complete (or `/done` is used), the model switches to **Report Mode**.
5. PCP-AI outputs a **strict JSON** report that matches `data/schema.json`.
6. The **transcript + final JSON** are saved to `outputs/sessions/latest.jsonl` so you can curate and recycle them as training examples.

---

## 3) Data Schema

The schema (`data/schema.json`) defines the visit-summary shape (e.g., `patient_id`, `visit_date`, `age`, `sex`, `complaints`, `history_of_present_illness`, `medical_history`, system reviews, etc.).

- JSON Schema validation enforces structure, types, and allowed values.
- Required fields ensure minimal completeness for a PCP intake.

> You can keep the schema provided in the repo as a starting point, and expand fields/constraints as your clinical requirements evolve.

---

## 4) Source Code → Business Logic Map

All implementation is under `src/`:

| File                 | What business logic it implements                                                                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **train.py**         | **LoRA SFT** of a base chat LLM (e.g., Llama-3.1). Teaches the model _how to conduct a PCP intake conversation_ and _how to emit the final JSON_. Trains only on the final assistant turn (gold JSON). |
| **terminal_chat.py** | **Intake runtime (Terminal UI).** Orchestrates the live conversation loop, maintains `case_state`, checks completeness, triggers report generation, prints JSON, and saves session in training format. |
| **report_gen.py**    | **Report generation** utilities. Given a `case_state` or a transcript, prompts the same model to produce **schema-valid JSON** (validated via `jsonschema`).                                           |
| **completeness.py**  | **Slot-filling rules.** Lists required fields (by dotted paths) and returns which are still missing to guide the next, focused question.                                                               |
| **utils.py**         | **Infrastructure helpers**: canonical JSON, schema loader, session saver, deep get/set for nested dicts, IDs, etc.                                                                                     |
| **eval.py**          | **Evaluation** on a validation set: JSON validity rate, schema validity rate, exact match rate (canonicalized).                                                                                        |

**Prompts**

- `prompts/system_chat.txt` — Instructions for **PCP intake mode**: collect facts only, one question at a time, no medical advice, say `<READY_TO_REPORT>` when complete.
- `prompts/system_report.txt` — Instructions for **report mode**: output **strict JSON** using only collected facts; unknowns must be null/per schema.

---

## 5) Running the System

### Setup (virtualenv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### (Optional) Fine-tuning

```bash
python src/train.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --train_file data/train.jsonl \
  --valid_file data/valid.jsonl \
  --output_dir outputs/lora \
  --epochs 1
```

- Produces a LoRA adapter in `outputs/lora/`.
- You can also run without fine-tuning (zero-shot), but quality improves with data.

### Interactive Intake (Terminal UI)

```bash
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct ADAPTER_DIR=outputs/lora \
python src/terminal_chat.py
```

In the session:

- Type your answers as the “patient.”
- Commands:

  - `/show` — print current `case_state`
  - `/done` — stop intake and generate the report
  - `/quit` — exit without saving

Outputs:

- A **schema-valid JSON** report printed to terminal
- A **JSONL row** appended to `outputs/sessions/latest.jsonl` with:

  - the full `messages` transcript
  - the `report_json` string
    (same structure as training data)

### Evaluation

```bash
python src/eval.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --adapter_dir outputs/lora \
  --valid_file data/valid.jsonl
```

Reports:

- `json_valid_rate`
- `schema_valid_rate`
- `exact_match_rate` (after canonicalization)

---

## 6) Creating a Training Set

**File format:** JSON Lines (`.jsonl`). One case per line:

```json
{
  "id": "case-0001",
  "messages": [
    {
      "role": "system",
      "content": "You are a primary care intake assistant... (rules: collect facts, one question at a time, no advice)."
    },
    {
      "role": "user",
      "content": "Hi, I'm here for a check-up but I also have a sore throat for 3 days."
    },
    {
      "role": "assistant",
      "content": "Thanks. To confirm, what is your age and sex?"
    },
    { "role": "user", "content": "57, female." },
    {
      "role": "assistant",
      "content": "Do you have a fever or other symptoms, such as cough or rash?"
    },
    {
      "role": "user",
      "content": "Fever yesterday. Productive cough with clear sputum."
    },
    { "role": "assistant", "content": "<READY_TO_REPORT>" }
  ],
  "report_json": "{\"patient_id\":\"P001\",\"visit_date\":\"2025-09-14\",\"visit_time\":\"11:15\",\"age\":57,\"sex\":\"F\",\"complaints\":[{\"symptom\":\"sore throat\",\"details\":\"sore throat reported for 3 days\",\"priority\":1},{\"symptom\":\"cough\",\"details\":\"productive, clear sputum\",\"priority\":2}],\"history_of_present_illness\":\"Gradual onset symptoms with variable intensity.\",\"medical_history\":{\"allergies\":[],\"family_history\":\"none\",\"current_medications\":[],\"past_infections\":[\"chickenpox\"],\"tuberculosis\":\"no\",\"std\":\"yes\",\"hepatitis\":\"no\",\"hiv\":\"positive\",\"infectious_contacts\":\"no\",\"chronic_conditions\":[\"hypertension\",\"asthma\",\"iron deficiency anemia\"],\"hospitalizations\":[\"2020 - bypass surgery\"],\"blood_transfusion\":\"no\"},\"status_praesens\":{\"general_condition\":\"satisfactory\",\"consciousness\":\"confused\",\"speech\":\"slurred\",\"hearing\":\"normal\",\"vision\":\"impaired\",\"activity\":\"limited\",\"constitution\":\"normal\",\"emotional_status\":\"anxious\",\"sleep\":\"normal\",\"appetite\":\"increased\"},\"skin_and_mucosa\":{\"temperature\":38.5,\"mucosa\":\"dry\",\"pharynx\":\"normal\",\"skin\":\"rash\",\"lymph_nodes\":\"enlarged, tender\"},\"respiratory_system\":{\"breathing_type\":\"mixed\",\"respiratory_rate\":16,\"auxiliary_muscles\":false,\"cough\":\"productive\",\"sputum\":\"clear\",\"auscultation\":\"vesicular breath sounds\"},\"cardiovascular_system\":{\"blood_pressure\":\"118/63\",\"pulse\":\"94 regular\",\"heart_rate\":89,\"edema\":\"none\",\"heart_sounds\":\"normal\",\"heart_murmurs\":\"present\"},\"digestive_system\":{\"tongue\":\"dry\",\"symptoms\":[\"constipation\",\"heartburn\"],\"abdomen\":\"soft\",\"liver\":\"enlarged\",\"stool\":\"constipation\"},\"urinary_system\":{\"urination\":\"normal\",\"incontinence\":\"no\",\"percussion\":\"non-tender\"},\"status_localis\":\"No focal pathological findings.\"}"
}
```

### Authoring Guidelines

- **Messages**

  - Start with a **system** instruction that defines PCP intake behavior.
  - Simulate realistic back-and-forth: assistant asks **targeted** questions, user provides **specific** answers.
  - End the intake with `<READY_TO_REPORT>` to indicate the assistant judged completeness.

- **Report JSON**

  - Must validate against `data/schema.json`.
  - Use **canonical** formatting when stored as a string:

    - `json.dumps(obj, sort_keys=True, separators=(",",":"))`

- **Splits**

  - Put most cases in `data/train.jsonl`; hold out a representative set in `data/valid.jsonl` (cover edge cases, missing data, typos, contradictions, time expressions).

### Bootstrapping your dataset

- Run real/simulated sessions via `src/terminal_chat.py`.
- Curate the appended rows from `outputs/sessions/latest.jsonl` (fix any mistakes) and move them into `data/train.jsonl`.
- Keep iterating: the more high-quality examples you add, the better the model’s intake and report behavior.

---

## 7) Reliability & Safety Notes

- **No diagnosis/treatment.** The agent collects information and formats it; defer medical decisions to clinicians.
- **No invention of facts.** If information is not present, fields should remain `null` or `"unknown"` per schema.
- **Schema validation.** The final JSON is validated with `jsonschema`. Consider adding **constrained decoding** later to guarantee validity at generation time.
- **Determinism.** Greedy decoding (`do_sample=False`) and fixed seeds increase stability on the same stack.
- **Privacy/PHI.** De-identify data used for training/evaluation; store only minimal logs; follow your org’s compliance rules.

---

## 8) Summary

- **PCP-AI** is a single-model, two-mode system:

  - Intake conversation → **slot-filling** guided by `completeness.py` and `prompts/system_chat.txt`.
  - Finalization → **schema-valid JSON** guided by `prompts/system_report.txt` and `data/schema.json`.

- **Terminal UI** lets you test the full loop and automatically produce new training rows.
- **LoRA SFT** provides a practical path to adapt a strong base LLM to your intake + reporting style.
- Over time, your system improves via a **data flywheel**: every good session becomes new training data.

---

If you want, I can also tweak the schema to include more PCP-specific sections (e.g., immunizations, social history, lifestyle factors, screening checklists) and add examples for those fields.
