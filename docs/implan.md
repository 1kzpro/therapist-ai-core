Awesome—here’s a tight, practical implementation plan for a **single-model** system that (1) chats with the patient, (2) fills a structured “case state”, and (3) emits a **schema-valid JSON report** at the end. You’ll also get a **terminal UI** for the whole flow, and the final saved sample will match your **training data format** for consistency.

---

# Project layout

```
visit-agent/
├─ env.yml
├─ data/
│  ├─ train.jsonl
│  ├─ valid.jsonl
│  └─ schema.json
├─ prompts/
│  ├─ system_chat.txt
│  └─ system_report.txt
├─ src/
│  ├─ train.py
│  ├─ terminal_chat.py
│  ├─ report_gen.py
│  ├─ eval.py
│  ├─ utils.py
│  └─ completeness.py
└─ outputs/
   ├─ lora/                   # saved LoRA adapter
   └─ sessions/               # transcripts + reports captured from terminal runs
```

---

# Data formats

## 1) Training & validation (JSONL)

Each line is one case:

```json
{
  "id": "ex-001",
  "messages": [
    {"role":"system","content":"You are a clinical scribe... (rules)"},
    {"role":"user","content":"Hi, my throat hurts for 3 days..."},
    {"role":"assistant","content":"Thanks. To confirm: what's your age and sex?"},
    {"role":"user","content":"57, female..."},
    ...
    {"role":"assistant","content":"<END_OF_CHAT>"}   // optional marker in training
  ],
  "report_json": "{\"patient_id\":\"P001\",\"visit_date\":\"2025-09-14\",...}"
}
```

- **messages**: complete multi-turn conversation that demonstrates good _slot-filling_.
- **report_json**: the canonical gold JSON (single-line string, minified).
- For fine-tuning we teach the model to (a) ask targeted questions, (b) produce the final JSON.

## 2) Schema (`data/schema.json`)

A JSON Schema matching your fields (dates, enums, number ranges). You’ll validate predictions against this.

---

# Prompts (two “modes” for the same model)

**prompts/system_chat.txt** (collection mode, excerpt)

```
You are a clinical intake assistant.
Goal: collect ONLY facts needed to fill the case schema. If unknown, record null. Never invent facts.
Ask one concise question at a time. Avoid medical advice; you are collecting information.
When you have enough info, say: <READY_TO_REPORT>.
```

**prompts/system_report.txt** (report mode, excerpt)

```
You are a structured reporter. Using ONLY the provided case_state facts (no new inferences), output STRICT JSON that matches the schema exactly. Omit extra keys. Unknowns must be null. JSON only, no commentary.
```

---

# Implementation plan

## A) Training (LoRA SFT on one model)

- Fine-tune a base chat model (e.g., Llama-3.1-8B-Instruct) with **one dataset** (messages → report_json).
- In training, teach both behaviors: targeted follow-ups _and_ final JSON style.

**src/train.py (core ideas)**

- Load `train.jsonl` / `valid.jsonl` with `datasets`.
- Transform each example into a chat where the final assistant turn is `report_json` (supervised target).
- Use **LoRA** (PEFT) to keep compute small.
- Train only on the **final assistant turn** (mask the prompt/earlier turns’ labels).

(You already have a near-complete script from earlier; reuse that.)

## B) Terminal UI for conversation

- One process, one model.

- Loop:

  1. Show assistant question → user types answer
  2. Extract facts from the assistant turn via a lightweight “tool-call” convention or by re-parsing the _user_ reply using the same model in “extract” style
  3. Update `case_state` (your in-app dict aligned to schema)
  4. Run completeness rules → if missing fields remain, ask follow-ups; if done, the assistant prints `<READY_TO_REPORT>`

- User can type `/done` to skip to report generation.

**src/completeness.py (example)**

```python
REQUIRED_FIELDS = [
  "patient_id", "visit_date", "visit_time", "age", "sex",
  "complaints", "status_praesens.general_condition", "skin_and_mucosa.temperature"
  # ... tailor to your schema
]

def missing_fields(case_state: dict) -> list[str]:
    # Check required leaves; return dotted paths that are null/empty
    # Implement simple getters for dotted paths
    ...
```

**src/utils.py (snippets)**

```python
import json

def canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",",":"))

def load_schema(path="data/schema.json"):
    with open(path) as f: return json.load(f)

def save_session(messages, report_json, out_dir="outputs/sessions"):
    # Save as training-format JSONL (one line)
    row = {
      "id": f"session-{...}",  # timestamp or uuid
      "messages": messages,
      "report_json": canonical_json(json.loads(report_json))
    }
    with open(f"{out_dir}/latest.jsonl","a") as f:
        f.write(canonical_json(row) + "\n")
```

**src/terminal_chat.py (simplified)**

```python
import json, readline
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from jsonschema import validate
from utils import canonical_json, save_session, load_schema
from completeness import missing_fields

BASE = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = "outputs/lora"      # after training
SCHEMA = load_schema()

def gen(model, tok, messages, max_new=512):
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=max_new,
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
    return text

def extract_facts(model, tok, case_state, user_text):
    # Small prompt that asks the same model to extract slots from user's last reply.
    sys = {"role":"system","content":
           "Extract ONLY facts present that map to the schema keys. Reply as JSON with keys and values; unknowns omitted."}
    usr = {"role":"user","content": user_text}
    raw = gen(model, tok, [sys, usr], max_new=256)
    try:
        obj = json.loads(raw)
        case_state.update(obj)  # shallow merge; you can implement deep-merge per section
    except:
        pass

def main():
    tok = AutoTokenizer.from_pretrained(ADAPTER, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base, ADAPTER)

    # Conversation starts
    case_state = {}   # initialize all schema leaves to null in your app
    messages = [{"role":"system","content":open("prompts/system_chat.txt").read()}]

    print("Type '/done' when finished. Type '/show' to view collected state.")
    while True:
        # Ask the model for the next question, conditioned on missing fields
        mf = missing_fields(case_state)
        prompt_for_assistant = (
          f"We still need these fields: {', '.join(mf)}.\n"
          f"Ask ONE concise question to collect the most important missing item."
          if mf else "Say <READY_TO_REPORT>."
        )
        messages.append({"role":"user","content":prompt_for_assistant})
        assistant = gen(model, tok, messages[-2:])   # last system content is persistent, supply last U turn for brevity
        print(f"\nAssistant: {assistant}")
        messages.append({"role":"assistant","content":assistant})

        if "<READY_TO_REPORT>" in assistant:
            break

        user = input("\nYou: ").strip()
        if user == "/done": break
        if user == "/show":
            print("\nCurrent case_state:", json.dumps(case_state, indent=2))
            continue

        # Record conversation
        messages.append({"role":"user","content":user})

        # Extract facts from user's reply and update case_state
        extract_facts(model, tok, case_state, user)

    # REPORT MODE (same model, different system prompt)
    sys_report = {"role":"system","content":open("prompts/system_report.txt").read()}
    # Provide the case_state verbatim
    user_report = {"role":"user","content":"Here is case_state:\n" + json.dumps(case_state)}
    report_raw = gen(model, tok, [sys_report, user_report], max_new=1024)

    # Basic repair + schema validation
    s = report_raw.find("{"); e = report_raw.rfind("}")
    report = json.loads(report_raw[s:e+1])
    validate(instance=report, schema=SCHEMA)
    final_report = canonical_json(report)

    print("\n=== FINAL REPORT (JSON) ===")
    print(final_report)

    # Save transcript + report in training format for future fine-tuning
    save_session(messages, final_report)

if __name__ == "__main__":
    main()
```

> This terminal app keeps **one model** and switches behavior via **prompts** + a tiny **extract_facts** helper. You can strengthen extraction later with a strict tool/spec or constrained decoding.

## C) Report generation as a library call

If you want to call “make report” from other code (batch jobs, API), separate it into `report_gen.py`:

**src/report_gen.py (essentials)**

```python
def generate_report(model, tok, case_state, schema_path="data/schema.json"):
    sys = {"role":"system","content":open("prompts/system_report.txt").read()}
    usr = {"role":"user","content":"Here is case_state:\n" + json.dumps(case_state)}
    raw = gen(model, tok, [sys, usr], max_new=1024)
    s, e = raw.find("{"), raw.rfind("}")
    obj = json.loads(raw[s:e+1])
    validate(obj, load_schema(schema_path))
    return canonical_json(obj)
```

---

# Evaluation (quick)

**src/eval.py**: iterate over `valid.jsonl`, ask the model to produce the JSON report from the **messages**, compare with gold:

- JSON validity
- Schema validity
- Exact match (canonical)
- Optional semantic similarity for narrative fields

(You can reuse the earlier `eval.py` sketch.)

---

# How you’ll run it

```bash
# 1) Create env
conda env create -f env.yml
conda activate visit-agent

# 2) (Optional) Fine-tune from your data
python src/train.py

# 3) Run terminal chat (uses your LoRA)
python src/terminal_chat.py
# ...chat...
# type /done to finalize
# It prints the JSON and appends a JSONL row to outputs/sessions/latest.jsonl
```

---

# Notes on reliability

- **Determinism**: use `do_sample=False`, fixed seeds, and deterministic kernels if needed.
- **Schema enforcement**: you can add constrained decoding (e.g., `lm-format-enforcer`) in `report_gen` to virtually eliminate invalid JSON.
- **State management**: the `case_state` dict is your source of truth; the chat model’s job is to **fill slots**, not to diagnose.
- **Data flywheel**: every terminal session is saved in **training format**; you can curate and add back into `data/train.jsonl` to continually improve.
