import os, json
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from jsonschema import validate
from utils import load_schema, canonical_json, save_session
from completeness import missing_fields

# Load from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Default to Qwen2.5-1.5B-Instruct for better performance
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "outputs/lora")  # set to "" to use base only

def load_model_tok():
    print(f"Loading model: {BASE_MODEL}")
    if os.path.isdir(ADAPTER_DIR):
        print(f"Loading adapter from: {ADAPTER_DIR}")
    else:
        print("Using base model only (no adapter)")
    
    tok = AutoTokenizer.from_pretrained(ADAPTER_DIR if os.path.isdir(ADAPTER_DIR) else BASE_MODEL, use_fast=True)
    
    # Use float32 for MPS compatibility
    import torch
    if torch.backends.mps.is_available():
        torch_dtype = torch.float32
        device_map = None
        print("Using MPS (Apple Silicon) with float32")
    else:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None
        print(f"Using {'CUDA' if torch.cuda.is_available() else 'CPU'} with {torch_dtype}")
    
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch_dtype, device_map=device_map
    )
    if os.path.isdir(ADAPTER_DIR):
        base = PeftModel.from_pretrained(base, ADAPTER_DIR)
        print("✅ Adapter loaded successfully")
    return base, tok

def gen(model, tok, messages, max_new=512):
    # Create attention mask to avoid warnings
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    attention_mask = inputs.ne(tok.pad_token_id)
    
    out = model.generate(
        inputs,
        attention_mask=attention_mask,
        do_sample=False, 
        num_beams=1, 
        max_new_tokens=max_new,
        eos_token_id=tok.eos_token_id, 
        pad_token_id=tok.eos_token_id
    )
    txt = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
    return txt

def extract_facts(model, tok, user_text: str, known_keys: List[str]) -> Dict[str, Any]:
    sys = {"role":"system","content":
           "Extract ONLY facts present that map to these keys. Reply JSON with only present keys. Missing keys omitted.\nKeys: " + ", ".join(known_keys)}
    usr = {"role":"user","content": user_text}
    raw = gen(model, tok, [sys, usr], max_new=256)
    try:
        s, e = raw.find("{"), raw.rfind("}")
        return json.loads(raw[s:e+1])
    except Exception:
        return {}

def all_schema_paths(schema: Dict[str, Any], prefix="") -> List[str]:
    # Collect dotted keys for leaf properties (1-level and nested objects)
    props = schema.get("properties", {})
    out = []
    for k, v in props.items():
        if v.get("type") == "object":
            sub = v.get("properties", {})
            for sk in sub.keys():
                out.append(f"{k}.{sk}")
        else:
            out.append(k)
    return out

def main():
    schema = load_schema("data/schema.json")
    keyspace = all_schema_paths(schema)

    model, tok = load_model_tok()

    system_chat = {"role":"system","content":open("prompts/system_chat.txt").read()}
    messages = [system_chat]

    case_state: Dict[str, Any] = {}  # you can pre-initialize with nulls, but sparse dict is fine

    print("Therapist AI Core (terminal)")
    print("Type your replies. Commands: /done (finish)  /show (show collected state)  /quit")
    print("-" * 60)

    # initial assistant greeting (configurable)
    initial_greeting = os.environ.get(
        "INITIAL_GREETING",
        "Hello, I'm your primary care intake assistant. What brings you in today?"
    )
    print("\nAssistant:", initial_greeting)
    messages.append({"role":"assistant","content":initial_greeting})

    while True:
        user = input("\nYou: ").strip()
        if user.lower() in ("/quit","/exit"):
            print("Goodbye.")
            return
        if user.lower() == "/show":
            print("\nCurrent case_state:", json.dumps(case_state, indent=2))
            continue
        if user.lower() == "/done":
            break

        messages.append({"role":"user","content":user})

        # extract facts from this user reply
        facts = extract_facts(model, tok, user, keyspace)
        # shallow merge into case_state; feel free to deep-merge if nested (simple overwrite is fine for demo)
        def merge(dst, src):
            for k,v in src.items():
                if "." in k:
                    # support dotted keys like "skin_and_mucosa.temperature"
                    parts = k.split(".")
                    cur = dst
                    for p in parts[:-1]:
                        cur.setdefault(p, {})
                        cur = cur[p]
                    cur[parts[-1]] = v
                else:
                    dst[k] = v
        merge(case_state, facts)

        # decide next step
        miss = missing_fields(case_state)
        if not miss:
            messages.append({"role":"user","content":"Say <READY_TO_REPORT>."})
        else:
            messages.append({"role":"user","content":f"We still need: {', '.join(miss)}. Ask ONE concise question to collect the most important missing item."})

        # Always include the system prompt plus a short rolling history
        context = [messages[0]] + messages[1:][-6:]
        assistant = gen(model, tok, context, max_new=256)
        print("\nAssistant:", assistant)
        messages.append({"role":"assistant","content":assistant})

        if "<READY_TO_REPORT>" in assistant:
            break

    # REPORT MODE
    sys_report = {"role":"system","content":open("prompts/system_report.txt").read()}
    user_report = {"role":"user","content":"Here is case_state:\n" + json.dumps(case_state)}
    raw = gen(model, tok, [sys_report, user_report], max_new=1024)

    s, e = raw.find("{"), raw.rfind("}")
    report = json.loads(raw[s:e+1])
    validate(instance=report, schema=schema)
    final_report = canonical_json(report)

    print("\n=== FINAL REPORT (JSON) ===")
    print(final_report)

    # Save transcript & report in training format
    save_session(messages, final_report)
    print("\nSaved to outputs/sessions/latest.jsonl")

if __name__ == "__main__":
    main()
