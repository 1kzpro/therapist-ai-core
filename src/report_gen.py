import json, torch
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from jsonschema import validate
from utils import load_schema, canonical_json

def seed_all(sd=7):
    import random, numpy as np
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def load_model_and_tokenizer(base_model:str, adapter_dir:str=None):
    tok = AutoTokenizer.from_pretrained(adapter_dir or base_model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto"
    )
    if adapter_dir:
        base = PeftModel.from_pretrained(base, adapter_dir)
    return base, tok

def generate(
    model, tok, messages: List[Dict[str,str]], max_new_tokens=1024, greedy=True
) -> str:
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        do_sample=not greedy,
        temperature=0.0 if greedy else 0.7,
        top_p=1.0,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id
    )
    text = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
    return text

def generate_report_from_state(
    model, tok, case_state: Dict[str, Any], schema_path: str = "data/schema.json"
) -> str:
    sys = {"role":"system","content":open("prompts/system_report.txt").read()}
    usr = {"role":"user","content":"Here is case_state:\n" + json.dumps(case_state)}
    raw = generate(model, tok, [sys, usr], max_new_tokens=1024, greedy=True)
    s, e = raw.find("{"), raw.rfind("}")
    obj = json.loads(raw[s:e+1])
    schema = load_schema(schema_path)
    validate(instance=obj, schema=schema)
    return canonical_json(obj)

def generate_report_from_messages(
    model, tok, messages: List[Dict[str,str]], schema_path: str = "data/schema.json"
) -> str:
    # Directly ask the model for the final report given the full conversation (matches training)
    sys = {"role":"system","content":open("prompts/system_report.txt").read()}
    inp = [{"role":m["role"], "content":m["content"]} for m in messages]
    raw = generate(model, tok, [sys] + inp, max_new_tokens=1024, greedy=True)
    s, e = raw.find("{"), raw.rfind("}")
    obj = json.loads(raw[s:e+1])
    schema = load_schema(schema_path)
    validate(instance=obj, schema=schema)
    return canonical_json(obj)
