import json, argparse, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from jsonschema import validate
from utils import canonical_json, load_schema
from report_gen import generate_report_from_messages

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--adapter_dir", type=str, default="outputs/lora")
    ap.add_argument("--valid_file", type=str, default="data/valid.jsonl")
    return ap

def main():
    args = build_argparser().parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter_dir if os.path.isdir(args.adapter_dir) else args.base_model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", device_map="auto")
    if os.path.isdir(args.adapter_dir):
        base = PeftModel.from_pretrained(base, args.adapter_dir)

    ds = load_dataset("json", data_files={"validation": args.valid_file})["validation"]
    schema = load_schema("data/schema.json")

    n = 0
    ok_json = 0
    ok_schema = 0
    exact = 0

    for ex in ds:
        n += 1
        messages = ex["messages"]
        gold = canonical_json(json.loads(ex["report_json"]))
        try:
            pred = generate_report_from_messages(base, tok, messages, schema_path="data/schema.json")
            ok_json += 1
            validate(instance=json.loads(pred), schema=schema)
            ok_schema += 1
            if pred == gold:
                exact += 1
        except Exception as e:
            # print(f"Error on example {ex.get('id','?')}: {e}")
            pass

    print({
        "n": n,
        "json_valid_rate": ok_json / max(1,n),
        "schema_valid_rate": ok_schema / max(1,n),
        "exact_match_rate": exact / max(1,n)
    })

if __name__ == "__main__":
    main()
