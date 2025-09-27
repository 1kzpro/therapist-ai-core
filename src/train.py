import os, json, argparse, random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

def seed_all(sd=7):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--train_file", type=str, default="data/train.jsonl")
    ap.add_argument("--valid_file", type=str, default="data/valid.jsonl")
    ap.add_argument("--output_dir", type=str, default="outputs/lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    return ap

def format_example(ex):
    # Append gold JSON as final assistant turn
    messages = ex["messages"] + [{"role":"assistant","content": ex["report_json"]}]
    return {"messages": messages}

def main():
    args = build_argparser().parse_args()
    seed_all()

    ds = load_dataset("json", data_files={"train": args.train_file, "validation": args.valid_file})
    ds = ds.map(format_example, remove_columns=ds["train"].column_names)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    def chat_collate(batch):
        # Convert messages → tokens and mask labels for all but final assistant turn
        input_ids, attention_mask, labels = [], [], []
        for ex in batch:
            messages = ex["messages"]
            assert messages[-1]["role"] == "assistant"
            # full conversation with final gold
            full_txt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            enc = tokenizer(full_txt, truncation=True, max_length=args.max_seq_len)
            # prompt without the last assistant (for masking prefix)
            prompt_txt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
            prompt_ids = tokenizer(prompt_txt, truncation=True, max_length=args.max_seq_len)["input_ids"]

            ids = enc["input_ids"]; attn = enc["attention_mask"]
            lab = [-100]*len(ids)
            start = len(prompt_ids)
            for i in range(start, len(ids)):
                lab[i] = ids[i]

            input_ids.append(ids); attention_mask.append(attn); labels.append(lab)

        # pad
        def pad(seq_list, pad_val):
            maxlen = max(len(x) for x in seq_list)
            out = []
            for x in seq_list:
                out.append(x + [pad_val]*(maxlen - len(x)))
            return out

        input_ids = pad(input_ids, tokenizer.eos_token_id)
        attention_mask = pad(attention_mask, 0)
        labels = pad(labels, -100)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=torch.cuda.is_available(),
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=lora_config,
        formatting_func=None,
        max_seq_length=args.max_seq_len,
        data_collator=chat_collate,
        packing=False,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
