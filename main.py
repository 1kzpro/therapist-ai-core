import os, json, random, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, TrainingArguments)
from trl import SFTTrainer
from peft import LoraConfig

SEED = 7
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "outputs/llama31-visit-lora"
MAX_SEQ_LEN = 4096   # adjust to your GPU
BATCH_SIZE_PER_DEVICE = 1
GRAD_ACCUM = 8

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def format_example(ex):
    """
    Convert messages + target report_json into a single supervised example.
    We create a chat where the final assistant turn is the gold JSON.
    We mask labels for the prompt turns; only learn on the final assistant reply.
    """
    messages = ex["messages"]
    # Append the training signal as assistant message
    messages_plus = messages + [{"role":"assistant", "content": ex["report_json"]}]
    return {"messages": messages_plus}

def chat_formatting_func(batch):
    """
    Use chat template to produce input_ids and labels correctly:
    - labels = -100 for all tokens except the final assistant turn (so we train only on the output)
    """
    new_batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for messages in batch["messages"]:
        # Find index of the final assistant turn (last one)
        last_ass_idx = len(messages) - 1
        assert messages[last_ass_idx]["role"] == "assistant"

        text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = enc["input_ids"]; attn = enc["attention_mask"]

        # Build labels: mask everything except the final assistant span
        # Strategy: re-render prompt-without-last-assistant, get its token length, mask that prefix.
        prompt_only = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
        prompt_ids = tokenizer(prompt_only, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
        labels = [-100]*len(input_ids)
        start = len(prompt_ids)
        for i in range(start, len(input_ids)):
            labels[i] = input_ids[i]

        new_batch["input_ids"].append(input_ids)
        new_batch["attention_mask"].append(attn)
        new_batch["labels"].append(labels)

    return new_batch

# Load & map
ds = load_dataset("json", data_files={"train":"data/train.jsonl", "validation":"data/valid.jsonl"})
ds = ds.map(format_example, remove_columns=ds["train"].column_names)
# Dynamic tokenization via formatting_func inside Trainer (faster for chat)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    learning_rate=2e-4,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    peft_config=lora_config,
    formatting_func=None,                     # we will use packing=False + custom collator
    max_seq_length=MAX_SEQ_LEN,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    packing=False,
    preprocess_logits_for_metrics=None
)

# Override trainer's collate to our chat_formatting
trainer.tokenizer = tokenizer
trainer._prepare_input = None
trainer.train_dataset = ds["train"]
trainer.eval_dataset = ds["validation"]
trainer.data_collator = lambda features: chat_formatting_func({"messages":[f["messages"] for f in features]})

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
