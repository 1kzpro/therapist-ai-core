from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",  # uses GPU if available
)

messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Say hi in one short sentence."},
]
out = pipe(messages, max_new_tokens=40)
print(out[0]["generated_text"][-1]["content"])
