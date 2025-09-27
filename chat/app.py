# chat/app.py
# Chat UI using an open model with messages + Gradio (new messages format)

from transformers import pipeline
import torch, gradio as gr

# Build a text-generation pipeline that accepts chat messages
pipe = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium",
    dtype=torch.float16,          # on Apple MPS use float16; try bfloat16 if it works for you
    device_map="auto",
)

SYSTEM = (
    "You are a friendly medical triage assistant (NOT a doctor). "
    "Answer clearly and briefly. If safety risk (e.g., chest pain, stroke signs, severe allergy), "
    "advise ER/urgent care. If unsure, say you’re unsure. Keep it practical."
)

def respond(messages):
    # messages is a list of dicts: [{"role":"user"/"assistant"/"system","content":"..."}]
    # Ensure first message is our system prompt
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": SYSTEM}] + messages

    out = pipe(messages, max_new_tokens=220)
    # transformers chat pipeline returns list; last item holds the assistant turn
    reply = out[0]["generated_text"][-1]["content"]
    return reply

with gr.Blocks(title="Therapist-AI Chat (Llama 3.1)") as demo:
    gr.Markdown("### Therapist-AI Chat (Llama 3.1 8B)")
    chat = gr.Chatbot(type="messages", height=420)  # <— new messages format
    txt = gr.Textbox(placeholder="Describe your symptoms or ask a question...")

    def on_submit(user_msg, history):
        history = history + [{"role":"user","content":user_msg}]
        bot_msg = respond(history)
        history = history + [{"role":"assistant","content":bot_msg}]
        return history, ""

    txt.submit(on_submit, [txt, chat], [chat, txt])
    gr.Button("Clear").click(lambda: [], None, chat)

if __name__ == "__main__":
    demo.launch()
