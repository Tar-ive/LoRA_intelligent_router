# app.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr

# Configuration - change these to the exact HF IDs you want to use
BASE_MODEL = "google/gemma-3-1b-it"
# Example LoRA adapter id or local path. Replace with the LoRA you pulled from HF:
LORA_ADAPTER = "codelion/gemma-3-1b-it-reasoning-grpo-lora"  # <--- change this

# If the model/adapters require gated access, provide a HF token in env var HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Helper: load base model with a preferred dtype and fallbacks
def load_base_model(model_id):
    # Try FP16 (saves memory). If MPS + fp16 causes issues, we'll fallback.
    # Note: on macOS MPS support is best with PyTorch 2.x; if you experience crashes,
    # try torch_dtype=torch.float32.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_auth_token=HF_TOKEN,
        )
        print("Loaded base model in float16")
        return model
    except Exception as e_fp16:
        print("FP16 load failed, retrying with float32:", e_fp16)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_auth_token=HF_TOKEN,
        )
        print("Loaded base model in float32")
        return model

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_auth_token=HF_TOKEN, use_fast=False)

print("Loading base model...")
base_model = load_base_model(BASE_MODEL)

print("Applying LoRA adapter (PEFT)...")
# Wrap base model with the LoRA adapter. PEFT will load weights and return a PeftModel.
# If LORA_ADAPTER is local path, set the path. If it's on the hub, set the repo id.
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER, torch_dtype=base_model.dtype, use_auth_token=HF_TOKEN)
print("LoRA adapter applied.")

# Move to device
print("Moving model to device:", device)
model.to(device)
model.eval()

# Optionally: merge and unload the adapter if you want a standalone merged model:
# model = model.merge_and_unload()  # uncomment to merge LoRA weights into base (permanent)

# Simple generation helper
def generate_answer(prompt: str, max_new_tokens: int = 128, temperature: float = 0.7):
    # Basic prompt -> tokens -> generate -> decode
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Minimal chat function for Gradio (keeps a short history)
def chat_with_model(user_message, history):
    if history is None:
        history = []
    # Very simple chat formatting: you can adapt to a specific system/user format used by the LoRA
    prompt = ""
    if history:
        for u, r in history:
            prompt += f"User: {u}\nAssistant: {r}\n"
    prompt += f"User: {user_message}\nAssistant:"
    answer = generate_answer(prompt, max_new_tokens=200, temperature=0.6)
    # trim any repeating prompt suffix
    # sometimes model repeats the prompt; remove prefix equal to prompt
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()
    history.append((user_message, answer))
    return history, history

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Local Gemma + LoRA Chat")
    chat = gr.Chatbot()
    state = gr.State([])
    txt = gr.Textbox(
    show_label=False, 
    placeholder="Enter message and press enter", 
    container=False  # Pass the argument here
    )
    txt.submit(chat_with_model, [txt, state], [chat, state])
    demo.launch(server_name="0.0.0.0", share=False)