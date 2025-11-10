#!/usr/bin/env python3
"""
chat_minlora.py

Load a base model, add minLoRA parametrization, preload multiple LoRA adapters
(saved as safetensors or torch .pt files), and run a small Gradio chat UI
that lets you switch adapters instantly.

Usage:
    python chat_minlora.py \
      --base_model google/gemma-3-1b-it \
      --lora_files lora_a.safetensors lora_b.safetensors \
      --hf_token $HF_TOKEN

Notes:
- Ensure each adapter was produced using convert_peft_to_minlora.py or otherwise contains only the lora_* keys expected by minLoRA.
- On M3 macbook: code will use MPS if available, fallback to CPU.
"""
"""
chat_minlora.py (patched to avoid dtype mismatch when adding minLoRA)

Key changes:
- Temporarily set torch default dtype to the base model dtype before calling minlora.add_lora(model)
  so the LoRAParametrization parameters are created with the same dtype as the model.
- When loading each LoRA state dict, cast tensor values to the model dtype before passing to minlora.load_multiple_lora().
- Restore default dtype to torch.float32 after parametrizations are added.
"""
"""
chat_minlora.py (patched to avoid device/dtype mismatch for minLoRA multi-adapter)

Usage example:
python chat_minlora.py \
  --base_model google/gemma-3-1b-it \
  --lora_files lora_adapter_1.safetensors lora_adapter_2.safetensors lora_brico.safetensors \
  --hf_token "$HF_TOKEN"
"""
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import minlora
import torch.nn as nn

try:
    from safetensors.torch import load_file as load_safetensors_file
    HAVE_SAFETENSORS = True
except Exception:
    HAVE_SAFETENSORS = False

def load_lora_state(path):
    if path.endswith(".safetensors"):
        if not HAVE_SAFETENSORS:
            raise RuntimeError("safetensors not installed; install safetensors to load .safetensors files.")
        return load_safetensors_file(path)
    else:
        return torch.load(path, map_location="cpu")

def move_minlora_unregistered_tensors_to_device(model, device, dtype):
    """
    Move minLoRA's unregistered adapter tensors (lora_As, lora_Bs, dropout mask)
    to the correct device and dtype. This is necessary because minlora.load_multiple_lora
    stores adapter tensors in Python lists that are not moved by model.to(device).
    """
    for module in model.modules():
        if isinstance(module, minlora.LoRAParametrization):
            # registered params/buffers should already be moved by model.to(device)
            # but ensure they are correct dtype/device
            if hasattr(module, "lora_A") and isinstance(module.lora_A, torch.nn.Parameter):
                module.lora_A.data = module.lora_A.data.to(device=device, dtype=dtype)
            if hasattr(module, "lora_B") and isinstance(module.lora_B, torch.nn.Parameter):
                module.lora_B.data = module.lora_B.data.to(device=device, dtype=dtype)
            if hasattr(module, "lora_dropout_mask"):
                module.lora_dropout_mask = module.lora_dropout_mask.to(device=device, dtype=dtype)

            # move any appended adapter lists (these are NOT moved by model.to)
            if hasattr(module, "lora_As") and isinstance(module.lora_As, (list, tuple)):
                new_As = []
                for t in module.lora_As:
                    if isinstance(t, torch.Tensor):
                        new_As.append(nn.Parameter(t.to(device=device, dtype=dtype)))
                    else:
                        new_As.append(t)
                module.lora_As = new_As

            if hasattr(module, "lora_Bs") and isinstance(module.lora_Bs, (list, tuple)):
                new_Bs = []
                for t in module.lora_Bs:
                    if isinstance(t, torch.Tensor):
                        new_Bs.append(nn.Parameter(t.to(device=device, dtype=dtype)))
                    else:
                        new_Bs.append(t)
                module.lora_Bs = new_Bs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_files", nargs="+", required=True, help="List of LoRA state files (.safetensors or .pt)")
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, use_auth_token=args.hf_token)

    print("Loading base model (FP16 preferred)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu",
            use_auth_token=args.hf_token,
        )
        print("Loaded base in float16 on CPU and will move to device later.")
    except Exception as e:
        print("FP16 load failed, falling back to float32:", e)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu",
            use_auth_token=args.hf_token,
        )

    # Ensure minLoRA creates params with same dtype as model weights
    prev_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(model.dtype)
        print("Temporarily set default dtype to", model.dtype, "before add_lora()")
        minlora.add_lora(model)
    finally:
        torch.set_default_dtype(prev_default)
        print("Restored default dtype to", prev_default)

    print("Loading LoRA adapter files into memory...")
    lora_state_dicts = []
    for p in args.lora_files:
        print("  loading", p)
        sd = load_lora_state(p)
        # Cast all tensor values to the base model dtype (keep them on CPU for now)
        if isinstance(sd, dict):
            casted = {}
            for k, v in sd.items():
                if isinstance(v, torch.Tensor):
                    casted[k] = v.to(dtype=model.dtype, device="cpu")
                else:
                    casted[k] = v
            sd = casted
        lora_state_dicts.append(sd)

    print("Preparing model for multiple LoRA adapters...")
    minlora.load_multiple_lora(model, lora_state_dicts)

    # Move base model (registered params) to device first
    print("Moving base model to device:", device)
    model.to(device)

    # Now move minLoRA's unregistered adapter tensors to device/dtype
    print("Moving unregistered LoRA tensors (lora_As/lora_Bs) to device and dtype...")
    move_minlora_unregistered_tensors_to_device(model, device=device, dtype=model.dtype)

    # select first adapter by default (or disable by passing None)
    if len(lora_state_dicts) > 0:
        minlora.select_lora(model, 0)
        active_idx = 0
    else:
        active_idx = None

    model.eval()

    def generate_answer(prompt: str, max_new_tokens=args.max_new_tokens, temperature: float = 0.7):
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

    def chat_with_switch(user_msg, history, adapter_choice):
        nonlocal active_idx
        if history is None:
            history = []

        # adapter_choice is an index string from the UI; map to integer
        if adapter_choice is None or adapter_choice == "(none)":
            sel_idx = -1
        else:
            sel_idx = int(adapter_choice.split("_")[-1])

        if sel_idx == -1:
            minlora.disable_lora(model)
            active_idx = None
        else:
            if sel_idx != active_idx:
                minlora.select_lora(model, sel_idx)
                active_idx = sel_idx

        # build prompt from full history
        prompt = ""
        for u, r in history:
            prompt += f"User: {u}\nAssistant: {r}\n"
        prompt += f"User: {user_msg}\nAssistant:"

        answer = generate_answer(prompt)
        if answer.startswith(prompt):
            answer = answer[len(prompt):].strip()
        history.append((user_msg, answer))
        return history, history

    choices = ["(none)"] + [f"adapter_{i}" for i in range(len(lora_state_dicts))]

    with gr.Blocks() as demo:
        gr.Markdown("# Local Gemma + minLoRA — multi-adapter switching")
        with gr.Row():
            adapter_dropdown = gr.Dropdown(choices=choices, value=choices[1] if len(choices) > 0 else "(none)", label="Active LoRA Adapter")
        chatbox = gr.Chatbot()
        state = gr.State([])
        txt = gr.Textbox(show_label=False, placeholder = "Enter Message and Press Enter", container=False)
        txt.submit(chat_with_switch, [txt, state, adapter_dropdown], [chatbox, state])
        demo.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    main()