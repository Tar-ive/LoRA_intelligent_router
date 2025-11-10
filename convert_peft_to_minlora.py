#!/usr/bin/env python3
"""
convert_peft_to_minlora.py

Use this to convert a Hugging Face PEFT adapter (repo or local dir saved with PeftModel.save_pretrained)
into a minLoRA-friendly state dict containing only lora_A / lora_B parameters.

Usage:
    python convert_peft_to_minlora.py \
      --base_model google/gemma-3-1b-it \
      --peft_adapter your-username/gemma-lora-adapter \
      --out_file lora_adapter_1.safetensors \
      --hf_token $HF_TOKEN

Notes:
- This script loads the base model (on CPU) and wraps it with the PEFT adapter (no GPU required).
- The output will be either .safetensors (recommended) or .pt (torch.save).
"""
import os
import argparse
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
try:
    from safetensors.torch import save_file as save_safetensors
    HAVE_SAFETENSORS = True
except Exception:
    HAVE_SAFETENSORS = False

def is_lora_key(k: str) -> bool:
    # minLoRA expects keys that end with lora_A or lora_B (or contain those)
    return k.endswith("lora_A") or k.endswith("lora_B") or ".lora_A" in k or ".lora_B" in k

def load_base_model_cpu(base_model_id, hf_token=None):
    # load on CPU to be memory-sparing; try float16 first then float32 fallback
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu",
            use_auth_token=hf_token,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu",
            use_auth_token=hf_token,
        )
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="HF base model id (same base used for the adapter)")
    ap.add_argument("--peft_adapter", required=True, help="HF adapter repo id or local path with adapter saved by PeftModel.save_pretrained")
    ap.add_argument("--out_file", required=True, help="Output filename (.safetensors recommended or .pt)")
    ap.add_argument("--hf_token", default=None, help="Hugging Face token (if required)")
    args = ap.parse_args()

    if args.out_file.endswith(".safetensors") and not HAVE_SAFETENSORS:
        raise RuntimeError("safetensors not available. Install safetensors or write to .pt")

    print("Loading base model (CPU, low memory usage)...")
    base = load_base_model_cpu(args.base_model, hf_token=args.hf_token)

    print("Applying PEFT adapter from:", args.peft_adapter)
    peft_model = PeftModel.from_pretrained(base, args.peft_adapter, torch_dtype=base.dtype, use_auth_token=args.hf_token)
    # peft_model now includes adapter weights in state_dict

    print("Extracting LoRA parameters (lora_A / lora_B)...")
    sd = peft_model.state_dict()
    lora_sd = {}
    for k, v in sd.items():
        if is_lora_key(k):
            lora_sd[k] = v.cpu().clone().detach()

    if len(lora_sd) == 0:
        raise RuntimeError("No LoRA keys found in state_dict. Check adapter compatibility or path.")

    print(f"Saving {len(lora_sd)} LoRA tensors to {args.out_file} ...")
    if args.out_file.endswith(".safetensors"):
        # safetensors expects a mapping of name->tensor (CPU)
        save_safetensors(lora_sd, args.out_file)
    else:
        torch.save(lora_sd, args.out_file)

    print("Done. You can now use this file with minlora.load_multiple_lora().")

if __name__ == "__main__":
    main()