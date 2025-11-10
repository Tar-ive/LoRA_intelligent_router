#!/usr/bin/env python3
"""
convert_peft_to_minlora_v2.py

Download an adapter checkpoint from the Hugging Face Hub and extract only the LoRA
tensors (keys that include 'lora' e.g. lora_A / lora_B). This avoids trying to load
embedding / lm_head weights that can differ in vocab size from your local base model.

Usage:
    python convert_peft_to_minlora_v2.py \
      --adapter_repo brico/simplifyd-gemma-3-1b-it-sft-lora \
      --out lora_adapter_brico.safetensors \
      --hf_token "$HF_TOKEN"

The script will try common filenames in the repo (safetensors or pytorch .bin/.pt).
"""
import argparse
import os
import tempfile
import torch

from huggingface_hub import hf_hub_download, list_repo_files
try:
    from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
    HAVE_SAFETENSORS = True
except Exception:
    HAVE_SAFETENSORS = False

COMMON_CANDIDATES = [
    "pytorch_model.bin",
    "pytorch_model.pt",
    "adapter_model.bin",
    "adapter_model.safetensors",
    "model.safetensors",
    "pytorch_model.safetensors",
    "rust_model.ot",
    "tf_model.h5",
    "pytorch_lora_weights.bin",
    "lora_weights.safetensors",
]

def try_find_checkpoint(repo_id, token=None):
    # list files in repo and pick first matching candidate
    try:
        files = list_repo_files(repo_id, token=token)
    except Exception:
        files = []
    for candidate in COMMON_CANDIDATES:
        if candidate in files:
            return candidate
    # fallback: return the largest .safetensors or .bin present (heuristic)
    for ext in (".safetensors", ".bin", ".pt"):
        matches = [f for f in files if f.endswith(ext)]
        if matches:
            # pick first
            return matches[0]
    return None

def load_checkpoint_file(path):
    if path.endswith(".safetensors"):
        if not HAVE_SAFETENSORS:
            raise RuntimeError("safetensors not available, install safetensors to load this file")
        return load_safetensors(path)
    else:
        # torch.load for .bin/.pt
        return torch.load(path, map_location="cpu")

def is_lora_key(k: str) -> bool:
    lk = k.lower()
    # include common patterns: lora_a, lora_b, lora_ etc.
    return "lora_" in lk or lk.endswith(".lora_a") or lk.endswith(".lora_b") or lk.endswith("lora_a") or lk.endswith("lora_b")

def extract_lora_from_state(sd: dict):
    lora_sd = {}
    for k, v in sd.items():
        if is_lora_key(k):
            # ensure tensor is on CPU
            t = v.cpu().clone().detach() if isinstance(v, torch.Tensor) else v
            lora_sd[k] = t
    return lora_sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_repo", required=True, help="HF repo id for the adapter (e.g. user/adapter-repo)")
    ap.add_argument("--out", required=True, help="Output filename (.safetensors recommended or .pt)")
    ap.add_argument("--hf_token", default=None, help="Hugging Face token if required")
    args = ap.parse_args()

    if args.out.endswith(".safetensors") and not HAVE_SAFETENSORS:
        raise RuntimeError("safetensors not installed; install safetensors or use .pt as output")

    print("Listing files in adapter repo to find a checkpoint...")
    candidate = try_find_checkpoint(args.adapter_repo, token=args.hf_token)
    if candidate is None:
        print("No obvious checkpoint filename found in the repo. You may need to specify the filename manually.")
        print("Falling back to trying common candidates directly.")
    else:
        print("Found candidate checkpoint file in repo:", candidate)

    # try candidates (if we found one, try that first)
    tried = []
    candidates_to_try = [candidate] + COMMON_CANDIDATES if candidate else COMMON_CANDIDATES
    download_path = None
    for fname in candidates_to_try:
        if not fname:
            continue
        if fname in tried:
            continue
        tried.append(fname)
        try:
            print("Attempting to download", fname)
            download_path = hf_hub_download(repo_id=args.adapter_repo, filename=fname, token=args.hf_token)
            print("Downloaded to", download_path)
            break
        except Exception as e:
            # skip if not found or permission error
            print("Could not download", fname, "->", str(e))
            download_path = None

    if download_path is None:
        raise RuntimeError("Could not find/download a checkpoint file from the adapter repo. Check the repo files or provide a direct file name.")

    print("Loading checkpoint file...")
    sd = load_checkpoint_file(download_path)

    # If the checkpoint is a 'state dict' nested inside another dict (common)
    # e.g., {'model': {...}} or {'state_dict': {...}} -> find nested dict with tensors
    if not any(isinstance(v, torch.Tensor) for v in (sd.values() if isinstance(sd, dict) else [])):
        # try common nests
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        else:
            # attempt to find the largest dict value with tensors
            candidates = [v for v in (sd.values() if isinstance(sd, dict) else []) if isinstance(v, dict)]
            best = None
            best_count = 0
            for c in candidates:
                cnt = sum(1 for vv in c.values() if isinstance(vv, torch.Tensor))
                if cnt > best_count:
                    best_count = cnt
                    best = c
            if best is not None:
                sd = best

    print("Total keys in checkpoint:", len(sd) if isinstance(sd, dict) else "N/A")

    print("Extracting LoRA parameters...")
    lora_sd = extract_lora_from_state(sd if isinstance(sd, dict) else {})
    if len(lora_sd) == 0:
        # print helpful diagnostics and exit
        print("ERROR: No 'lora' keys found in checkpoint. Keys present (sample up to 50):")
        sample_keys = list(sd.keys())[:50] if isinstance(sd, dict) else []
        for k in sample_keys:
            print(" ", k)
        raise RuntimeError("No LoRA parameters found. The adapter checkpoint may be a full model or uses a different naming scheme.")

    print(f"Found {len(lora_sd)} LoRA tensors. Saving to {args.out} ...")
    if args.out.endswith(".safetensors"):
        # convert to pure CPU tensors if needed
        st = {k: (v.cpu().clone().detach() if isinstance(v, torch.Tensor) else v) for k, v in lora_sd.items()}
        save_safetensors(st, args.out)
    else:
        torch.save(lora_sd, args.out)

    print("Done. The output file contains only LoRA tensors and should be safe to load with minlora.load_multiple_lora().")

if __name__ == "__main__":
    main()