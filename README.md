# LoRA_intelligent_router
A router that auto selects which LoRA adapters to use per given query using intent classification 
# Local Gemma + minLoRA — Multi‑Adapter Chat (M3 MacBook Pro)

Summary
-------
This project lets you run a Gemma family base model locally on an M3 MacBook Pro, load multiple LoRA adapters (converted from Hugging Face PEFT adapters), and switch adapters instantly during a chat session via a small Gradio UI.

What we did (story)
-------------------
- Chose the base model: `google/gemma-3-1b-it`.
- Faced a mismatch when converting some HF LoRA adapters that included full model tensors (embedding / lm_head) with a slightly different vocab size. We solved that by writing `convert_peft_to_minlora_v2.py`, a robust converter that downloads adapter checkpoint files from the hub and extracts only LoRA tensors (keys containing "lora_*") into a `.safetensors` or `.pt` file.
- Built a chat app `chat_minlora.py` that:
  - loads the base Gemma model (preferring FP16 to save memory),
  - adds minLoRA parametrizations,
  - preloads multiple LoRA state dicts,
  - ensures dtype alignment so parametrizations do not fail,
  - moves unregistered per‑adapter tensors to the target device (MPS) so runtime operations succeed,
  - exposes a Gradio chat UI with a dropdown to switch adapters instantly (the full history is injected into the prompt to preserve conversation coherence across switches).
- Iterated on and fixed three runtime issues:
  1. Adapter contained extra base weights → extract only LoRA tensors (v2 converter).
  2. Dtype mismatch when register parametrization (model FP16 vs LoRA params default FP32) → temporarily set PyTorch default dtype to model dtype before calling `minlora.add_lora(...)` and cast LoRA tensors to model dtype.
  3. Device mismatch (some LoRA tensors left on CPU) → explicitly move unregistered LoRA tensors (`lora_As`/`lora_Bs` and `lora_dropout_mask`) to device/dtype after `model.to(device)`.

Files
-----
- `convert_peft_to_minlora_v2.py` — robust converter: download adapter checkpoint from HF and save only LoRA tensors (recommended output: `.safetensors`).
- `chat_minlora.py` — main chat application: loads base model, adds minLoRA param, preloads adapters, moves tensors to device/dtype, runs Gradio UI (adapter switching).
- `requirements.txt` — Python dependencies (transformers, peft, minlora from Git, gradio, safetensors, ...).
- This README — usage summary and commands.

Prerequisites
-------------
- macOS with Apple Silicon (M3) and sufficient RAM (you have 18 GB).
- Python 3.8+ (the examples used a venv).
- A Hugging Face account with consent accepted for Gemma models. Export your HF token if the model or adapter requires it:
  - export HF_TOKEN="hf_xxx"

Important: install PyTorch for Apple Silicon (MPS) using the official instructions at https://pytorch.org/get-started/locally/ — select macOS / pip / Apple silicon (MPS) to get the correct wheel. Do not rely on the generic `pip install torch` from PyPI unless it explicitly matches MPS.

Setup (commands)
----------------

1) Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

2) Install PyTorch (MPS) using the command shown on https://pytorch.org (choose macOS / pip / Apple silicon).
Example (please use the exact command provided by pytorch.org for your macOS/Python):
```bash
# Example - DO NOT copy blindly, use the command from pytorch.org for Apple silicon:
pip install torch torchvision torchaudio
```

Verify MPS availability:
```bash
python -c "import torch; print(torch.__version__); print('mps built:', torch.backends.mps.is_built()); print('mps available:', torch.backends.mps.is_available())"
```

3) Install the Python requirements (after installing the correct torch wheel):
```bash
pip install -r requirements.txt
```
If you want manual installs:
```bash
pip install transformers peft gradio safetensors sentencepiece huggingface-hub
pip install git+https://github.com/changjonathanc/minLoRA.git
```

Convert Hugging Face LoRA adapters to minLoRA state files
-------------------------------------------------------
When an adapter repo includes extra model tensors or was saved with extra vocabulary tokens, loading it into the base wrapper can fail. Use the `convert_peft_to_minlora_v2.py` script to extract only the LoRA tensors.

Example:
```bash
export HF_TOKEN="hf_..."   # if adapter or model is gated
python convert_peft_to_minlora_v2.py \
  --adapter_repo codelion/gemma-3-1b-it-reasoning-grpo-lora \
  --out lora_adapter_1.safetensors \
  --hf_token "$HF_TOKEN"

python convert_peft_to_minlora_v2.py \
  --adapter_repo brico/simplifyd-gemma-3-1b-it-sft-lora \
  --out lora_brico.safetensors \
  --hf_token "$HF_TOKEN"
```

Notes:
- The script tries common checkpoint filenames in the adapter repo, downloads a file, loads it and extracts keys containing "lora_" (case-insensitive), then saves a `.safetensors` (preferred) or `.pt` file with only the adapter tensors.
- If a repo doesn't contain an adapter checkpoint, inspect the repo files on Hugging Face to find the correct file to download.

Run the chat app (multi-adapter switching)
-----------------------------------------
Run the chat UI and preload multiple adapters (the script will move tensors to MPS and align dtypes):

```bash
export HF_TOKEN="hf_..."
python chat_minlora.py \
  --base_model google/gemma-3-1b-it \
  --lora_files lora_adapter_1.safetensors lora_adapter_2.safetensors lora_brico.safetensors \
  --hf_token "$HF_TOKEN"
```

Or (single-line style):
```bash
python chat_minlora.py --base_model google/gemma-3-1b-it --lora_files lora_adapter_1.safetensors lora_adapter_2.safetensors lora_brico.safetensors --hf_token "$HF_TOKEN"
```

The Gradio UI will show at `http://0.0.0.0:7860` by default. Use the dropdown to switch adapters mid-conversation. The chat rebuilds the full prompt from history on each generation to preserve context across adapter switches.

Troubleshooting & common errors
-------------------------------

1) "size mismatch for embed_tokens.weight" (adapter vs base vocab)
- Cause: adapter checkpoint included embeddings/LM head for a different vocab size.
- Fix: use `convert_peft_to_minlora_v2.py` to extract only LoRA tensors (we already used this to fix `brico/...`).

2) "Registering a parametrization may not change the dtype ... unparametrized dtype: float16 parametrized dtype: float32"
- Cause: base model weights were float16 but minLoRA created new LoRA parameters as float32.
- Fix: the `chat_minlora.py` script temporarily sets PyTorch default dtype to the model dtype before calling `minlora.add_lora(...)`. LoRA tensors loaded from files are also cast to the base model dtype.

3) "Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!"
- Cause: minLoRA stores extra adapter tensors (lists `lora_As`, `lora_Bs`) as attributes not registered as parameters/buffers, so `model.to(device)` didn’t move them.
- Fix: `chat_minlora.py` includes a helper that converts these lists into `nn.Parameter` on the target device and dtype after `model.to(device)`.

4) MPS / FP16 stability
- If you experience crashes on MPS with FP16, try falling back to CPU with `dtype=float32` (edit the script to prefer float32), or try a different PyTorch release. We try FP16 by default to reduce memory usage but fallback behavior is included.

Inspecting adapter keys (if you need diagnostics)
-------------------------------------------------
If you want to see which keys are present in a downloaded checkpoint, run a small Python snippet:

```python
import torch
sd = torch.load("pytorch_model.bin", map_location="cpu")
print(sorted(list(sd.keys()))[:200])   # print sample keys
```

Or for a safetensors file:
```python
from safetensors.torch import load_file
sd = load_file("adapter.safetensors")
print(sorted(list(sd.keys()))[:200])
```

If the adapter includes `embed_tokens.weight` or `lm_head.weight`, that's a sign it contains full model weights — use the v2 converter to extract only `lora_` keys.

Memory & performance notes
--------------------------
- `google/gemma-3-1b-it` in FP16 + a few LoRA adapters should fit in 18 GB, but you must keep batch size small and prefer FP16.
- bitsandbytes QLoRA workflows are not generally supported on macOS MPS; do not rely on `load_in_4bit` unless you are on a supported CUDA/NVIDIA machine.
- If you need even faster local inference, merge adapter weights into the base (PEFT `merge_and_unload()` or minLoRA merge function) and convert to a quantized ggml/gguf model for `llama.cpp`-like backends — this is a different workflow.

What else I adjusted for you
----------------------------
- Provided `convert_peft_to_minlora_v2.py` to handle adapters that include extra weights or have different naming, extracting only `lora_*` tensors.
- Patched `chat_minlora.py` to:
  - set default dtype during `add_lora`,
  - cast loaded LoRA tensors to model dtype,
  - move unregistered adapter tensors to the target device/dtype.

Next ideas (optional)
---------------------
- Add a "merge current adapter" button in the UI that runs a `merge_lora(model)` and saves a merged checkpoint.
- Patch minLoRA upstream so adapter tensors are registered as buffers/parameters and follow `model.to()` automatically (I can draft a small PR if you want).
- Add persistent conversation storage per adapter and allow per-adapter system prompts.

Contact / Credits
-----------------
- minLoRA: https://github.com/changjonathanc/minLoRA
- The scripts in this folder were created to convert and adapt HF PEFT LoRA adapters for minLoRA-style multi-adapter runtime on macOS MPS.

If you want, I can:
- produce a single `run.sh` to automate venv creation, torch install check instructions, adapter conversions, and launching the chat,
- or draft the small minLoRA patch to register adapter tensors properly and show the diff.
