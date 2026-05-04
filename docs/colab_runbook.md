# Colab Pro Runbook

This runbook is a cell-by-cell checklist for running the GPU-dependent parts of the project in Google Colab Pro. It is intentionally explicit about where each command runs.

Use this first as a smoke run. After the smoke path works, increase `TRAIN_N` and run the full experiment grid.

Colab shell cells in this document use `%%bash`. Copy the whole fenced block into one Colab code cell. Do not copy only the first line of a heredoc command into a Python cell.

## What Runs Where

Run locally:

- Push the current repo to GitHub.
- Keep editing code, README, and experiment notes.
- Run local unit tests and tiny OpenAI smoke tests.

Run in Colab:

- Install CUDA-dependent LoRA dependencies.
- Generate or reuse larger synthetic training data.
- Execute `training_hub.lora_sft` through `src.train_lora --execute`.
- Save checkpoints and generated artifacts to Google Drive.

## Local Prep

Run locally before opening Colab:

```bash
git status --short
git log --oneline -5
git remote -v
git push
```

If the repo is private, make sure Colab can clone it. Options:

- Use a GitHub fine-grained token with read access.
- Temporarily use a private clone URL with token in the Colab session only.
- Upload a zip if GitHub access is inconvenient.

Do not commit `.env`, API keys, checkpoints, raw JSONL outputs, or generated training JSONL files.

## Colab Runtime Setup

Run in Colab. Use `Runtime > Change runtime type > GPU`.

```bash
%%bash
set -euo pipefail
nvidia-smi
python --version
pwd
```

Expected:

- `nvidia-smi` shows a CUDA GPU.
- Python is `>=3.11`.
- Working directory is usually `/content`.

Mount Google Drive for persistent artifacts:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Define run paths:

```python
from pathlib import Path
import os

PROJECT_ROOT = Path("/content/red-hat-ai-take-home")
DRIVE_ROOT = Path("/content/drive/MyDrive/red-hat-ai-take-home")
CHECKPOINT_ROOT = DRIVE_ROOT / "checkpoints"
RESULTS_ROOT = DRIVE_ROOT / "results"

CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("CHECKPOINT_ROOT:", CHECKPOINT_ROOT)
print("RESULTS_ROOT:", RESULTS_ROOT)

os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["DRIVE_ROOT"] = str(DRIVE_ROOT)
os.environ["CHECKPOINT_ROOT"] = str(CHECKPOINT_ROOT)
os.environ["RESULTS_ROOT"] = str(RESULTS_ROOT)
```

## Clone Repo

Run in Colab. Replace the URL with your repo.

For a public repo:

```bash
%%bash
set -euo pipefail
cd /content
git clone https://github.com/<your-user>/<your-repo>.git red-hat-ai-take-home
cd /content/red-hat-ai-take-home
git status --short
git log --oneline -5
```

For a private repo, prefer a short-lived token stored in Colab secrets. Avoid pasting tokens into committed files.

```python
from google.colab import userdata
import os

os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
```

```bash
%%bash
set -euo pipefail
cd /content
git clone https://oauth2:${GITHUB_TOKEN}@github.com/<your-user>/<your-repo>.git red-hat-ai-take-home
cd /content/red-hat-ai-take-home
git status --short
git log --oneline -5
```

## Install Dependencies

Run in Colab.

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home
pip install -q uv
uv sync
uv pip install --python .venv/bin/python "training-hub[lora]>=0.8.0"
```

Verify the CUDA and LoRA path:

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home
.venv/bin/python - <<'PY'
import unsloth
import torch
import training_hub

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("unsloth import: ok")
print("training_hub:", getattr(training_hub, "__version__", "?"))
PY
```

If this fails, stop and capture the error in the project notes. The most likely issue is a CUDA/Unsloth dependency mismatch. If you see a warning about skipped C++ extensions for the current Torch version, continue to the dry-run step first; treat it as a performance warning unless training fails.

The default training wrapper uses fp16 and disables bf16 because Colab T4 GPUs do not support bf16. If you later get an Ampere+ GPU and want bf16, pass `--bf16 --no-fp16` explicitly.

## Configure Secrets

Run in Colab. Prefer Colab secrets named `OPENAI_API_KEY` and `HF_TOKEN`.

```python
from google.colab import userdata
import os

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")

hf_token = userdata.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

print("OPENAI_API_KEY set:", bool(os.environ.get("OPENAI_API_KEY")))
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
```

Optional Hugging Face login:

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home
if [ -n "${HF_TOKEN:-}" ]; then
  .venv/bin/hf auth login --token "$HF_TOKEN"
else
  echo "HF_TOKEN is not set; skipping Hugging Face login."
fi
```

## Baseline Checks

Run in Colab.

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home
.venv/bin/python -m pytest \
  tests/test_train_lora.py \
  tests/test_validate_training_data.py \
  tests/test_filter_training_data.py
.venv/bin/python -m src.prepare_eval_set --sample-size 50 --output data/eval_gsm8k_50.jsonl
```

Run a tiny `sdg_hub` generation smoke before generating a larger file:

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home
.venv/bin/python -m src.data_generation \
  --n 3 \
  --output data/_smoke_augmented_train_3.jsonl

.venv/bin/python -m src.validate_training_data \
  data/_smoke_augmented_train_3.jsonl \
  --fail-on-mismatch
```

## Generate Training Data

Run in Colab. Start with `TRAIN_N=25` or `100`; increase only after the path works.

```python
import os

TRAIN_N = 100
TRAIN_PATH = f"data/augmented_train_{TRAIN_N}.jsonl"
VALID_TRAIN_PATH = f"data/augmented_train_{TRAIN_N}_valid.jsonl"
INVALID_TRAIN_PATH = f"data/augmented_train_{TRAIN_N}_invalid.jsonl"
print(TRAIN_PATH)
print(VALID_TRAIN_PATH)
print(INVALID_TRAIN_PATH)

os.environ["TRAIN_N"] = str(TRAIN_N)
os.environ["TRAIN_PATH"] = TRAIN_PATH
os.environ["VALID_TRAIN_PATH"] = VALID_TRAIN_PATH
os.environ["INVALID_TRAIN_PATH"] = INVALID_TRAIN_PATH
```

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home
.venv/bin/python -m src.data_generation \
  --n "$TRAIN_N" \
  --output "$TRAIN_PATH" \
  --max-concurrency 1

.venv/bin/python -m src.validate_training_data \
  "$TRAIN_PATH"

.venv/bin/python -m src.filter_training_data \
  "$TRAIN_PATH" \
  --valid-output "$VALID_TRAIN_PATH" \
  --invalid-output "$INVALID_TRAIN_PATH" \
  --min-accuracy 0.95
```

Do not train on the unfiltered file if validation reports mismatches. The filter step writes valid records to `VALID_TRAIN_PATH` and invalid records, with validation metadata, to `INVALID_TRAIN_PATH`.

Persist the generated training file to Drive:

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home
mkdir -p /content/drive/MyDrive/red-hat-ai-take-home/data
cp "$TRAIN_PATH" /content/drive/MyDrive/red-hat-ai-take-home/data/
cp "$VALID_TRAIN_PATH" /content/drive/MyDrive/red-hat-ai-take-home/data/
cp "$INVALID_TRAIN_PATH" /content/drive/MyDrive/red-hat-ai-take-home/data/
ls -lh /content/drive/MyDrive/red-hat-ai-take-home/data/
```

## Dry-Run LoRA Training

Run in Colab.

```python
import os

CKPT_DIR = f"/content/drive/MyDrive/red-hat-ai-take-home/checkpoints/qwen2.5-1.5b-gsm8k-lora-n{TRAIN_N}"
print(CKPT_DIR)

os.environ["CKPT_DIR"] = CKPT_DIR
```

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home
.venv/bin/python -m src.train_lora \
  --data-path "$VALID_TRAIN_PATH" \
  --ckpt-output-dir "$CKPT_DIR"
```

Expected:

- Prints record count.
- Prints `training_hub.lora_sft` kwargs.
- Does not launch training unless `--execute` is present.

## Execute LoRA Training

Run in Colab only after the dry-run output looks correct.

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home
.venv/bin/python -m src.train_lora \
  --data-path "$VALID_TRAIN_PATH" \
  --ckpt-output-dir "$CKPT_DIR" \
  --execute
```

Record the wall-clock training time, GPU type, and any peak memory notes. These are needed for cost accounting and README discussion.

After training:

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home
find "$CKPT_DIR" -maxdepth 2 -type f | sort | head -50
du -sh "$CKPT_DIR"
```

## Save Run Metadata

Run in Colab after training.

```python
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess

repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
gpu = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    text=True,
).splitlines()[0]

notes = "\n".join(
    [
        f"date_utc={datetime.now(timezone.utc).isoformat()}",
        f"repo_commit={repo_commit}",
        f"train_n={os.environ['TRAIN_N']}",
        f"train_path={os.environ['TRAIN_PATH']}",
        f"valid_train_path={os.environ['VALID_TRAIN_PATH']}",
        f"invalid_train_path={os.environ['INVALID_TRAIN_PATH']}",
        f"ckpt_dir={os.environ['CKPT_DIR']}",
        f"gpu={gpu}",
        "",
    ]
)

notes_path = Path("/content/drive/MyDrive/red-hat-ai-take-home/train_run_notes.txt")
notes_path.write_text(notes, encoding="utf-8")
print(notes)
```

## Optional: Copy Artifacts Back Into Repo Workspace

Run in Colab if you want the local Colab workspace to see persisted artifacts:

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home
mkdir -p checkpoints data
cp -r "$CKPT_DIR" checkpoints/
cp "/content/drive/MyDrive/red-hat-ai-take-home/data/augmented_train_${TRAIN_N}_valid.jsonl" data/
```

These files are intentionally ignored by git.

## Handoff Back to Local

Run locally after Colab training completes:

```bash
# Option A: download artifacts from Google Drive manually.
# Option B: use gcloud/rclone if configured.

git status --short
```

Keep checkpoints outside git. In the final submission, include:

- Training command and GPU notes in README.
- Final aggregate CSV and figures.
- A short note that raw JSONL and checkpoints are intentionally omitted because of size.

## Next After Training

After a checkpoint exists, the next project step is to serve base Qwen and the LoRA adapter behind an OpenAI-compatible endpoint. The current inference script is already designed for that:

```bash
.venv/bin/python -m src.run_its_experiment \
  --endpoint http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --strategy sc \
  --budget 4 \
  --n-eval 50 \
  --output results/raw/qwen_base_sc4.jsonl
```

The exact serving command will depend on whether we use vLLM, TGI, or another OpenAI-compatible server. Do not start the full inference grid until the base-model smoke and LoRA-adapter smoke both produce `answer_format_ok_rate=1.0` on a tiny subset.
