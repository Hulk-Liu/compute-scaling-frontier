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

Do not commit `.env`, API keys, checkpoints, generated training JSONL files, or smoke raw JSONL outputs. Final-grid raw JSONL outputs can be committed because they are small and support cost/error auditability.

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

## Smoke Test the LoRA Adapter

Run in Colab before generating a larger training set. This step loads the saved LoRA adapter directly with Unsloth and runs greedy generation on three eval questions. It does not use `its_hub` yet; the goal is to prove the adapter artifact can be loaded and produces parseable answers.

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home
.venv/bin/python -m src.smoke_lora_adapter \
  --adapter-path "$CKPT_DIR" \
  --n-eval 3 \
  --output results/raw/_smoke_qwen_lora_n100_greedy.jsonl

.venv/bin/python -m src.aggregate_results \
  results/raw/_smoke_qwen_lora_n100_greedy.jsonl \
  --train-size "$TRAIN_N" \
  --strategy lora_greedy \
  --budget 1 \
  --train-gpu-hours 0.03 \
  --model-tokens-per-sample 0 \
  --output results/_smoke_qwen_lora_n100_greedy.csv

sed -n '1,2p' results/_smoke_qwen_lora_n100_greedy.csv
```

The `--train-gpu-hours 0.03` value is a placeholder for the 99-record smoke run. Replace it with the measured GPU time for final experiments.

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
- A short note that checkpoints are intentionally omitted because of size, while final-grid raw JSONL outputs are committed for auditability.

## Qwen Serving Smoke

Run this after the n100 and n500 LoRA adapter smokes have passed. The goal is to prove that the same `its_hub.OpenAICompatibleLanguageModel` path can talk to a local Qwen server before launching the full grid.

Install vLLM after training is complete. This may update inference dependencies inside the Colab venv, so do it after the `training_hub` LoRA runs are done.

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home

uv pip install vllm --torch-backend=auto
which vllm
vllm --version
```

Start one OpenAI-compatible vLLM server with the base model and both LoRA adapters. The adapter names below become the `--model` values for eval requests.

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home

export VLLM_API_KEY=local-vllm
export VLLM_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export LORA_N100=/content/drive/MyDrive/red-hat-ai-take-home/checkpoints/qwen2.5-1.5b-gsm8k-lora-n100
export LORA_N500=/content/drive/MyDrive/red-hat-ai-take-home/checkpoints/qwen2.5-1.5b-gsm8k-lora-n500
export VLLM_BIN="$(command -v vllm)"

pkill -f "vllm serve" || true
nohup "$VLLM_BIN" serve "$VLLM_BASE_MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --enable-lora \
  --max-loras 2 \
  --max-lora-rank 16 \
  --lora-modules \
    qwen-gsm8k-lora-n100="$LORA_N100" \
    qwen-gsm8k-lora-n500="$LORA_N500" \
  > /content/vllm_qwen_lora.log 2>&1 &

echo $! > /content/vllm_server.pid
echo "Started vLLM server pid=$(cat /content/vllm_server.pid)"
```

Wait for the server to become ready and confirm the base model plus both LoRA adapters are listed.

```bash
%%bash
set -euo pipefail

export VLLM_API_KEY=local-vllm
for i in $(seq 1 180); do
  if curl -sf \
    -H "Authorization: Bearer $VLLM_API_KEY" \
    http://127.0.0.1:8000/v1/models > /content/vllm_models.json; then
    cat /content/vllm_models.json
    exit 0
  fi

  if ! kill -0 "$(cat /content/vllm_server.pid)" 2>/dev/null; then
    echo "vLLM process exited."
    tail -160 /content/vllm_qwen_lora.log
    exit 1
  fi

  sleep 2
done

tail -160 /content/vllm_qwen_lora.log
exit 1
```

Run the tiny Qwen serving smoke. Keep `n_eval=3`; this is a server/API acceptance test, not the final benchmark.

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home

export OPENAI_API_KEY=local-vllm
export VLLM_ENDPOINT=http://127.0.0.1:8000/v1

.venv/bin/python -m src.run_its_experiment \
  --endpoint "$VLLM_ENDPOINT" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --strategy greedy \
  --n-eval 3 \
  --output results/raw/_smoke_qwen_base_vllm_greedy.jsonl

.venv/bin/python -m src.run_its_experiment \
  --endpoint "$VLLM_ENDPOINT" \
  --model qwen-gsm8k-lora-n500 \
  --strategy greedy \
  --n-eval 3 \
  --output results/raw/_smoke_qwen_lora_n500_vllm_greedy.jsonl

.venv/bin/python -m src.run_its_experiment \
  --endpoint "$VLLM_ENDPOINT" \
  --model qwen-gsm8k-lora-n500 \
  --strategy sc \
  --budget 4 \
  --n-eval 3 \
  --output results/raw/_smoke_qwen_lora_n500_vllm_sc4.jsonl
```

Aggregate the smoke outputs. The `--model-tokens-per-sample 0` placeholder keeps this smoke focused on correctness; final runs should use a measured or documented inference-cost estimate.

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home

.venv/bin/python -m src.aggregate_results \
  results/raw/_smoke_qwen_base_vllm_greedy.jsonl \
  --train-size 0 \
  --strategy greedy \
  --budget 1 \
  --train-gpu-hours 0 \
  --model-tokens-per-sample 0 \
  --output results/_smoke_qwen_base_vllm_greedy.csv

.venv/bin/python -m src.aggregate_results \
  results/raw/_smoke_qwen_lora_n500_vllm_greedy.jsonl \
  --train-size 500 \
  --strategy greedy \
  --budget 1 \
  --train-gpu-hours 0.125 \
  --model-tokens-per-sample 0 \
  --output results/_smoke_qwen_lora_n500_vllm_greedy.csv

.venv/bin/python -m src.aggregate_results \
  results/raw/_smoke_qwen_lora_n500_vllm_sc4.jsonl \
  --train-size 500 \
  --strategy sc \
  --budget 4 \
  --train-gpu-hours 0.125 \
  --model-tokens-per-sample 0 \
  --output results/_smoke_qwen_lora_n500_vllm_sc4.csv

sed -n '1,2p' results/_smoke_qwen_base_vllm_greedy.csv
sed -n '1,2p' results/_smoke_qwen_lora_n500_vllm_greedy.csv
sed -n '1,2p' results/_smoke_qwen_lora_n500_vllm_sc4.csv
```

Do not start the full inference grid until these smoke rows have `answer_format_ok_rate=1.0` or until any format failures are understood and documented.

Stop the server when done:

```bash
%%bash
set -euo pipefail
if [ -f /content/vllm_server.pid ]; then
  kill "$(cat /content/vllm_server.pid)" || true
fi
```

## Final Eval Grid

After the Qwen serving smoke passes, run the required grid:

- Model ids: `Qwen/Qwen2.5-1.5B-Instruct`, `qwen-gsm8k-lora-n100`, `qwen-gsm8k-lora-n500`.
- Train sizes: `0`, `100`, `500`.
- Strategies: greedy budget `1`, Self-Consistency budget `4`, Self-Consistency budget `8`.
- Eval size: `50`.

First confirm the planned cells:

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home

.venv/bin/python -m src.run_eval_grid \
  --endpoint http://127.0.0.1:8000/v1 \
  --n-eval 50 \
  --dry-run
```

Then run the grid. Keep the vLLM server running from the smoke step.

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home

export OPENAI_API_KEY=local-vllm

.venv/bin/python -m src.run_eval_grid \
  --endpoint http://127.0.0.1:8000/v1 \
  --n-eval 50 \
  --output results/aggregated.csv

sed -n '1,20p' results/aggregated.csv
```

If Colab disconnects or one cell fails after some raw JSONL files have been written, restart the vLLM server if needed and rerun with `--resume`:

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home

export OPENAI_API_KEY=local-vllm

.venv/bin/python -m src.run_eval_grid \
  --endpoint http://127.0.0.1:8000/v1 \
  --n-eval 50 \
  --resume \
  --output results/aggregated.csv
```

The first final grid uses `--model-tokens-per-sample 0`, so its initial cost columns include training cost but not serving-token cost yet. After the accuracy grid is stable, estimate serving cost from the raw outputs and regenerate figures:

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home

.venv/bin/python -m src.estimate_serving_costs \
  --aggregate results/aggregated.csv \
  --raw-dir results/raw \
  --token-method auto \
  --output results/aggregated.csv

.venv/bin/python -m src.plot_results

sed -n '1,20p' results/aggregated.csv
ls -lh results/figures
```

`--token-method auto` tries the Qwen Hugging Face tokenizer first and falls back to the documented 4-characters-per-token heuristic if the tokenizer is unavailable. For exact reproducibility with the committed local aggregate, use `--token-method char`.

Package final result artifacts for local download:

```bash
%%bash
set -euo pipefail
cd /content/red-hat-ai-take-home

tar -czf /content/results_final_grid.tar.gz results
ls -lh /content/results_final_grid.tar.gz
```

Download `/content/results_final_grid.tar.gz` from the Colab file browser, then unpack it in the local repo and commit `results/aggregated.csv`, final figures, and final-grid raw JSONL outputs. Smoke JSONL outputs remain ignored by git.

Best-of-N @4 is optional and should only be added after the required grid is complete.
