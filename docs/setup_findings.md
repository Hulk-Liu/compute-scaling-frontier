# Day 0 Setup Findings

> Working notes from the import-only smoke tests (`./scripts/verify_setup.sh`).
> Will be distilled into the README "What Worked / What Didn't" section in Day 3.

**中文版本：[setup_findings.zh.md](./setup_findings.zh.md)**

---

## Summary

All three libraries install cleanly into a Python 3.11 + uv virtual env on macOS (Apple Silicon, M2 Pro). All three import successfully. The only expected failure surfaced — `unsloth` is not available, so `training_hub.lora_sft` cannot run locally — and is the reason the plan routes training through Colab Pro.

---

## sdg_hub (v0.9.2)

**Worked**:
- Clean install via `uv add sdg-hub`.
- `FlowRegistry.discover_flows()` registered 16 built-in flows.
- Several look directly relevant to our use case — particularly **`stellar-peak-605` ("Document Based Knowledge Tuning Dataset Generation Flow")** which generates QA pairs from raw documents. We may be able to fork this rather than write our flow YAML from scratch.

**Doc gaps / minor friction**:
- No `__version__` attribute on the top-level package — version had to be read from `pyproject.toml`. (Also true for `training_hub`.)
- The API for listing flows returns `dict` objects with `{'id': ..., 'name': ...}`, not bare strings — the README example shows it being passed straight into `Flow.from_yaml(FlowRegistry.get_flow_path(...))`, so the dict shape isn't important for normal use, but it tripped my smoke script and would trip first-time users.

---

## its_hub (v1.0.0)

**Worked**:
- Clean install via `uv add its_hub`.
- Top-level exports include `BestOfN`, `SelfConsistency`, plus the abstract base classes (`AbstractLanguageModel`, `AbstractScalingAlgorithm`, etc.).

**Doc gap (worth surfacing)**:
- The README example imports `OpenAICompatibleLanguageModel` from `its_hub` directly, but **that name is not in the top-level namespace**. It must live in a submodule (`its_hub.lms` or similar — to be confirmed in Day 1).
- This is the kind of small README-vs-code drift that the take-home brief explicitly invites candidates to surface.

---

## training_hub (v0.8.0)

**Worked**:
- Clean install via `uv add training-hub` (without the `[lora]` extra).
- Imports cleanly; top-level surface exposes the expected algorithm/backend classes: `LoRASFTAlgorithm`, `UnslothLoRABackend`, `InstructLabTrainingSFTBackend`, `MiniTrainerOSFTBackend`, `LoRAGRPOAlgorithm`, plus their `*Estimator` variants for memory budgeting.

**Expected failure (the documentation gap we wanted to confirm)**:
- `import unsloth` → `ModuleNotFoundError`. This is the backend that `training_hub.lora_sft()` calls into.
- The take-home brief itself states: "small models like Qwen2.5-1.5B work well on CPU." In practice the LoRA path requires Unsloth, and Unsloth currently has no Apple Silicon training support (their docs say "MLX training coming soon"). On a Mac the only way to use `lora_sft()` is to install the `[lora]` extra, which itself fails because Unsloth's wheel targets CUDA.
- **Implication for the project**: train in Colab Pro. **Implication worth raising during onsite**: the brief's CPU claim is misleading for the LoRA code path; either the brief should be updated, or `training_hub` should ship a CPU/MPS-compatible LoRA backend as a fallback (PEFT alone would suffice).

**Minor noise**:
- `torch` import emits `W0503 ... NOTE: Redirects are currently not supported in Windows or MacOs.` on every run. Harmless, but verbose. Suppressible via `warnings.filterwarnings`.

**Side observations worth remembering**:
- `InstructLabTrainingSFTBackend` and `MiniTrainerOSFTBackend` are full-SFT and OSFT backends that don't go through Unsloth. **If Colab access ever becomes a problem**, we could fall back to one of these locally — at the cost of much higher memory (full SFT of 1.5B on M2 Pro is feasible but tight; ~6GB activations + optimizer state).

---

## Tier 2 (live API) — not run yet

The smoke tests support a `--live` flag that calls `gpt-4o-mini` to confirm `OPENAI_API_KEY` works end-to-end. We deferred this until the user copies `.env.example` to `.env` and fills in the key — running it now would only test that I can compose an API call, which the openai SDK already guarantees.

---

## What this unlocks for Day 1

- ✅ All imports work, so we can start writing the actual data-generation flow against `sdg_hub`
- ✅ The `its_hub` doc gap (top-level missing `OpenAICompatibleLanguageModel`) just means we'll spend 5 min reading the source on Day 1 to find the correct import path
- ✅ The Unsloth-on-Mac confirmation means we can commit fully to the Colab Pro path without further local-fallback exploration
