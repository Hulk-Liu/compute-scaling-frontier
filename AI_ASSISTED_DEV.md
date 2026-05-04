# AI-Assisted Development Notes

**中文版本：[AI_ASSISTED_DEV.zh.md](./AI_ASSISTED_DEV.zh.md)**

This project was built with AI assistance throughout, mainly Claude Code for early planning and Codex for implementation review, debugging, and documentation polish. I treated the tools as pair-programming partners rather than as an authority: useful for speed and breadth, but always subject to local tests, source-code inspection, and small reversible steps.

## Tools and Workflow Structure

- **Claude Code** helped with initial project framing, plan comparison, and the first repo skeleton.
- **Codex** helped review the skeleton, implement focused changes, write tests, debug Colab/vLLM issues, and turn raw experiment results into analysis.
- I kept the workflow incremental: one small slice at a time, run tests or smoke checks, then commit.
- For unfamiliar Red Hat libraries, I used AI to map likely API paths, but verified behavior by reading docs/source, importing real modules, and running small end-to-end smokes.

## Where AI Accelerated the Work

- It quickly reframed the project from a generic fine-tuning demo into a sharper question: under a fixed compute budget, when should spend move from training-time compute to inference-time scaling?
- It helped design a practical grid that fit a take-home scope: `train_size in {0, 100, 500}` crossed with greedy, SC@4, and SC@8 on a 50-row GSM8K subset.
- It caught integration gaps early, including missing `its-hub[lm]` extras, smoke tests that could false-pass, and result files that should not be ignored.
- It made Colab debugging faster by turning errors into the next minimal diagnostic: first CUDA/Unsloth availability, then training kwargs, then vLLM model listing, then tiny serving smokes.
- It helped extract experiment interpretation from raw results, especially the distinction between accuracy gains and format-control gains.

## Where AI Fell Short

- Early smoke tests were too weak. A top-level `import its_hub` passed even when the actual OpenAI-compatible LM path was unavailable.
- A first `sdg_hub` smoke only checked an OpenAI call and did not prove that the project flow used `sdg_hub`.
- Some commands needed correction in Colab because environment details differed from local assumptions, for example `huggingface-cli` being deprecated in favor of `hf`, and `vllm` being installed outside `.venv/bin`.
- AI suggestions were sometimes too optimistic about training quality. The final results showed that synthetic LoRA improved formatting but hurt accuracy versus the base model.

## Review and Validation Practices

- I required live smokes to exercise the real integration path, not just imports.
- I added unit tests around answer extraction, data validation, aggregation, training-call preparation, grid planning, plotting, and serving-cost estimation.
- I used tiny runs first (`n_eval=3`, `TRAIN_N=3/100`) before spending time and API calls on the full grid.
- I inspected raw examples after aggregation, especially cases where fine-tuning hurt and where Self-Consistency recovered errors.
- I documented known caveats instead of hiding them: 50-row eval size, cost-estimation assumptions, tokenizer fallback, and omitted raw JSONL files.

## Recommended Team Practices

1. Use AI to generate hypotheses and implementation options, but require a concrete validation gate before accepting a change.
2. Keep AI-assisted work small and committable. Small commits make it easier to isolate whether an AI suggestion helped or introduced drift.
3. Write smoke tests that exercise the exact production path. Import-only checks are usually not enough for optional dependency stacks.
4. Ask AI to explain trade-offs, then verify the claims against source code, docs, and observed behavior.
5. Treat failed AI suggestions as useful signals: they often reveal documentation gaps, implicit assumptions, or confusing API boundaries.
6. Preserve an audit trail: commands, run metadata, aggregate outputs, and concise notes about what changed between iterations.
