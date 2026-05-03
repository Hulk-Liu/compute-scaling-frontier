# Compute-Matched Pareto Frontier

> **Status: Work in Progress.** This is a take-home prototype for the Red Hat AI Innovation Team. Sections marked _TBD_ will be filled in as experiments complete.

**中文版本：[README.zh.md](./README.zh.md)**

---

## TL;DR

When you have a fixed compute budget for an LLM-powered system, where should you spend it — on **fine-tuning** a small model, or on **inference-time scaling** at query time? The answer depends on **how many queries you expect to serve**: training is a one-time cost, inference scales with usage. Somewhere there is a break-even point.

This project finds that point empirically for **Qwen2.5-1.5B-Instruct on GSM8K**, by integrating three Red Hat OSS libraries:

- **[sdg_hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub)** — generate synthetic training data from a stronger teacher model
- **[training_hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub)** — LoRA fine-tune the small student model
- **[its_hub](https://github.com/Red-Hat-AI-Innovation-Team/its_hub)** — apply Self-Consistency and Best-of-N strategies at inference time

## Headline Result

_TBD — to be filled in after experiments complete._

A Pareto frontier in (USD cost, accuracy) space, with curves for query volumes N ∈ {1K, 10K, 100K, 1M}, and the break-even N\* annotated.

## Sections

- [Problem & Approach](#problem--approach) _(TBD)_
- [How to Run](#how-to-run) _(TBD)_
- [Architecture](#architecture) _(TBD)_
- [Cost Accounting Methodology](#cost-accounting-methodology) _(TBD)_
- [Results](#results) _(TBD)_
- [Design Decisions & Scope Boundaries](#design-decisions--scope-boundaries) _(TBD)_
- [What Worked / What Didn't](#what-worked--what-didnt) _(TBD)_
- [Other Tools & Why](#other-tools--why) _(TBD)_
- [What I'd Improve with More Time](#what-id-improve-with-more-time) _(TBD)_
- [AI-Assisted Development](./AI_ASSISTED_DEV.md)

---

## Problem & Approach

_TBD_

## How to Run

_TBD — see `scripts/verify_setup.sh` for the current smoke-test entrypoint._

## Architecture

_TBD_

## Cost Accounting Methodology

_TBD — assumptions live in `prices.yaml`._

## Results

_TBD_

## Design Decisions & Scope Boundaries

_TBD_

## What Worked / What Didn't

_TBD_

## Other Tools & Why

_TBD_

## What I'd Improve with More Time

_TBD_
