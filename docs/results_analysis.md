# Results Analysis

This note summarizes the first full Qwen evaluation grid:

- Eval set: 50 deterministic GSM8K test rows.
- Model variants: base Qwen2.5-1.5B-Instruct, LoRA n100, LoRA n500.
- Inference strategies: greedy, Self-Consistency @4, Self-Consistency @8.
- Final-grid raw outputs are committed under `results/raw/` for auditability; smoke outputs remain ignored by git.
- Aggregate metrics are in `results/aggregated.csv`.

## Main Accuracy Result

| train_size | strategy | budget | accuracy | answer_format_ok_rate |
| ---: | --- | ---: | ---: | ---: |
| 0 | greedy | 1 | 0.68 | 0.90 |
| 0 | sc | 4 | 0.76 | 0.84 |
| 0 | sc | 8 | 0.76 | 0.78 |
| 100 | greedy | 1 | 0.38 | 0.98 |
| 100 | sc | 4 | 0.46 | 0.98 |
| 100 | sc | 8 | 0.52 | 1.00 |
| 500 | greedy | 1 | 0.54 | 1.00 |
| 500 | sc | 4 | 0.64 | 1.00 |
| 500 | sc | 8 | 0.70 | 1.00 |

The strongest accuracy cell is still the base model with Self-Consistency @4 or @8 (`0.76`). The n500 LoRA model improves over n100 and benefits from inference-time scaling, but it does not surpass the base model on this 50-row sample.

## Cost Result

Serving cost is estimated from the raw prompt and completion text using the self-hosted Qwen assumptions in `prices.yaml`: A10-class GPU at `$0.40/hr`, sustained throughput of `600 tok/s`, and therefore roughly `$0.18 / 1M tokens`. The committed aggregate uses `token_estimation_method=char_heuristic_4` so the result does not depend on a locally cached Qwen tokenizer. This is an estimate, not metered production billing.

| train_size | strategy | budget | accuracy | tokens/sample | cost/query | total@1K | total@1M |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | greedy | 1 | 0.68 | 316 | $0.000057 | $0.057 | $56.88 |
| 0 | sc | 4 | 0.76 | 302 | $0.000217 | $0.217 | $217.44 |
| 0 | sc | 8 | 0.76 | 313 | $0.000451 | $0.451 | $450.72 |
| 100 | greedy | 1 | 0.38 | 180 | $0.000032 | $0.144 | $32.51 |
| 100 | sc | 4 | 0.46 | 171 | $0.000123 | $0.235 | $123.23 |
| 100 | sc | 8 | 0.52 | 173 | $0.000249 | $0.361 | $249.23 |
| 500 | greedy | 1 | 0.54 | 193 | $0.000035 | $0.585 | $35.29 |
| 500 | sc | 4 | 0.64 | 195 | $0.000140 | $0.690 | $140.95 |
| 500 | sc | 8 | 0.70 | 193 | $0.000278 | $0.828 | $278.47 |

The cost view changes the interpretation by query volume:

- At `1K` queries, training cost is still visible, and base SC@4 is an attractive point: `0.76` accuracy for about `$0.22` total estimated cost.
- At `1M` queries, per-query sampling dominates. Base SC@4 still has the best accuracy, but it costs about `$217` versus about `$57` for base greedy.
- The n500 LoRA model is cheaper than base SC@4 at high volume only when using greedy decoding, but its accuracy is lower (`0.54` vs `0.76`).
- The n500 SC@8 model reaches `0.70`, but it costs more than base SC@4 at `1M` queries in this estimate because it uses twice as many samples and still trails in accuracy.

## What Fine-Tuning Changed

Fine-tuning improved output controllability more clearly than answer accuracy.

- Base greedy answered 34/50 correctly, but only 45/50 outputs had a valid final marker.
- Base SC@8 answered 38/50 correctly, but only 39/50 outputs had a valid final marker.
- n500 had valid final-answer formatting for all 150 outputs across greedy, SC@4, and SC@8.

The trade-off is that both LoRA adapters reduced raw GSM8K accuracy relative to the base model:

| comparison | helped | hurt | net |
| --- | ---: | ---: | ---: |
| n100 greedy vs base greedy | 4 | 19 | -15 |
| n500 greedy vs base greedy | 3 | 10 | -7 |

n500 is materially better than n100, but the synthetic SFT signal appears too small or too narrow to improve reasoning accuracy over the pretrained instruction model.

## What Inference-Time Scaling Changed

Self-Consistency improved accuracy for every model variant:

| model | SC@4 - greedy | SC@8 - SC@4 | SC@8 - greedy |
| --- | ---: | ---: | ---: |
| base | +0.08 | +0.00 | +0.08 |
| n100 | +0.08 | +0.06 | +0.14 |
| n500 | +0.10 | +0.06 | +0.16 |

For the base model, the benefit saturated at budget 4. For LoRA, budget 8 continued to help. This suggests fine-tuning and inference-time scaling did not simply substitute for each other; in these runs, LoRA made outputs more format-stable while SC recovered some reasoning errors by sampling multiple candidate answers.

## Self-Consistency Oracle Gap

The raw SC outputs also show that the correct answer often appeared among candidates but was not selected by majority vote.

| model | strategy | selected_correct | oracle_contains_correct | missed_by_vote |
| --- | --- | ---: | ---: | ---: |
| base | SC@4 | 38 | 45 | 7 |
| base | SC@8 | 38 | 43 | 5 |
| n100 | SC@4 | 23 | 34 | 11 |
| n100 | SC@8 | 26 | 40 | 14 |
| n500 | SC@4 | 32 | 39 | 7 |
| n500 | SC@8 | 35 | 42 | 7 |

This is a useful diagnostic for the optional Best-of-N direction: there is headroom beyond simple majority voting, but exploiting it requires a better selector or verifier. If that selector is an external LLM judge, the result should be described as judge-assisted reranking rather than pure small-model inference-time scaling.

## Representative Error Patterns

### Fine-Tuning Hurt Some Previously Correct Problems

Examples where base greedy was correct but n500 greedy was wrong:

- `gsm8k-test-125`: spoon-count algebra. Base extracted `10`; n500 extracted `1`.
- `gsm8k-test-156`: two-month banana order. Base extracted `1400`; n500 extracted `600`.
- `gsm8k-test-421`: cats remaining after boats and a fraction runs away. Base extracted `12`; n500 extracted `15`.
- `gsm8k-test-458`: percentage of spotted puppies. Base extracted `35`; n500 extracted `60`.
- `gsm8k-test-519`: double Rob's time plus 40 minutes. Base extracted `280`; n500 extracted `440`.

These failures look like reasoning regressions, not parser failures. The LoRA model is format-compliant but sometimes performs shallower arithmetic or misreads quantities.

### Self-Consistency Fixed Several LoRA Greedy Errors

Examples where n500 greedy was wrong and n500 SC@8 selected the right answer:

- `gsm8k-test-125`: greedy `1`; SC@8 selected `10` with counts `{"10": 4, ...}`.
- `gsm8k-test-156`: greedy `600`; SC@8 selected `1400` with counts `{"1400": 4, ...}`.
- `gsm8k-test-421`: greedy `15`; SC@8 selected `12` with counts `{"12": 5, "15": 2, ...}`.
- `gsm8k-test-508`: greedy `125`; SC@8 selected `120`, but votes were highly fragmented.
- `gsm8k-test-519`: greedy `440`; SC@8 selected `280` with counts `{"280": 4, "440": 2, ...}`.

This supports the interpretation that inference-time scaling can recover from stochastic reasoning failures even after fine-tuning.

## Current Interpretation

The first full grid supports a nuanced answer:

1. Inference-time scaling is the clearest accuracy win in this setup.
2. The synthetic LoRA runs improve answer formatting and controllability, but they do not improve accuracy over the base model.
3. More synthetic data helps relative to less synthetic data: n500 consistently beats n100.
4. Fine-tuning and SC are not pure substitutes. Here, FT improves format reliability while SC improves answer accuracy.
5. Cost projection makes the traffic-volume trade-off explicit: SC is cheap enough to justify at small volumes, but high-volume serving makes extra samples materially more expensive.
6. The SC oracle gap motivates a possible Best-of-N or verifier-guided extension, but that should be framed as a separate judge-assisted strategy.

The main caveat is sample size: this is a 50-row evaluation designed for a take-home prototype. The result should be presented as evidence about this pipeline and data recipe, not as a general claim about Qwen2.5 or LoRA on GSM8K.
