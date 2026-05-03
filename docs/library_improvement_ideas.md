# Library Improvement Ideas / 库改进建议

These notes summarize improvement opportunities found while integrating `sdg_hub`, `training_hub`, and `its_hub` for this prototype. They are framed as product and architecture feedback from real usage, not as blockers.

这些记录来自本项目实际集成 `sdg_hub`、`training_hub`、`its_hub` 时遇到的问题。这里的重点不是指出缺陷，而是把真实使用中的 friction 转化成可以讨论的产品和架构改进点。

## 1. its_hub: Task-Aware Self-Consistency Projection

**Observation.** `its_hub.SelfConsistency` already has the right low-level extension point: `consistency_space_projection_func`. However, the default projection votes on the entire stripped response text. For mathematical reasoning, that can be the wrong semantic space. Four responses can all end with `#### 540`, but if their reasoning text differs, exact-text voting treats them as four separate one-vote answers.

**Why it matters.** This is an ML behavior issue, not a small utility issue. Self-Consistency is supposed to estimate answer-level agreement. If the voting space is too literal, inference-time scaling can degrade into random tie-breaking among stylistic variants, which makes the measured value of extra inference compute misleading.

**Suggested direction.** Keep the existing custom function escape hatch, but expose common projection modes as first-class API:

```python
SelfConsistency(vote_on="text")          # current default
SelfConsistency(vote_on="final_number")  # GSM8K / MATH-style numeric answers
SelfConsistency(vote_on="boxed")         # LaTeX \boxed{...}
SelfConsistency(vote_on="regex", pattern=r"####\s*(.+)")
SelfConsistency(projection_func=my_func) # custom escape hatch
```

It would also help if results included the projected values used for voting:

```python
result.projections == ["540", "540", "540", "540"]
result.response_counts == {"540": 4}
```

**What this repo does now.** `src.run_its_experiment` passes a `final_answer_projection` function that reuses the evaluator's final-number extraction, so Self-Consistency votes in the same answer space used for exact-match evaluation.

**中文说明。** `its_hub.SelfConsistency` 已经提供了底层扩展点 `consistency_space_projection_func`，但默认按整段 response text 投票。对数学题来说，这个投票空间不够语义化：多个回答可能都得到 `#### 540`，但推理文字不同，默认投票会把它们拆成多个 1 票答案。

这个问题影响的是 ML 行为本身。Self-Consistency 想衡量的是答案层面的多数一致性；如果投票空间定义错了，inference-time scaling 的收益评估会被扭曲。本项目现在通过 `final_answer_projection` 把投票空间改成最终数字，并和 evaluator 使用同一套 final-number extraction。

## 2. its_hub: Clearer Optional Dependency Contract

**Observation.** A top-level `import its_hub` can pass even when the OpenAI-compatible language-model path is unusable because optional LM dependencies are missing. In this project, installing `its-hub` without `[lm]` led to a missing dependency on the real path, while a shallow smoke test still looked green.

**Suggested direction.**

- Make installation docs explicit about which extras are required for OpenAI-compatible inference.
- Raise a helpful error when optional LM exports are requested without their dependencies.
- Consider a small `its_hub doctor` command that verifies the exact runtime paths a user plans to use.

**What this repo does now.** The dependency is pinned as `its-hub[lm]>=1.0.0`, and the live smoke test instantiates the OpenAI-compatible LM instead of only importing the package.

**中文说明。** 顶层 `import its_hub` 成功并不代表 OpenAI-compatible LM 路径可用。如果缺少 optional LM 依赖，浅层 smoke test 会误报成功。本项目已经把依赖改成 `its-hub[lm]>=1.0.0`，并让 live smoke 真的实例化项目会用到的 LM 路径。

## 3. training_hub: Backend Compatibility Preflight

**Observation.** The LoRA SFT path is the right abstraction for this project, but local execution depends on backend availability. On this Mac setup, the Unsloth-backed LoRA path is not runnable locally, so training is routed to Colab Pro.

**Suggested direction.**

- Add a `training_hub doctor` or `lora_sft(..., dry_run=True)` mode.
- Check platform, CUDA, backend imports, model/data paths, dataset schema, and estimated memory before launching training.
- Return a structured report that users can include in project notes or CI logs.

**What this repo does now.** `src.train_lora` acts as a thin preflight wrapper: it validates chat-template JSONL, builds the `training_hub.lora_sft` kwargs, and defaults to dry-run mode unless `--execute` is passed.

**中文说明。** `training_hub` 的 LoRA SFT 抽象适合本项目，但实际能否运行取决于 backend。当前 Mac 环境无法跑 Unsloth-backed LoRA，所以训练被安排到 Colab Pro。更好的库体验是提供 `doctor` 或 `dry_run=True`，在真正训练前检查平台、CUDA、backend、数据 schema 和显存估算。本 repo 的 `src.train_lora` 目前临时承担了这个 preflight 角色。

## 4. training_hub: First-Class Dataset Schema Validation

**Observation.** LoRA SFT supports chat-template style data, but users still need to discover and validate the expected JSONL shape themselves.

**Suggested direction.** Provide reusable validators such as:

```python
from training_hub import validate_dataset

report = validate_dataset(
    data_path="data/augmented_train.jsonl",
    dataset_type="chat_template",
    field_messages="messages",
)
```

The report could include row counts, missing fields, unsupported roles, empty assistant messages, and examples of malformed rows.

**What this repo does now.** `src.train_lora.validate_training_file` validates the subset of chat-template schema needed by the current `training_hub.lora_sft` call.

**中文说明。** LoRA SFT 支持 chat-template 数据，但用户仍然需要自己确认 JSONL schema 是否正确。库里如果提供 `validate_dataset(...)`，可以直接检查 row count、缺失字段、非法 role、空 assistant message 和坏样例。本 repo 现在在 `src.train_lora.validate_training_file` 里实现了一个最小版本。

## 5. sdg_hub: Built-In Quality Gates and Provenance

**Observation.** `sdg_hub` makes it straightforward to define YAML flows that generate synthetic data. For training workflows, generation is only half the problem; users also need quality checks and provenance.

**Suggested direction.**

- Make evaluator/filter blocks a common built-in pattern after generation.
- Preserve provenance fields such as source id, teacher model, prompt template version, flow version, validation score, and failure reason.
- Support common task validators, for example final-answer exact match for GSM8K-style math data.

**What this repo does now.** `src.validate_training_data` checks whether each generated teacher response has the same final numeric answer as the GSM8K gold answer before allowing the generated records into training.

**中文说明。** `sdg_hub` 很适合用 YAML flow 生成 synthetic data，但训练数据 pipeline 不应该止步于生成；还需要质量过滤和 provenance。理想情况下，flow 可以自然接上 evaluator/filter blocks，并保留 source id、teacher model、prompt template version、flow version、validation score、failure reason 等字段。本 repo 现在用 `src.validate_training_data` 做了一个最小质量门：teacher response 的最终数字必须匹配 GSM8K gold answer。

## Recommended Discussion Order / 推荐讨论顺序

1. **Lead with `its_hub` projection.** It is the strongest ML-depth point because voting-space design directly changes inference-time scaling behavior.
2. **Then discuss `training_hub` preflight.** It shows engineering rigor around running unfamiliar training stacks safely.
3. **Close with `sdg_hub` quality gates.** It connects synthetic data generation to data quality, provenance, and downstream training reliability.

中文推荐顺序：

1. **先讲 `its_hub` projection。** 这是最有 ML 深度的点，因为投票空间会直接改变 inference-time scaling 的行为。
2. **再讲 `training_hub` preflight。** 这体现了面对陌生训练栈时的工程可靠性判断。
3. **最后讲 `sdg_hub` quality gates。** 这能把 synthetic data generation、数据质量、provenance 和训练可靠性串起来。
