# Compute-Matched Pareto Frontier（计算-精度的帕累托前沿）

> 状态：可运行原型。必需的 Qwen evaluation grid 已完成，repo 里包含 aggregate results、serving-cost estimates、figures、结果分析和 AI-assisted development write-up。Best-of-N 仍然是可选扩展。

**English version: [README.md](./README.md)**

## TL;DR

当一个 LLM 系统有固定 compute 预算时，这笔预算应该花在一次性的 fine-tuning 上，还是花在每个 query 多采样几次的 inference-time scaling 上？这个 trade-off 会随着预期 query volume 改变：训练成本只付一次，而 inference-time scaling 的成本会随着服务请求数线性增长。

本项目用 **GSM8K** 和 **Qwen2.5-1.5B-Instruct** 测这个问题，并集成 Red Hat AI Innovation Team 维护的三个库：

- **[sdg_hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub)**：用更强的 teacher model 生成数学推理 SFT 数据。
- **[training_hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub)**：运行 Qwen2.5-1.5B 的 LoRA fine-tuning。
- **[its_hub](https://github.com/Red-Hat-AI-Innovation-Team/its_hub)**：通过 OpenAI-compatible endpoint 运行 greedy 和 Self-Consistency inference。

最终产物是一个第一版 `(USD cost, accuracy)` Pareto view，query volume 包括 `N in {1K, 10K, 100K, 1M}`。

## 当前结果

完整 Qwen grid 已经在固定的 50 条 GSM8K eval subset 上跑完。Aggregate metrics 在 [results/aggregated.csv](./results/aggregated.csv)，更详细的错误分析在 [docs/results_analysis.md](./docs/results_analysis.md)。Final-grid raw per-example JSONL outputs 已提交到 [results/raw](./results/raw)，方便 reviewer 审计 cost estimate 和 error analysis；smoke outputs 仍然被 ignore。

| train_size | model variant | Greedy | SC@4 | SC@8 | best format rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | base Qwen2.5-1.5B-Instruct | 0.68 | **0.76** | **0.76** | 0.90 |
| 100 | LoRA n100 | 0.38 | 0.46 | 0.52 | 1.00 |
| 500 | LoRA n500 | 0.54 | 0.64 | 0.70 | 1.00 |

![Accuracy heatmap](./results/figures/accuracy_heatmap.png)

核心结果不是“fine-tuning 一定好”或“ITS 一定好”这么简单。Self-Consistency 是最清晰的 accuracy win；synthetic LoRA 更明显地改善了 output controllability 和 final-answer format。Base model 在 accuracy 上仍然最强（SC@4/SC@8 都是 `0.76`），但 heavier sampling 下格式合规率下降。n500 LoRA 的 accuracy 较低（SC@8 是 `0.70`），但整个 grid 的 final-answer formatting 都是合规的。

![Accuracy vs total cost by query volume](./results/figures/cost_accuracy_by_volume.png)

Serving cost 根据 raw generated outputs 的平均 prompt+completion 长度估算，并使用 [prices.yaml](./prices.yaml) 中的 self-hosted Qwen serving 假设。提交的 aggregate 使用 `token_estimation_method=char_heuristic_4`，这样 cost columns 不依赖本地是否缓存了 Qwen tokenizer；在 Colab 或 tokenizer 可用的环境里，同一个 estimator 可以切换到 Hugging Face tokenizer。

低 query volume 下，SC 额外采样的成本很低，所以 base SC@4 是最干净的 accuracy/cost point。高 query volume 下，per-query inference cost 会主导总成本，greedy variants 会变得更便宜；但在这次运行中，fine-tuned models 没有恢复足够多的 accuracy，因此不能在质量上超过 base SC。

一个重要实现细节来自 Self-Consistency smoke：`its_hub.SelfConsistency()` 默认会对整个 stripped response text 投票。对 GSM8K 来说，这不是正确语义空间，因为多个 responses 可能最终都得到 `#### 540`，但推理文字不同，默认投票会把它们拆成不同答案。本项目显式传入 `consistency_space_projection_func=final_answer_projection`，让投票空间变成 evaluator 抽取出来的 final number。

为了让格式问题可见，aggregation 会记录 `has_final_marker_rate`、`answer_format_ok_rate`、`missing_final_marker_count` 和 `malformed_final_marker_count`。

## 问题与方法

核心问题：fine-tuning 和 inference-time scaling 在固定 compute budget 下到底是互补，还是互相替代？

主实验 grid：

- Model variants：base Qwen2.5-1.5B-Instruct (`train_size=0`)、LoRA n100、LoRA n500。
- Inference strategies：greedy、Self-Consistency @4、Self-Consistency @8。
- Optional bonus：如果时间允许，加 Best-of-N @4。
- Cost views：`1K`、`10K`、`100K`、`1M` expected queries。不同 query volume 只是复用同一批 eval outputs 做 cost projection，不需要重新跑模型。

直觉假设是：低流量时，inference-time scaling 可能更划算；高流量时，fine-tuning 的一次性成本被摊薄，理论上应该更有吸引力。这次实验结果显示这个假设只部分成立，因为 LoRA 的 accuracy 没有超过 base model。

## 如何运行

GPU training path 见 [docs/colab_runbook.md](./docs/colab_runbook.md)。

安装依赖：

```bash
uv sync
```

设置 `OPENAI_API_KEY` 到 shell 或 `.env`。`.env` 会被代码读取，但不会提交到 git。

运行 setup smoke tests：

```bash
bash scripts/verify_setup.sh --live
```

准备固定 eval subset：

```bash
.venv/bin/python -m src.prepare_eval_set --sample-size 50 --output data/eval_gsm8k_50.jsonl
```

生成并验证一个 tiny synthetic training set：

```bash
.venv/bin/python -m src.data_generation --n 3 --output data/_smoke_augmented_train_3.jsonl
.venv/bin/python -m src.validate_training_data data/_smoke_augmented_train_3.jsonl --fail-on-mismatch
.venv/bin/python -m src.filter_training_data data/_smoke_augmented_train_3.jsonl
```

本地 dry-run LoRA training call：

```bash
.venv/bin/python -m src.train_lora --data-path data/_smoke_augmented_train_3.jsonl
```

通过 OpenAI-compatible `its_hub` path 跑 tiny inference/eval smoke：

```bash
.venv/bin/python -m src.run_its_experiment \
  --model gpt-4o-mini \
  --strategy greedy \
  --n-eval 3 \
  --output results/raw/_smoke_gpt4omini_greedy.jsonl

.venv/bin/python -m src.run_its_experiment \
  --model gpt-4o-mini \
  --strategy sc \
  --budget 4 \
  --n-eval 3 \
  --output results/raw/_smoke_gpt4omini_sc4.jsonl
```

从一个 raw result file 聚合 metrics：

```bash
.venv/bin/python -m src.aggregate_results results/raw/_smoke_gpt4omini_sc4.jsonl \
  --train-size 0 \
  --strategy sc \
  --budget 4 \
  --model-tokens-per-sample 0 \
  --output results/_smoke_gpt4omini_sc4.csv
```

从 final grid raw outputs 估算 serving-token costs：

```bash
.venv/bin/python -m src.estimate_serving_costs \
  --token-method char \
  --output results/aggregated.csv
```

生成 figures：

```bash
.venv/bin/python -m src.plot_results
```

运行测试：

```bash
.venv/bin/python -m pytest
```

当 Qwen 通过 vLLM 或其他 OpenAI-compatible server 提供服务时，同一个 inference script 只需要替换 `--endpoint` 和 `--model`：

```bash
.venv/bin/python -m src.run_its_experiment \
  --endpoint http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --strategy sc \
  --budget 4 \
  --n-eval 50 \
  --output results/raw/qwen_base_sc4.jsonl
```

## 架构

当前原型刻意保持 script-first：

1. `src.prepare_eval_set` 采样固定 GSM8K eval subset。
2. `src.data_generation` 读取 GSM8K train rows，并运行自定义 `sdg_hub` flow。
3. `src.validate_training_data` 检查 teacher answers 是否匹配 GSM8K gold answers。
4. `src.train_lora` 验证 chat-template data，并准备 `training_hub.lora_sft` call。
5. `src.run_its_experiment` 通过 `its_hub` 跑 greedy 或 Self-Consistency inference cell。
6. `src.run_eval_grid` 跑 3-model x 3-strategy Qwen eval grid，并写 raw cell outputs。
7. `src.aggregate_results` 计算 exact-match accuracy 和初始 cost columns。
8. `src.estimate_serving_costs` 从 raw outputs 估算 serving-token cost，并更新 aggregate cost columns。
9. `src.plot_results` 从 `results/aggregated.csv` 生成 figures。

## 成本核算方法

所有价格假设都在 [prices.yaml](./prices.yaml)，由 `src.cost_accounting` 读取。

当前模型拆成四部分：

- Synthetic data generation cost per example。
- One-time training cost：generated-example count + GPU hours。
- Per-query inference cost：model tokens、strategy budget、可选 judge tokens。
- Total serving cost：`training_cost + query_count * inference_cost_per_query`。

对提交的 Qwen grid，`src.estimate_serving_costs` 从 raw prompt 和 completion text 估算 `model_tokens_per_sample`，再按 strategy budget 放大。默认 `auto` mode 会优先尝试 Qwen Hugging Face tokenizer，失败时 fallback 到 4-characters-per-token heuristic；提交的 CSV 使用显式 heuristic path，避免依赖本地 tokenizer cache。

这个 accounting model 故意保持简单，但它把 break-even 思路暴露得很清楚，review 时也容易替换价格假设。

## 设计决策与 Scope 边界

- **先统一 OpenAI-compatible abstraction。** 同一条 `its_hub.OpenAICompatibleLanguageModel` path 可以服务 OpenAI smoke tests，也可以服务 Qwen/vLLM。
- **数学 Self-Consistency 用 final-answer projection。** 对整个 response text 投票太脆弱，所以本项目对抽取出的 final number 投票。
- **本地 dry-run，Colab 执行训练。** `training_hub` 的 LoRA path 使用 Unsloth backend，本地 Mac 环境无法直接跑；本地脚本负责验证 data 和 kwargs，Colab T4 执行真实 LoRA run。
- **先小规模 smoke，再跑固定 eval。** 开发时用 `n_eval=3` 控制调试成本，路径稳定后再跑 50-row fixed subset。
- **提交最终证据，忽略 smoke noise。** Final-grid raw outputs 会提交，因为它们支撑 cost estimation 和 error analysis；smoke outputs、generated training JSONL 和 checkpoints 继续忽略。

## 什么有效 / 什么没效

有效：

- `sdg_hub.Flow` 可以运行自定义 GSM8K teacher-response flow，并用 `gpt-4o-mini` 生成合成训练数据。
- Synthetic generation 生成了两个过滤后的 LoRA training sets：100-sample run 中 99 条 valid，500-sample run 中 497 条 valid。
- `training_hub.lora_sft` 在 Colab T4 上通过 Unsloth backend 完成了 n100 和 n500 的 Qwen2.5-1.5B LoRA runs。
- 保存的 LoRA adapters 可以在 Colab 中重新加载，也可以通过 vLLM 暴露为 OpenAI-compatible models。
- `its_hub` greedy 和 Self-Consistency paths 在安装 `[lm]` extra 后可用。
- evaluator、aggregation、serving-cost estimator 和 cost-accounting tests 让实验输出可检查。

不顺利：

- 如果只安装 `its-hub` 而不带 `[lm]` extra，OpenAI-compatible LM export 会静默缺失。依赖已改成 `its-hub[lm]>=1.0.0`。
- 只 import top-level `its_hub` 的 smoke test 不够；live smoke 必须实例化项目实际使用的 LM path。
- 一个只直接调用 OpenAI 的 `sdg_hub` live smoke 会误导人；现在 smoke 会跑 tiny `sdg_hub.Flow`。
- 本地 Mac LoRA execution 被 Unsloth backend 阻塞，所以训练放到 Colab Pro。
- 太低的 `max_tokens` 会截断数学输出，让 “last number” extraction 看起来通过，但 strict final-line format 不合规。`256`-token smoke 暴露后，inference default 改成 `512`。

Library improvement ideas 的中英双语记录见 [docs/library_improvement_ideas.md](./docs/library_improvement_ideas.md)。

## 其它工具

- `uv`：本地 dependency management。
- `pytest`：覆盖 data shape、evaluation、aggregation、cost accounting、training-call preparation。
- Hugging Face `datasets`：加载 GSM8K。
- OpenAI `gpt-4o-mini`：低成本 teacher 和早期 smoke model。
- Colab Pro：Qwen LoRA training 和 vLLM OpenAI-compatible serving。

## 更多时间会改进什么

1. 加 Best-of-N @4；如果使用外部 verifier，需要明确标注为 judge-assisted strategy。
2. 如果时间/算力允许，把 eval size 从 50 扩到更大。
3. 根据 LoRA failure modes 做第二轮 targeted synthetic data generation。
