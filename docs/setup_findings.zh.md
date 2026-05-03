# Day 0 Setup 发现记录

> 来自 import-only smoke test（`./scripts/verify_setup.sh`）的工作笔记。
> Day 3 会精炼后写到 README 的 "What Worked / What Didn't" section。

**English version: [setup_findings.md](./setup_findings.md)**

---

## 总览

三个库在 Python 3.11 + uv 虚拟环境（macOS Apple Silicon, M2 Pro）下都干净安装、干净导入。唯一预期的失败浮出水面 —— `unsloth` 不可用，所以 `training_hub.lora_sft` 本地跑不起来 —— 这正是 plan 选择走 Colab Pro 的依据。

---

## sdg_hub (v0.9.2)

**正常工作的部分**：
- `uv add sdg-hub` 干净安装
- `FlowRegistry.discover_flows()` 注册了 16 个 built-in flow
- 有几个跟我们 use case 直接相关 —— 特别是 **`stellar-peak-605`（"Document Based Knowledge Tuning Dataset Generation Flow"）**，从原始文档生成 QA 对。我们可能可以 fork 这个 flow 而不是从头写 YAML。

**Doc gap / 小摩擦**：
- 顶层包没有 `__version__` 属性，版本只能从 `pyproject.toml` 读。（`training_hub` 也一样。）
- `list_flows()` 返回的是 `{'id': ..., 'name': ...}` dict 列表而不是裸字符串。README 示例直接 `FlowRegistry.get_flow_path(...)` 传过去用，所以正常使用感知不到，但 first-time user 写 list-flow 代码时会被绊一下（我的 smoke script 就被绊到了）。

---

## its_hub (v1.0.0)

**正常工作的部分**：
- `uv add its_hub` 干净安装
- 顶层 export 包含 `BestOfN`、`SelfConsistency`，以及 abstract base classes（`AbstractLanguageModel`、`AbstractScalingAlgorithm` 等）。

**Doc gap（值得 surface）**：
- README 示例从 `its_hub` 顶层 import `OpenAICompatibleLanguageModel`，但**这个名字不在顶层 namespace 里**。它一定在某个 submodule 里（`its_hub.lms` 或类似 —— Day 1 确认）。
- 这种 README-vs-代码的小漂移，正是 take-home 说明里明确鼓励候选人 surface 的。

---

## training_hub (v0.8.0)

**正常工作的部分**：
- `uv add training-hub`（不加 `[lora]` extra）干净安装
- 干净导入；顶层暴露了预期的 algorithm/backend classes：`LoRASFTAlgorithm`、`UnslothLoRABackend`、`InstructLabTrainingSFTBackend`、`MiniTrainerOSFTBackend`、`LoRAGRPOAlgorithm`，以及它们的 `*Estimator` 变体（用于显存预算估算）。

**预期的失败（我们想确认的 documentation gap）**：
- `import unsloth` → `ModuleNotFoundError`。这就是 `training_hub.lora_sft()` 底下用的 backend。
- Take-home 说明里明确说 "small models like Qwen2.5-1.5B work well on CPU"。实际上 LoRA 路径强制依赖 Unsloth，而 Unsloth 当前没有 Apple Silicon 训练支持（他们文档说 "MLX training coming soon"）。Mac 上唯一让 `lora_sft()` 工作的方式是装 `[lora]` extra，但 extra 本身就装不上（Unsloth wheel 只有 CUDA target）。
- **对项目的含义**：训练放到 Colab Pro。**对 onsite 值得提的点**：take-home 说明里关于 CPU 的说法对 LoRA code path 是误导的；要么修说明，要么 `training_hub` 应该提供一个 CPU/MPS-compatible 的 LoRA backend 作为 fallback（光用 PEFT 就够了）。

**小噪音**：
- 每次 `torch` 导入都会喷 `W0503 ... NOTE: Redirects are currently not supported in Windows or MacOs.`。无害但 verbose。可用 `warnings.filterwarnings` 压住。

**值得记的 side observation**：
- `InstructLabTrainingSFTBackend` 和 `MiniTrainerOSFTBackend` 是不走 Unsloth 的全参 SFT / OSFT backend。**如果 Colab 出问题**，可以本地用其中一个 fallback —— 代价是显存大很多（M2 Pro 上做 1.5B 全参 SFT 可行但紧，~6GB activations + optimizer state）。

---

## Tier 2（live API 调用）—— 暂未运行

Smoke test 支持 `--live` flag，会调 `gpt-4o-mini` 验证 `OPENAI_API_KEY` 端到端通。我们暂缓到用户把 `.env.example` 复制为 `.env` 并填好 key 之后再跑 —— 现在跑只能测"我能否拼出 API call"，这件事 openai SDK 本身就保证了。

---

## 这给 Day 1 解锁了什么

- ✅ 所有 import 通，可以开始针对 `sdg_hub` 写实际的 data-generation flow
- ✅ `its_hub` 的 doc gap（顶层缺 `OpenAICompatibleLanguageModel`）意味着 Day 1 花 5 分钟读源码找正确的 import path 即可
- ✅ Unsloth-on-Mac 已确认，可以 commit 到 Colab Pro 路径，不用再做本地 fallback 探索
