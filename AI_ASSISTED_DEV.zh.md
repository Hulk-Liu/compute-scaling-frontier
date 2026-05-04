# AI 辅助开发记录

**English version: [AI_ASSISTED_DEV.md](./AI_ASSISTED_DEV.md)**

这个项目从规划到实现都使用了 AI 辅助，早期主要用 Claude Code 做方向选择和骨架设计，后期主要用 Codex 做代码 review、实现、Colab/debugging 和文档整理。我把 AI 当作 pair-programming partner，而不是权威来源：它能提升速度和覆盖面，但每个关键结论都需要通过本地测试、源码检查和小规模端到端 smoke 来验证。

## 工具与 Workflow

- **Claude Code**：帮助做项目 framing、比较不同方案，并生成第一版 repo skeleton。
- **Codex**：帮助 review skeleton、补小步实现、写测试、debug Colab/vLLM 问题、把实验输出整理成分析文档。
- 我把 workflow 保持得很小：一个小 slice、一个测试或 smoke、一次 commit。
- 对不熟悉的 Red Hat libraries，我先让 AI 帮忙找可能的 API path，但一定会通过真实 import、源码/文档阅读和 tiny end-to-end smoke 来确认。

## AI 明显加速的地方

- 它帮助把项目从普通 fine-tuning demo 重构成更尖锐的问题：固定 compute budget 下，什么时候应该把预算从 training-time compute 转到 inference-time scaling？
- 它帮助把 scope 收敛成 take-home 可完成的 grid：`train_size in {0, 100, 500}`，配合 greedy、SC@4、SC@8，在 50 条 GSM8K subset 上运行。
- 它较早发现了 integration gaps：比如 `its-hub[lm]` extra 缺失、smoke test 可能 false-pass、final figures 不应被 `.gitignore` 忽略。
- 它让 Colab debugging 更快：每次错误都转化成下一个最小诊断步骤，比如 CUDA/Unsloth availability、training kwargs、vLLM model listing、tiny serving smoke。
- 它帮助从 raw results 中提炼出更准确的解释：这次 LoRA 的主要收益是 format/control，而不是 answer accuracy。

## AI 拖慢或出错的地方

- 早期 smoke tests 太弱。只 `import its_hub` 会通过，但真正需要的 OpenAI-compatible LM path 仍然可能不可用。
- 第一版 `sdg_hub` live smoke 只直接调用 OpenAI，没有证明项目实际使用的 `sdg_hub.Flow` 能运行。
- 有些 Colab 命令需要根据真实环境修正，比如 `huggingface-cli` 已被 `hf` 替代，以及 `vllm` 安装在 `/usr/local/bin` 而不是 `.venv/bin`。
- AI 对 training quality 的预期有时过于乐观。最终结果显示 synthetic LoRA 改善了格式合规，但相对 base model 降低了 accuracy。

## Review 与验证方式

- Live smoke 必须跑真实 integration path，而不是只做 import check。
- 我为 answer extraction、data validation、aggregation、training-call preparation、grid planning、plotting 和 serving-cost estimation 都加了 unit tests。
- 在花较多 API/Colab 时间前，先跑 tiny runs：`n_eval=3`、`TRAIN_N=3/100`。
- Aggregation 后会检查 raw examples，尤其是 fine-tuning hurt 的 case，以及 Self-Consistency 能修复的 case。
- 对 caveats 不做隐藏：50-row eval size、cost-estimation assumptions、tokenizer fallback、raw JSONL 没有提交等都写进文档。

## 给团队的建议

1. 用 AI 生成 hypotheses 和 implementation options，但每个 change 都要有明确 validation gate。
2. AI-assisted work 应该小步、可提交；小 commits 更容易定位 AI 建议是帮忙了还是引入了 drift。
3. Smoke tests 要覆盖真实 production/integration path。Optional dependency stack 下，import-only checks 通常不够。
4. 可以让 AI 解释 trade-offs，但要用源码、文档和实际行为验证它的判断。
5. 把 AI 出错当成信号：这些错误经常暴露 documentation gap、隐含假设或 API boundary 不清。
6. 保留 audit trail：命令、run metadata、aggregate outputs，以及每轮 iteration 发生了什么变化。
