# Compute-Matched Pareto Frontier（计算-精度的帕累托前沿）

> **状态：进行中。** 这是为 Red Hat AI Innovation Team 准备的 take-home 原型。标记 _TBD_ 的段落会随实验进度填充。

**English version: [README.md](./README.md)**

---

## TL;DR

当你给一个 LLM 系统固定 compute 预算时，钱应该花在**fine-tune 一个小模型**上，还是花在 **inference-time scaling** 上？答案取决于**你预计要服务多少 queries** — 训练是一次性投入，inference 随用量增长。两者之间存在一个 break-even point。

本项目用 **Qwen2.5-1.5B-Instruct on GSM8K** 实证地找出这个点，集成 Red Hat 三个 OSS 库：

- **[sdg_hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub)** — 用强 teacher 模型合成训练数据
- **[training_hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub)** — 对小 student 模型做 LoRA 微调
- **[its_hub](https://github.com/Red-Hat-AI-Innovation-Team/its_hub)** — inference 时跑 Self-Consistency 和 Best-of-N

## 核心结论

_TBD — 实验完成后填充。_

(USD 成本, accuracy) 空间下的 Pareto frontier，多条曲线对应 query 量 N ∈ {1K, 10K, 100K, 1M}，并标出 break-even 点 N\*。

## 章节目录

- [问题 & 方法](#问题--方法) _(TBD)_
- [如何运行](#如何运行) _(TBD)_
- [架构](#架构) _(TBD)_
- [成本核算方法](#成本核算方法) _(TBD)_
- [实验结果](#实验结果) _(TBD)_
- [设计决策 & Scope 边界](#设计决策--scope-边界) _(TBD)_
- [什么有效 / 什么没效](#什么有效--什么没效) _(TBD)_
- [其它工具与原因](#其它工具与原因) _(TBD)_
- [更多时间会改进什么](#更多时间会改进什么) _(TBD)_
- [AI 辅助开发](./AI_ASSISTED_DEV.zh.md)

---

## 问题 & 方法

_TBD_

## 如何运行

_TBD — 当前只有 `scripts/verify_setup.sh` 这个 smoke-test 入口。_

## 架构

_TBD_

## 成本核算方法

_TBD — 所有价格假设见 `prices.yaml`。_

## 实验结果

_TBD_

## 设计决策 & Scope 边界

_TBD_

## 什么有效 / 什么没效

_TBD_

## 其它工具与原因

_TBD_

## 更多时间会改进什么

_TBD_
