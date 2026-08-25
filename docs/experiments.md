# lex_rag — 生成层实验归档

从 CLAUDE.md 移出的历史迭代记录（v1 / v2 / v3）。当前最优配置 v4 仍保留在 CLAUDE.md。

**生成层 Baseline v1**（Run: `20260527T225006Z`，200样本，30样本 RAGAS）：

| 指标 | 值 |
|------|----|
| semantic_hit_rate | 0.680（threshold=0.70） |
| false_positive_rate | 0.173 |
| false_negative_rate | 0.120 |
| faithfulness | 0.667 |
| answer_relevancy | 0.867 |
| avg_latency_ms | 803 |

配置：Gemini JSON mode + `When in doubt, refuse` + doc_meta 注入 + reranker，top_k=5，generate_k=5

**生成层 v2**（Run: `20260529T171148Z`，200样本，30样本 RAGAS）：

| 指标 | 值 |
|------|----|
| semantic_hit_rate | 0.740 |
| false_positive_rate | 0.200 |
| false_negative_rate | 0.040 |
| faithfulness | 0.500 |
| answer_relevancy | 0.967 |
| avg_latency_ms | 752 |

配置：Gemini JSON mode + `When in doubt, refuse` + few-shot 示例 + reranker，top_k=10，generate_k=8

> doc_meta 注入实验（`20260529T183159Z`）无改善，已废弃。根因：RAGAS judge 不把 doc_meta 计入上下文，模型用 meta 回答被误判为幻觉。

**生成层 v3**（Run: `20260530T110845Z`）：

| 指标 | 值 |
|------|----|
| semantic_hit_rate | 0.760 |
| false_negative_rate | 0.100 |
| faithfulness | 0.667 |
| answer_relevancy | 0.867 |

配置：逐字引用约束 + few-shot + 无 doc_meta，generate_k=8

---

# 迁移到 Z.ai GLM-4.7-Flash 后的生成层实验（2026-08-25）

模型接入层从 Gemini 换到 GLM-4.7-Flash 后，生成层指标全面回落，其中拒答门
（false_positive_rate）从 0.200 塌到 0.731。以下是定位过程与结论。

## 三方对照（50 样本，v4 为 200 样本）

| 指标 | v4 (Gemini) | GLM 迁移后 | + prompt 修复 |
|------|-------------|-----------|--------------|
| false_positive_rate | 0.200 | 0.731 (19/26) | 0.667 (16/24) |
| semantic_hit_rate | 0.820 | 0.737 | 0.667 |
| false_negative_rate | 0.040 | 0.105 | 0.083 |

> faithfulness / answer_relevancy 两项在这两轮中**不可用**：judge 调用大量
> 被 429 打断（最后一轮 30 次里失败 18 次），失败项按中性分 0.5 计入。
> 结果文件里的 `n_judge_failed` 字段是判断这两行能不能用的唯一依据。

## 失败形态：不是幻觉，是"相关但不同的条款"

模型引用的都是合同里**真实存在**的文本：

```
Q: Termination For Convenience
A: No. "Either party may terminate this Agreement upon 30 days prior written
       notice to the other upon the occurrence of any event of default..."
```

它找到一条"因违约终止"条款，答了 No 并引用为证据。CUAD 把这类问题标注为
无答案，评测记为假阳性。

根因是 prompt 自相矛盾：拒答规则说"信息不存在就拒答"，作答格式又说
"Yes/No 问题要以 Yes 或 No 开头并引用条款"。Gemini 选了前者，GLM 选了后者。

## 结论一：改 prompt 无效

把上述矛盾写成明规则（条款缺失→拒答、相关但不同的条款算错误答案、补一条
few-shot 拒答示例）后，FP 从 0.731 降到 0.667——**差值 0.064，而该样本量下
标准误约 0.13，在噪声内**。

## 结论二：thinking 是有效但双向的杠杆

定向 A/B（各 10 条，对照组直接用已有结果，省一半请求）：

| 子集 | thinking=false（对照） | thinking=true |
|------|----------------------|--------------|
| 上一轮判为 FP 的 10 条 | 全部被作答（refused=0） | **6/10 转为正确拒答** |
| 上一轮判为 TP 的 10 条 | 全部正确作答 | **3/10 被误拒** |

`Termination For Convenience` 值得单独一提：它正是写进 few-shot 的那个例子，
光靠示例模型不听，开 thinking 才拒答——这类判断需要的是推理过程，不是更
明确的指令。

**外推到当前分布（各 24 条）：**

| | 现状 | 开 thinking（外推） | v4 |
|---|---|---|---|
| FP 率 | 0.667 | **0.267** | 0.200 |
| FN 率 | 0.083 | **0.358** | 0.040 |

FP 降 0.40，FN 升 0.275。**不是净收益，是精度/召回的旋钮。**

误拒的三条里有 `Document Name` 和 `Expiration Date`——最基础的元数据问题都
拒答，实用性上难以接受。

> 翻转率各基于 10 条样本，误差约 ±15pp，外推值只用于判断量级和方向。

## 下一步的候选方案

1. **两段式**：thinking=false 快速生成，仅当模型给出答案时，再跑一次廉价的
   "该类条款是否真的存在"校验。代价是每个作答问题多一次调用，但不影响拒答
   路径的延迟。这与 `docs/agentic_loop_upgrade.md` 第 2.3 节的
   `sufficiency_judge` 是同一个机制，可以合并实现。
2. **换生成模型**：GLM-4.6 / 4.7 非 Flash 版本。
3. **接受现状**：把 FP 高这一事实写进已知限制，优先推进 agentic 改造。

## 环境约束（影响所有后续实验）

- GLM-4.7-Flash 免费档默认并发 1，持续请求会触发每小时配额，速度从
  1~1.5 秒/条掉到 80~140 秒/条，429 有 1302（配额）与 1305（过载）两种。
- 全量 200 样本 + 30 judge 约需 2.5~3 小时，建议在配额空闲时段跑。
