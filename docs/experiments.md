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
