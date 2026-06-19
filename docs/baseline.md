# 检索系统优化基线记录

> **当前 Base（后续优化起点）：** Contextual RAG + overlap=100  
> **综合均值最高配置，详见第四阶段。**

---

## 阶段一：Grid Search v1（无 Reranker）

**Run ID：** `20260523T003159Z` | **日期：** 2026-05-23  
**搜索空间：** 36 组合（chunk_chars × overlap × strategy × mode）

### 最优配置

| 参数 | 值 |
|------|----|
| strategy | recursive |
| mode | hybrid |
| chunk_chars | 1000 |
| overlap | 100 |
| reranker | false |

### 指标

| hit@1 | hit@5 | hit@10 | mrr@5 | recall@5 | latency |
|-------|-------|--------|-------|----------|---------|
| 0.374 | 0.676 | 0.769 | 0.477 | 0.605 | 3.8ms |

### 各维度分析

| Mode | hit@5 | mrr@5 | latency |
|------|-------|-------|---------|
| hybrid | **0.608** | **0.447** | 5.4ms |
| bm25 | 0.567 | 0.367 | 4.4ms |
| vector | 0.563 | 0.408 | **0.9ms** |

| chunk_chars | hit@5 | mrr@5 |
|-------------|-------|-------|
| 1000 | **0.600** | **0.419** |
| 800 | 0.574 | 0.403 |
| 600 | 0.563 | 0.400 |

| strategy | hit@5 | mrr@5 |
|----------|-------|-------|
| recursive | **0.601** | **0.417** |
| fixed | 0.558 | 0.398 |

---

## 阶段二：加入 Reranker（bge-reranker-v2-m3，top_k=20）

**Run IDs：** 无 reranker `20260524T010942Z`，有 reranker `20260524T003247Z` | **日期：** 2026-05-24  
**配置：** recursive + hybrid + chunk1000 + overlap=100 + rerank_top_k=20

| 指标 | 无 reranker | +reranker | Delta |
|------|------------|-----------|-------|
| hit@1 | 0.381 | 0.495 | **+11.4pp** |
| hit@5 | 0.626 | 0.758 | **+13.2pp** |
| hit@10 | 0.779 | 0.815 | +3.6pp |
| mrr@5 | 0.464 | 0.600 | **+13.6pp** |
| recall@5 | 0.569 | 0.696 | +12.7pp |
| latency | 5.6ms | 1558ms | +276× |

**结论：** Reranker 大幅提升排序质量，hit@10 受限于候选池，是下一步瓶颈。

---

## 阶段三：扩大 rerank_top_k + overlap 调优（无 Contextual）

**Run ID：** `20260524T102655Z` | **日期：** 2026-05-24  
**搜索空间：** chunk × overlap × rerank_top_k=40（共 4 组）

| 配置 | hit@1 | hit@5 | hit@10 | mrr@5 | recall@5 | latency |
|------|-------|-------|--------|-------|----------|---------|
| chunk1000 + overlap=**150** | 0.509 | **0.819** | **0.865** | 0.629 | **0.756** | 2114ms |
| chunk1200 + overlap=150 | 0.516 | 0.815 | 0.843 | 0.631 | 0.755 | 2119ms |
| chunk1000 + overlap=100 | 0.534 | 0.801 | 0.851 | 0.631 | 0.734 | 2084ms |
| chunk1200 + overlap=100 | **0.548** | 0.786 | 0.847 | **0.637** | 0.723 | 2069ms |

**结论：**
- top_k=40 将 hit@5 从 0.758 提升至 0.819（**+6.1pp**），hit@10 从 0.815 → 0.865（+5pp）
- overlap=150 是 hit@5 与 recall@5 的甜点，overlap=100 在 hit@1 上更优
- chunk_chars=1200 没有带来整体提升，chunk=1000 在 hit@5/recall 维度更优

**阶段三最优（recall 优先）：** chunk1000 + overlap=150，hit@5=0.819，recall@5=0.756

---

## 阶段四：Contextual RAG（Gemini gemini-3.1-flash-lite-preview）

**Run IDs：** `20260524T230351Z`（overlap=100）、`20260525T005706Z`（overlap=150） | **日期：** 2026-05-24~25  
**方法：** ingest 阶段为每个 chunk 调用 Gemini 生成 1-2 句法律上下文描述，拼在 chunk 原文之前，同时提升 embedding 和 BM25 检索质量。

| 配置 | hit@1 | hit@5 | hit@10 | mrr@5 | recall@5 | 综合均值 |
|------|-------|-------|--------|-------|----------|---------|
| 无 Contextual + overlap=150 | 0.509 | **0.819** | **0.865** | 0.629 | **0.756** | 0.716 |
| 无 Contextual + overlap=100 | 0.516 | 0.815 | 0.843 | 0.631 | 0.755 | 0.712 |
| **Contextual + overlap=100** | **0.577** | 0.794 | **0.865** | **0.658** | 0.739 | **0.727** |
| Contextual + overlap=150 | 0.505 | **0.819** | 0.858 | 0.634 | 0.754 | 0.714 |

**结论：**
- Contextual RAG 显著提升 hit@1（+6.8pp）和 mrr@5（+2.9pp），排序质量最优
- hit@10 与无 Contextual 最优持平（0.865）
- 综合均值最高配置：**Contextual + overlap=100（0.727）**
- recall 单项最优：无 Contextual + overlap=150（recall@5=0.756）

---

## 全阶段演进汇总

| 阶段 | 关键变更 | hit@1 | hit@5 | hit@10 | mrr@5 | recall@5 |
|------|---------|-------|-------|--------|-------|----------|
| 阶段一 | Grid Search 基线 | 0.374 | 0.676 | 0.769 | 0.477 | 0.605 |
| 阶段二 | +Reranker top_k=20 | 0.495 | 0.758 | 0.815 | 0.600 | 0.696 |
| 阶段三 | top_k=40 + overlap=150 | 0.509 | 0.819 | 0.865 | 0.629 | 0.756 |
| **阶段四** | **+Contextual RAG** | **0.577** | 0.794 | **0.865** | **0.658** | 0.739 |

**与阶段一基线对比提升：**

| 指标 | 基线 | 当前 Base | 提升 |
|------|------|-----------|------|
| hit@1 | 0.374 | **0.577** | **+54.3%** |
| hit@5 | 0.676 | 0.794 | **+17.5%** |
| hit@10 | 0.769 | **0.865** | **+12.5%** |
| mrr@5 | 0.477 | **0.658** | **+37.9%** |
| recall@5 | 0.605 | 0.739 | **+22.1%** |

---

## 当前 Base 配置（后续优化起点）

> **Run ID：** `20260524T230351Z`

| 参数 | 值 |
|------|----|
| table | chunks_contextual |
| strategy | recursive |
| mode | hybrid |
| chunk_chars | 1000 |
| overlap | 100 |
| reranker | true（BAAI/bge-reranker-v2-m3） |
| rerank_top_k | 40 |
| contextual | true（gemini-3.1-flash-lite-preview） |

---

## 后续优化方向

- **Query Rewrite：** 针对 CUAD 模板问题扩展同义词，提升 BM25 召回
- **overlap 细化：** 在 120~160 之间进一步搜索，寻找 Contextual 模式下的甜点
- **Multi-query：** 多路召回合并，提升边界案例覆盖
- **Reranker 升级：** 尝试 bge-reranker-v2-gemma 或 Qwen2-Reranker
