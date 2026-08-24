# Agentic 检索循环改造规格

> **状态**：待实施。建议分支 `feature/agentic-retrieval-loop`。
> **本文档自包含**，实施时不需要读任何外部文档。

---

## 0. 为什么要做这次改造

当前 lex_rag 是一条**固定流程**：检索 → RRF 融合 → 重排 → 生成 → 拒答门。
五个高级检索技术（Contextual RAG、HyDE、multi-query、parent-child chunking、
agentic query rewriting）都实现了，但它们是**配置文件里写死的静态开关**。

这带来两个问题：

1. **策略选择与问题无关。** 法律合同里既有"违约金是多少"这类精确术语查询
   （BM25 更强），也有"这份合同对乙方有哪些限制"这类概念性查询（向量更强）。
   静态配置只能取一个折中值，对两类问题都不是最优。
2. **失败无法归因。** 流程固定意味着出错时只有 5 个候选步骤，
   而且没有任何记录能说明"系统当时为什么这么做"。

**改造目标**：加一层**运行时决策**——让系统根据问题和上一轮的检索结果，
自己决定下一轮用哪个策略、还要不要再来一轮。

**同时必须保住的东西**：CUAD 标注是这个项目最核心的资产（免费的 ground truth）。
本次改造**不加任何数据集**，标签体系完整保留。

> 一句话：让它变**深**（多一层决策），不要让它变**宽**（多几个数据源）。

---

## 1. 现状诊断（已核实）

### 1.1 `lex_rag/agent.py` 的循环实际上是死代码

```python
chunks = self.pipeline.query(current_query, doc_id=doc_id, k=k)
...
if chunks:
    yield f"✅ 找到 {len(all_chunks)} 个相关片段，开始生成..."
    break
```

重试的触发条件是 **`chunks` 为空列表**。而 `_query_impl` 走 hybrid 检索
（向量 + BM25 融合），**几乎永远返回非空结果**，所以第二轮基本不会发生。

真正的问题从来不是"没检索到"，而是"**检索到了但不对 / 不全**"。

同时动作空间只有一个：`_rewrite_query`。检索失败的原因有很多种，
重写查询只能救其中一种。

### 1.2 五个策略被焊死在 `__init__`

`lex_rag/pipeline.py` 的 `RAGPipeline.__init__`：

```python
self.reranker        = RerankClient(cfg.reranker) if cfg.reranker.enabled else None
self.contextualizer  = ...                        if cfg.contextual.enabled else None
self._hyde           = HyDEClient(cfg.contextual)  if cfg.hyde_enabled else None
self._expander       = QueryExpander(cfg.contextual, n=cfg.multi_query_n) \
                                                   if cfg.multi_query_enabled else None
```

全部是**构造时**从 config 决定。`_query_impl` 内部也是直接读 `self.cfg.retrieval.mode`。
运行时改不了，所以没有任何"决策"可言。

---

## 2. 六个任务

### 2.1 `RetrievalStrategy` 对象 + `_query_impl` 重构（约 2 天）

新建 `lex_rag/strategy.py`：

```python
@dataclass(frozen=True)
class RetrievalStrategy:
    mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    use_hyde: bool = False
    use_multi_query: bool = False
    multi_query_n: int = 3
    fetch_k: int = 50
    top_k: int = 10
    expand_parent: bool = True
    rerank: bool = True
    query_text: str | None = None      # 重写后的查询；None 表示用原问题

    def key(self) -> str:
        """用于防重复：同一策略不许在一次运行里跑两次。"""
```

改造要点：

1. `_query_impl(self, question, doc_id=None, k=None)`
   → `_query_impl(self, question, doc_id=None, strategy: RetrievalStrategy | None = None)`
   `strategy=None` 时从 `self.cfg` 构造默认策略，**保证向后兼容**
2. `__init__` 里那四个 `if cfg.xxx_enabled` 改为**懒加载 property**，
   不再受 config 限制——五个客户端都要能随时被调用，config 只决定默认值
3. 方法体内所有 `self.cfg.xxx` 的读取改为读 `strategy.xxx`

### 2.2 回归验证（约 0.5 天）· **硬门禁**

用默认策略跑全量 CUAD，产出的每一个指标必须与改造前**完全一致**。

- 对照基准：**改造前先跑一遍存下来**（`eval/results_v*.csv` 有 15 个历史版本，
  但无法确定哪个对应当前 HEAD，不要直接拿来当基准）
- 不一致 → 重构引入了 bug，**不许继续往下做**
- 产出：`docs/refactor_regression.md`，贴改造前后对照表

> 跳过这一步的代价：后续所有"失败分析"都可能是在分析你自己重构引入的 bug，
> 而你无法区分。

### 2.3 `sufficiency_judge`（约 1 天）

用 Gemini 的 `tool_choice` 强制结构化输出（不要正则解析 JSON）：

```python
{
  "sufficient":   bool,
  "missing":      str,    # 不够的话缺什么
  "out_of_scope": bool,   # 合同里根本没有 → 走拒答门
  "confidence":   float   # 0-1
}
```

`missing` 是关键字段。它不是给人看的，是**给下一轮的策略选择器看的**：

| `missing` 的内容 | 下一轮该换什么 |
|---|---|
| 缺具体金额 / 条款编号 | 换 BM25（精确匹配） |
| 缺条款的上下文 | 换 parent 粒度 |
| 概念表述不匹配 | 上 HyDE |
| 问题涉及多个方面 | 上 multi-query |

### 2.4 策略选择器 + 防重复（约 1 天）

```
state = {question, doc_id, tried: [(strategy_key, missing)], chunks: []}

for round in range(3):
    strategy = select(question, state)          # LLM 决策，禁止选 tried 里已有的 key
    chunks   = pipeline.query(question, doc_id, strategy)
    state.chunks = dedupe(state.chunks + chunks)
    verdict  = sufficiency_judge(question, state.chunks)

    if verdict.out_of_scope:  terminated_by = "refused";     break
    if verdict.sufficient:    terminated_by = "sufficient";  break
    state.tried.append((strategy.key(), verdict.missing))
else:
    terminated_by = "max_rounds"

generate(state.chunks)
```

- `max_iterations` 从 2 改为 3
- **防重复要在执行层拦截**（同一 `strategy.key()` 直接拒绝并返回
  "already tried, result was: …"），不要只在 prompt 里写"不要重复"——不可靠
- `terminated_by` 必须落盘

### 2.5 trace 埋点扩展（约 1.5 天）· **最容易超时的一步**

`lex_rag/tracing.py` 已有基础。每一轮必须落盘：

| 字段 | 用途 |
|---|---|
| 本轮完整 `strategy` 对象 | 事后判断"策略选错了" |
| 选择器给出的**理由文本** | 唯一能还原"系统当时怎么想"的字段 |
| 返回的 `chunk_id` 列表 + 分数 | 判断召回情况 |
| judge 的 `sufficient` / `missing` / `confidence` | 判断"提前停止 / 白烧轮次" |
| 累积 chunk 数、累积 token 数 | 判断"上下文污染" |
| `terminated_by` | 失败分类的第一层 |

**每一步的 input / output 全文都要落盘**（文件或 SQLite blob 均可）。

> 依据：失败归因研究（Who&When Pro，12,326 条 trace）的实验结论是，
> 完整 trace 相对仅有最终输出，步骤级归因准确率相对提升约 76%。
> 埋点质量直接决定后续一切分析的上限，这一步省不得。

### 2.6 产出 trace 语料（约 1 天）

跑三组配置 × CUAD 全量：

1. 默认配置（基线）
2. 关掉 reranker
3. 强制固定策略（不让系统自己选）

三组之间天然构成对照组，用于后续的差异分析。

---

## 3. 这次改造会制造出的新失败类

制造它们是**目的**，不是副作用——这六类在改造前不存在：

1. **策略选错** —— 法律文本精确术语多，该走 BM25 却选了 vector
2. **judge 假阳性 → 提前停止** —— 上下文其实不够却判定够了，答案必错
3. **judge 假阴性 → 白烧轮次** —— 已经够了却继续循环，成本 ×3、延迟 ×3
4. **策略震荡** —— 在两个策略间来回跳，防重复失效
5. **累积上下文污染** —— 多轮 chunk 累加导致生成被无关片段带偏
6. **拒答时机错误** —— 第 1 轮就该拒答，却烧到第 3 轮

### 其中 2 和 3 有全自动标签

CUAD 有 gold span，因此可以**逐轮**检查"当前累积 chunks 是否已包含 gold span"，
与 judge 的判断对照：

| gold span 在累积 chunks 里吗 | judge 说够了吗 | 结论 |
|---|---|---|
| ✅ 在 | ✅ 够 | 正确停止 |
| ✅ 在 | ❌ 不够 | **假阴性 → 白烧一轮** |
| ❌ 不在 | ✅ 够 | **假阳性 → 提前停止，答案必错** |
| ❌ 不在 | ❌ 不够 | 正确继续 |

**零人工标注成本。** 这是本次改造最有价值的副产品，务必实现
（建议 `scripts/gold_round_check.py`）。

---

## 4. 明确不要做

- ❌ **不加数据集**（会摧毁 CUAD 标签这个核心资产）
- ❌ 不加第 6 种检索技术（要的是决策层，不是技术层）
- ❌ 不做多智能体
- ❌ 不动 CUAD 相关的评估代码路径
- ❌ 不跳过 2.2 的回归验证

---

## 5. 完成标准

- [ ] 2.2 回归验证通过，改造前后对照表已提交
- [ ] 三种终止原因（`sufficient` / `refused` / `max_rounds`）都能稳定复现
- [ ] 每轮 trace 含第 2.5 节表格的全部字段，且每步 input/output 全文可取回
- [ ] `scripts/gold_round_check.py` 可跑，能输出第 3 节那张 2×2 表
- [ ] 三组配置 × CUAD 全量的 trace 语料已落盘

---

## 6. 下游消费者

本次改造产出的 trace 语料，会被一个独立项目
（`D:\Python_Projects\tracelens`，失败归因与回归工具）作为**带标签的验证语料**使用。

这不影响本仓库的实施——上面的规格是自包含的。
但它解释了为什么第 2.5 节的埋点要求比一般项目严格：
下游需要从 trace 里还原出每一步的完整输入输出，才能定位失败发生在哪一步。
