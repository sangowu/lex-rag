# OCR → RAG 端到端：一份扫描合同走完全链路

> Run: `20260904T152725Z`（clean）/ `20260904T153225Z`（OCR），CENTRACK 网站托管协议，
> CUAD 全部 41 条问题。表：`chunks` vs `chunks_ocr`。

`scripts/ingest_ocr.py` 一直存在，但从没端到端跑过一次——`data/scanned_docs` 是空的。
这篇把它跑通，并且回答那个唯一值得问的问题：**OCR 误差吃掉多少端到端质量。**

## 结论先写

| | 干净文本 | **OCR 文本** |
|---|---|---|
| semantic_hit_rate | 1.000 | **1.000** |
| └ 仅 cosine（旧尺子） | 0.800 | 0.800 |
| false_positive_rate | 0.032 | **0.032** |
| false_negative_rate | 0.000 | **0.000** |
| chunks | 21 | 19 |

**端到端一个点都没掉，而 OCR 的字符错误率是 10.1%。**

这不是因为 OCR 干净，是因为**合同文本冗余**。下面把这两句都拆开。

---

## 1. 语料：为什么是合成的扫描件

OmniDocBench 的九类里**没有合同**（academic_literature / research_report / book /
PPT2PDF / colorful_textbook / magazine / exam_paper / newspaper / note）。拿一张论文
扫描件去演法务 RAG，跑通了也说明不了什么。而真实扫描合同拿不到 ground truth——
没有 ground truth 就只能截图说"你看它跑通了"。

所以从 CUAD 原文**反向渲染**：等宽字体排版（SEC 的 EDGAR 本来就是等宽），加轻微歪斜、
焦外、噪点、对比度损失，输出 6 页 PDF。这样同时拿到逐字 ground truth 和这份合同现成的
41 条问答对。

```bash
uv run scripts/make_scanned_demo.py          # 6 页 -> data/scanned_docs/*.pdf，种子定死
```

⚠️ **这里的 CER 不代表真实扫描件。** 装订阴影、透印、折痕、非均匀光照、二次复印，
这里一个都没有。**可比的是"同一份文本走 OCR 和不走 OCR 的差"，不是 CER 的绝对值。**
退化幅度也刻意压得保守——把噪声开到 OCR 崩掉很容易，但那样得到的数字只说明我把噪声
开大了。

## 2. 跑法

```bash
# 扫描件 → MinerU 在线 API → Markdown → chunk/embed → pgvector
uv run scripts/ingest_ocr.py --input-dir data/scanned_docs --table chunks_ocr

# 只取这份合同的 41 条问题
python -c "import json;[print(l,end='') for l in open('data/qa_cuad.jsonl',encoding='utf-8') if json.loads(l)['doc_id'].startswith('CENTRACK')]" > data/qa_cuad_centrack.jsonl

# 两臂
uv run scripts/eval_generation.py --qa data/qa_cuad_centrack.jsonl --limit 0 --reranker --sim-threshold 0.70 --table chunks
uv run scripts/eval_generation.py --qa data/qa_cuad_centrack.jsonl --limit 0 --reranker --sim-threshold 0.70 --table chunks_ocr
```

6 页 OCR 墙钟 **117 秒**（含上传、轮询、下载 zip）。

**输出必须是单个多页 PDF，不能是逐页 PNG**：`ingest_ocr.py` 用 `path.stem` 当 `doc_id`，
拆成 6 张图会把一份合同变成 6 个文档，按 `doc_id` 检索就对不上 CUAD 的问答对了。

## 3. OCR 到底错在哪：不是认错字，是**整段丢失**

| | 值 |
|---|---|
| CER | **0.1012** |
| WER | 0.1452 |
| 原文字符（归一化） | 14388 |
| OCR 字符（归一化） | 12975 |

**OCR 输出比原文短 1413 字符**——误差主要是**删除**，不是识别错。被删掉的最长几段：

```
[21 词] control. IN WITNESS WHEREOF, the parties have executed this Agreement
        as of the date first set forth above. CENTRACK INTERNATIONAL, INC.
[20 词] writing. In the event that any provision hereof is found invalid or …
[18 词] TERMINATION The term of this Agreement for the Hosted Site shall commence upon April 1, 1999 …
[18 词] fees. INDEMNIFICATION The Customer agrees to indemnify and hold harmless i-on …
```

删除总计 **239 词（约 7%）**。注意形态：丢的几乎都是**标题前后那一句**
（`control. IN WITNESS WHEREOF`、`fees. INDEMNIFICATION`、`TERMINATION The term…`）。
版面分析把标题切出来时连带吃掉了相邻正文——**整个签名块就这么没了**。

## 4. 关键的那张表：gold span 的存活

逐字比对每条 CUAD gold answer 在两份文本里的存在性。**左边一列是对照**——
gold 是从原文抽出来的，本来就该全在；它不满分就说明我的归一化在骗自己。

| | 原始 txt | OCR 文本 |
|---|:---:|:---:|
| gold span 逐字存在 | **17 / 17** ✅ | **13 / 17** |

**4 条 gold 确因 OCR 消失**，波及 3 个问题（Effective Date / Expiration Date 共用同一段
gold，Cap On Liability、Parties 各一）：

- `Parties` —— 签名块的 `CENTRACK INTERNATIONAL, INC.` 整块没了
- `Cap On Liability` —— 两句话的 gold 掉了第一句（"lost profits or other consequential damages"）
- `Effective Date` / `Expiration Date` —— 期限条款开头那句被标题吃掉

**而这 3 个问题全部答对了。** 为什么：

| 问题 | 靠什么答对 |
|---|---|
| Parties | 序言里还有一份 `Centrack International, a Florida corporation` |
| Cap On Liability | gold 的第二句（`limited to one (1) month's fees`）完好，模型引的是它 |
| Expiration Date | 条款主体幸存，只丢了开头 |

**救回它们的是合同文本自身的冗余，不是 OCR 质量。** 合同会把同一个事实在序言、
定义、条款、签名块里重复好几遍，OCR 打掉一处，别处还在。

⚠️ **所以"零损失"不能外推。** 一个只出现一次的事实——签名块里的公司全称、某个唯一的
日期、一处金额——被 OCR 吃掉就是**不可恢复且无声**的：检索不会报错，生成层会拿着
剩下的上下文照常作答。这份合同恰好没有把答案押在那种单点上。**换一份合同，
同样的 10% CER 可能就直接变成答错。**

## 5. 检索层评测跨不过 OCR 这道边界

`scripts/eval.py` 的 gold 判据是 `chunk.start / chunk.end`——**原始文档的字符偏移**。
OCR 文本是另一份文本，偏移完全对不上，hit@k / mrr@k 在 `chunks_ocr` 上算出来是垃圾。

**能跨过去的只有生成层**，因为它的判据是内容（逐字包含 或 余弦）而不是位置。
这不是 bug，是两种判据的固有区别，但它意味着：**OCR 链路上没有检索层指标可用**，
只能看端到端。

## 6. 过程中抓到的 bug：`--table` 从来没生效过

第一次跑完，两臂指标完全相同——连 `avg_tokens` 都一模一样（in 5296 / 5296，
total 218,333 / 218,245）。

**`eval_generation.py` 的 `--table` 只被声明、从没被读过。** 参数在，应用它的那一行
不在，于是 `--table chunks_ocr` 被静默忽略，OCR 那一臂其实跑在 `chunks` 上。评测照样
跑完、照样出一份漂亮结果。

抓住它的唯一线索是 **provenance 里老实写着 `table: chunks`**，以及两臂 token 数一个
字节都不差。

> **这是本仓库第五次栽在同一个形态上**：配置变了、读它的人没跟着变、而且完全无声。
> 前四次是 `serve.py` 写死 `top_k`、`reranker.enabled`、`eval.py` 写死 `hit@k`、
> 门禁自带的第二份归一化。**五次全部是靠人眼发现的。**

修法是把覆盖逻辑抽成纯函数 `apply_cli_overrides(cfg, args)` 并由
`tests/test_eval_generation_args.py` 钉住，其中一条是反向的：*凡是它认得的开关，
传了值就必须改动 cfg*——新加一个只声明不读的开关会在那里露出来。

⚠️ **判据不能是"跑起来不报错"**，坏掉的那版也不报错。

> 讽刺的是修完之后两臂指标**依然完全相同**（1.000 / 0.032 / 0.000）。
> 区别是这一次它是真的。

## 7. 复现

```bash
docker compose up -d db
uv run scripts/make_scanned_demo.py
uv run scripts/ingest_ocr.py --input-dir data/scanned_docs --table chunks_ocr
uv run scripts/eval_generation.py --qa data/qa_cuad_centrack.jsonl --limit 0 --reranker --sim-threshold 0.70 --table chunks_ocr
```

需要 `.env` 里的 `MINERU_API_TOKEN`。MinerU 免费额度每账号每天 1000 页最高优先级，
6 页的 demo 无压力。

⚠️ 换 `--seed` 等于换了一份语料，上面所有数字都要重测。
