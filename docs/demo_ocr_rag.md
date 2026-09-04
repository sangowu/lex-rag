# OCR → RAG 端到端：4 份扫描合同走完全链路

> Run: `20260904T181646Z`（clean）/ `20260904T182610Z`（OCR）。
> 4 份 CUAD 合同 / 32 页 / 164 条问题（50 条有答案）。表：`chunks` vs `chunks_ocr`。
>
> ⚠️ **第一版只做了 1 份合同（CENTRACK），结论和机制都不对，见第 6 节。**
> 这一版是那次的复制实验。

`scripts/ingest_ocr.py` 一直存在，但从没端到端跑过一次。这篇把它跑通，
并且回答：**OCR 误差吃掉多少端到端质量。**

## 结论先写

| | 干净文本 | **OCR 文本** | McNemar |
|---|:---:|:---:|:---:|
| semantic_hit_rate | 0.820 | **0.860** | p=0.500（净 +2） |
| └ 仅 cosine | 0.680 | 0.680 | p=1.000（净 0） |
| false_positive_rate | 0.096 | **0.053** | p=0.125（净 +5） |
| false_negative_rate | 0.140 | **0.120** | p=1.000（净 +1） |

**逐条配对下 OCR 一条回归都没造成**（`只有A好 = 0`）。方向甚至微微偏向 OCR，
但四项全部落在噪声里，**别把 +0.040 当成收益**。

单变量检查确认两臂只差 `table` 一个字段。

而同一批文档的 OCR 字符错误率是 **0.8% ~ 19.5%**。下面解释为什么这两件事能同时成立。

---

## 1. 语料：为什么是合成的扫描件

OmniDocBench 的九类里**没有合同**（academic_literature / research_report / book /
PPT2PDF / colorful_textbook / magazine / exam_paper / newspaper / note）。拿一张论文
扫描件去演法务 RAG，跑通了也说明不了什么。而真实扫描合同拿不到 ground truth——
没有 ground truth 就只能截图说"你看它跑通了"。

所以从 CUAD 原文**反向渲染**：等宽字体（SEC 的 EDGAR 本来就是等宽），加轻微歪斜、
焦外、噪点、对比度损失。四份合同刻意覆盖短→长，且都选单-gold 问题占比高的
（单 gold = 没有第二条 gold 兜底）。

```bash
uv run scripts/make_scanned_demo.py     # 4 份 / 32 页 -> data/scanned_docs/*.pdf
```

⚠️ **这里的 CER 不代表真实扫描件。** 装订阴影、透印、折痕、非均匀光照、二次复印，
一个都没有。**可比的是"同一份文本走 OCR 和不走 OCR 的差"，不是 CER 的绝对值。**

⚠️ **每份合同用自己的 rng**（`seed + crc32(doc_id)`）。共用一个的话，往列表里插一份
就会改变后面所有合同的噪声，已发表的数字全部作废。代价是 CENTRACK 的图与第一版
不同，所以它的 CER 从 0.1012 变成 0.1029——**同一份合同、同一档退化，换个噪声实例
就差 0.0017，这也是个有用的刻度**。

## 2. 跑法

```bash
uv run scripts/ingest_ocr.py --input-dir data/scanned_docs --table chunks_ocr
uv run scripts/eval_generation.py --qa data/qa_cuad_ocrdemo.jsonl --limit 0 --reranker --sim-threshold 0.70 --table chunks
uv run scripts/eval_generation.py --qa data/qa_cuad_ocrdemo.jsonl --limit 0 --reranker --sim-threshold 0.70 --table chunks_ocr
uv run scripts/eval_generation.py --compare data/runs/gen_eval/A.json data/runs/gen_eval/B.json
```

32 页 OCR 墙钟 **4 分 09 秒**（含上传、轮询、下载 zip）。

**输出必须是单个多页 PDF，不能是逐页 PNG**：`ingest_ocr.py` 用 `path.stem` 当 `doc_id`，
拆成 N 张图会把一份合同变成 N 个文档，按 `doc_id` 检索就对不上 CUAD 的问答对。

## 3. OCR 错在哪

| 合同 | 页 | CER | WER | 文本长度变化 |
|---|:--:|:--:|:--:|:--:|
| SIBANNAC 战略联盟 | 3 | **0.0083** | 0.0429 | −0.7% |
| CENTRACK 网站托管 | 6 | 0.1029 | 0.1518 | −10.0% |
| FTENETWORKS 战略联盟 | 12 | 0.1140 | 0.1364 | −11.3% |
| ADAMSGOLF 代言 | 11 | **0.1953** | 0.2573 | −16.1% |

**CER 与文本缩短量几乎同步**——误差以**删除**为主，不是认错字。同一档模拟退化下
跨合同差 23 倍（0.008 vs 0.195），排版密度和条款结构的影响远大于噪声本身。

被删掉的形态高度一致，几乎都是**标题相邻的那一句**：

```
control. IN WITNESS WHEREOF, the parties have executed this Agreement …
fees. INDEMNIFICATION The Customer agrees to indemnify and hold harmless i-on …
TERMINATION The term of this Agreement for the Hosted Site shall commence upon April 1, 1999 …
```

版面分析把标题切出来时连带吃掉了相邻正文。

## 4. gold span 存活：三把尺子，答案在中间

**左列是对照**——gold 抽自原文本就该全在；它不满分说明归一化在骗自己。

| | 原始 txt | OCR 文本 |
|---|:---:|:---:|
| gold span 逐字存在 | **87 / 87** ✅ | **52 / 87** |

35 条 gold 失去了逐字形态。但"逐字没了"离"内容没了"还很远，一个错字就判死刑。
往下追了三层：

| 判据 | 结果 | 问题 |
|---|---|---|
| 逐字子串 | 35 条失去 | 太严：一个 OCR 错字就判消失 |
| `quote_overlap`（最长**连续**公共 token 段 ≥0.70） | 其中 23 条仍可识别，12 条不可 | 仍偏严：散布的小错会把 200 词的正确条款打成 0.3 |
| 词级覆盖率 ≥0.90 | 12 条**全部**通过 | 太松：`CENTRACK INTERNATIONAL INC` 的词全文到处都是 |
| **滑窗最小归一化编辑距离** | **12 条全部能定位，距离 0.03 ~ 0.39** | 决定性 |

**结论：87 条 gold span 里，真正在 OCR 文档中不存在的是 0 条。**
8 条是"在场但被改花"（距离 ≤0.25），4 条是"部分丢失"（0.25~0.45，通常是被标题吃掉的
开头那一句）。

> ⚠️ 前两把尺子一严一松，各自都会给出错误结论：只看 `quote_overlap` 会说"12 条没了"，
> 只看词覆盖率会说"一条都没丢"。**两把尺子夹出来的区间才是答案。**

## 5. 所以为什么端到端没掉

50 条有答案的问题里，**19 条的 gold 全部失去逐字形态**。其中：

- **14 条仍然答对**
- **5 条没答对——而这 5 条在干净文本上本来就没答对**（`clean_hit=False`）

逐条配对（McNemar）的 `只有A好 = 0`：**没有任何一条是被 OCR 弄坏的。**

原因不神秘：**那段文字还在，只是拼错了几个字符**。reranker 和生成层都容得下这个程度的
噪声，而生成层引用时引的是 OCR 版本，语义相似度照样过阈值。

## 6. ⚠️ 这一版推翻了上一版的机制解释

第一版（1 份合同，PR #48）写的是：

> "**救回它们的是合同文本自身的冗余**（同一事实在序言 / 定义 / 条款 / 签名块里重复
> 好几遍），不是 OCR 质量。所以只出现一次的事实被吃掉就是不可恢复且无声的。"

**这个机制是错的，至少是未经证明的。** 复制实验里 12 条"消失"的 gold **全部**能在
OCR 文档里定位到——文字根本没被删掉，谈不上"靠别处的重复救回来"。真实机制简单得多：
**条款还在原地，只是被改花了。**

第一版另外两处也要修正：

| 第一版的说法 | 实测（4 份合同） |
|---|---|
| "整个签名块没了" | 签名块确实被删，但那条 gold 仍能在文档里定位到（dist=0.18） |
| 干净基线 semantic_hit **1.000** | 那是 CENTRACK 一份合同的数；4 份是 **0.820**——**那份合同异常好答** |

> **n=1 的教训**：第一版的每个数字都是对的，错的是从一份合同推出的**因果解释**和
> **基线**。1.000 那个数让整篇文章读起来像"这系统满分"，而它只是碰上了一份容易的合同。

## 7. 检索层评测跨不过 OCR 这道边界

`scripts/eval.py` 的 gold 判据是 `chunk.start / chunk.end`——**原始文档的字符偏移**。
OCR 是另一份文本，偏移完全对不上，hit@k / mrr@k 在 `chunks_ocr` 上算出来是垃圾。

**能跨过去的只有生成层**，因为它的判据是内容（逐字包含 或 余弦）而不是位置。
不是 bug，是两种判据的固有区别，但它意味着 **OCR 链路上没有检索层指标可用**。

## 8. 还没测的

- **只测了"OCR 后仍能答对"，没测"OCR 后编造"**。FP 从 0.096 降到 0.053，方向是好的，
  但 p=0.125，样本不够。
- **退化只有一档**，而实测 CER 在同一档下跨合同差 23 倍。真正决定 CER 的是版面，
  不是我加的噪声——所以"更脏的扫描件会怎样"这个问题这套语料答不了。
- **没有单份合同上答案只出现一次的构造样本**。第一版担心的那个失败模式（唯一事实被
  删掉且无声）在这 4 份里一次都没触发，所以它**仍然只是一个未被观察到的假设**，
  既没被证实也没被证伪。

## 9. 复现

```bash
docker compose up -d db
uv run scripts/make_scanned_demo.py
uv run scripts/ingest_ocr.py --input-dir data/scanned_docs --table chunks_ocr
python -c "import json;docs={'SIBANNAC,INC_12_04_2017-EX-2.1-Strategic Alliance Agreement','CENTRACKINTERNATIONALINC_10_29_1999-EX-10.3-WEB SITE HOSTING AGREEMENT','ADAMSGOLFINC_03_21_2005-EX-10.17-ENDORSEMENT AGREEMENT','FTENETWORKS,INC_02_18_2016-EX-99.4-STRATEGIC ALLIANCE AGREEMENT'};[print(l,end='') for l in open('data/qa_cuad.jsonl',encoding='utf-8') if json.loads(l)['doc_id'] in docs]" > data/qa_cuad_ocrdemo.jsonl
```

需要 `.env` 里的 `MINERU_API_TOKEN`。MinerU 免费额度每账号每天 1000 页最高优先级，
32 页无压力。

⚠️ 换 `--seed` 等于换了一份语料，上面所有数字都要重测。

> **本机踩到的坑**：某些网络下 `mineru.net` 的 A/AAAA 记录会间歇性解析失败
> （NS/SOA/TXT/MX 都正常——它要跟一次 CNAME 到阿里全球加速），公共 DNS 则正常。
> 症状是 `getaddrinfo failed`。换用 `223.5.5.5` / `8.8.8.8` 即可，与本仓库代码无关。
