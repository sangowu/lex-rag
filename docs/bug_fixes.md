# Bug Fixes

## BM25 全零问题（2026-05-22）

**文件：** `legal_rag_v1/store.py` → `search_bm25()`

### 现象

Grid search 结果中 bm25 模式 hit@k 全部为 0.000，hybrid 与 vector 结果完全相同。

### 根本原因

`plainto_tsquery('english', query)` 生成 **AND 语义**查询，要求所有词同时出现在同一 chunk 中。

CUAD 数据集的问题是固定模板：
```
"Highlight the parts (if any) of this contract related to 'X'
 that should be reviewed by a lawyer. Details: ..."
```

解析后得到：
```sql
'highlight' & 'part' & 'contract' & 'relat' & 'lawyer' & 'review' & ...
```

"highlight"、"lawyer"、"review" 属于提问模板，不会出现在合同正文 chunk 中，AND 查询 0 命中。

### 修复

在 Python 层将 tsquery 的 `&` 替换为 `|`，改为 OR 语义：

```python
# store.py: search_bm25()
cur.execute("SELECT replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')", (query,))
tsq_or = cur.fetchone()[0]
if not tsq_or:
    return []
# 后续用 to_tsquery('english', tsq_or) 执行查询
```

### 修复效果（100 条 QA）

| Mode | hit@1 | hit@5 | mrr@5 |
|------|-------|-------|-------|
| vector | 0.400 | 0.743 | 0.544 |
| bm25 | 0.314 | 0.571 | 0.391 |
| hybrid | **0.457** | **0.800** | **0.605** |

修复前 bm25=0，hybrid 与 vector 完全一致；修复后 hybrid 通过 RRF 融合带来明显增益。

> **注：** `data/runs/grid/20260522T*` 两次历史运行的 bm25/hybrid 结果因此 bug 无效。


---

## 只读查询把连接永久留在 `idle in transaction`（2026-08-28）

### 症状

起着 `serve.py` 的同时另起一个进程跑脚本，脚本**卡住不动，没有任何报错**。
`pg_stat_activity` 一看就清楚：

```
pid 212606  idle in transaction   SELECT contract_type, party_a, ...   ← serve.py
pid 212688  active  Lock/relation ALTER TABLE chunks ADD COLUMN ...    ← 新进程
```

新进程的 `VectorStore.__init__` → `_init_schema()` 要跑 `ALTER TABLE chunks`，
而 serve 的连接攥着 `chunks` 的 ACCESS SHARE 锁不放，于是无限等待。

### 根因

`store.py` 用的是 `psycopg.connect(dsn)`，**默认 `autocommit=False`**：第一条语句
就隐式开一个事务。而 `_cursor()` 只在**出错时** rollback，成功路径既不 commit 也不
rollback：

```python
try:
    with self.conn.cursor() as cur:
        yield cur
except Exception:
    with contextlib.suppress(Exception):
        self.conn.rollback()
    raise
# ← 成功路径什么都不做，事务一直开着
```

所以**只读**查询做完，连接就停在 `idle in transaction`，直到进程退出或下一次出错。
写路径末尾有 `conn.commit()`，所以问题只出在读路径——而读路径正是长驻进程用得最多的。

复现（两行就能看到）：

```
旧写法 connect(dsn) 只读之后 = idle in transaction
```

### 两个后果，第二个更隐蔽

1. **别的进程对该表做 DDL 会无限阻塞，且不报错。** 上面那个症状。
2. **长期 idle in transaction 钉住事务快照，VACUUM 回收不掉死元组。**
   `serve.py` 是长驻进程，跑得越久表膨胀越严重。

⚠️ **这两条都不会被功能测试发现**——查询照样返回正确结果，指标一个都不动。

### 修复

连接改成 `autocommit=True`，写路径按需显式声明事务边界：

| 方法 | 处理 | 理由 |
|---|---|---|
| 所有读路径 | 无事务 | 每条语句自己结束，读完即释放锁 |
| `truncate` / `save_meta` / `add_doc_meta` | 无事务 | 单条语句，autocommit 下自带原子性 |
| `add_chunks` | **`conn.transaction()`** | 循环里一行一条 INSERT，不包就是每行一次 fsync，且中途崩会写一半 |
| `_init_schema` | **故意不包** | DDL 全是 IF NOT EXISTS，逐条提交让锁尽早释放；包成一个事务会拉长锁窗口，而多 worker 并发建表互相死锁正是这里出过的事故 |

⚠️ **事务块内不能调 `conn.rollback()`**：

```
ProgrammingError - Explicit rollback() forbidden within a Transaction context.
```

所以 `add_chunks` 里用 `conn.cursor()` 而不是会自动回滚的 `_cursor()`——
回滚本来就该由 `transaction()` 自己做。

### 验证

对着真库跑（`scripts` 之外的一次性脚本）：

```
① 只读查询之后连接状态 = 'idle'
   get_doc_meta 之后      = 'idle'
② 另一个连接的 ALTER TABLE 在 3s 内拿到锁并完成   （原来无限等，故设 lock_timeout=3s）
③ 写入后读回 3 条，连接状态 = 'idle'
```

回归测试见 `tests/test_store_transactions.py`（5 条）：连接必须 autocommit、
读路径不再 commit、`add_chunks` 必须开事务、单语句写不许开事务、
`_init_schema` 不许把 DDL 包成一个事务。
