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
