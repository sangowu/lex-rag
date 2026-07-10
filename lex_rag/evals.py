from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from tqdm import tqdm

from lex_rag.config import EvalConfig
from lex_rag.cuad import QAItem, Span 
from lex_rag.pipeline import RAGPipeline
from lex_rag.chunking import ChunkWindow


@dataclass
class EvalResult:
    hit_at_k:       dict[int, float] = field(default_factory=dict)
    mrr_at_k:       dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k:    dict[int, float] = field(default_factory=dict)
    avg_latency_ms: float = 0.0


def _hit(chunks: list[ChunkWindow], spans: list[Span], k: int) -> int:
    for chunk in chunks[:k]:
        for span in spans:
            if chunk.start <= span.start < chunk.end or chunk.start < span.end <= chunk.end:
                return 1
    return 0


def _reciprocal_rank(chunks: list[ChunkWindow], spans: list[Span], k: int) -> float:
    for rank, chunk in enumerate(chunks[:k]):
        for span in spans:
            if chunk.start <= span.start < chunk.end or chunk.start < span.end <= chunk.end:
                return 1.0 / (rank + 1)
    return 0.0

def _precision(chunks: list[ChunkWindow], spans: list[Span], k: int) -> float:
    """前 k 个 chunk 中，命中 span 的 chunk 数 / k"""
    hits = 0
    for chunk in chunks[:k]:
        for span in spans:
            if chunk.start <= span.start < chunk.end or chunk.start < span.end <= chunk.end:
                hits += 1
                break  
    return hits / k if k > 0 else 0.0


def _recall(chunks: list[ChunkWindow], spans: list[Span], k: int) -> float:
    """spans 中，被前 k 个 chunk 覆盖到的比例"""
    if not spans: 
        return 0.0
    covered = 0
    for span in spans:
        for chunk in chunks[:k]:
            if chunk.start <= span.start < chunk.end or chunk.start < span.end <= chunk.end:
                covered += 1
                break 
    return covered / len(spans)


def evaluate(
    pipeline: RAGPipeline,
    qa_items: list[QAItem],
    cfg: EvalConfig,
) -> EvalResult:
    """对所有 QAItem 跑 query，累积指标，返回平均值。"""
    result = EvalResult()
    k_values = cfg.k_values
    max_k = max(k_values)

    # 初始化累加器
    hit_sum  = {k: 0.0 for k in k_values}
    mrr_sum  = {k: 0.0 for k in k_values}
    precision_sum = {k: 0.0 for k in k_values}
    recall_sum    = {k: 0.0 for k in k_values}
    latency_sum   = 0.0
    n_evaluated = 0

    for item in tqdm(qa_items, desc="Evaluating"):
        doc_id = item.doc_id if cfg.scope == "contract" else None
        t0 = time.perf_counter()
        chunks = pipeline.query(item.question, doc_id=doc_id, k=max_k)
        latency_sum += (time.perf_counter() - t0) * 1000
        spans = item.spans
        if not spans:
            continue  
        n_evaluated += 1

        for k in k_values:
            hit_sum[k] += _hit(chunks, spans, k)
            mrr_sum[k] += _reciprocal_rank(chunks, spans, k)
            precision_sum[k] += _precision(chunks, spans, k)
            recall_sum[k]    += _recall(chunks, spans, k)

    n_total = len(qa_items)
    for k in k_values:
        result.hit_at_k[k]       = hit_sum[k]       / n_evaluated if n_evaluated > 0 else 0.0
        result.mrr_at_k[k]       = mrr_sum[k]        / n_evaluated if n_evaluated > 0 else 0.0
        result.precision_at_k[k] = precision_sum[k]  / n_evaluated if n_evaluated > 0 else 0.0
        result.recall_at_k[k]    = recall_sum[k]     / n_evaluated if n_evaluated > 0 else 0.0
    result.avg_latency_ms = latency_sum / n_total if n_total > 0 else 0.0
    return result


def save_result(result: EvalResult, path: Path) -> None:
    """将 EvalResult 序列化为 JSON 保存到 path。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.__dict__, f, indent=2)
    
