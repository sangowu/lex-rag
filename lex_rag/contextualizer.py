"""
Contextual RAG — 在 ingest 阶段为每个 chunk 生成上下文前缀。

提供三种实现：
- ContextualClient：原版，每个 chunk 独立调用 Gemini（最高质量）
- HierarchicalContextualizer：先切 section，每 section 一次调用，~10x 减少 API 调用
- MetadataExtractor：每文档一次调用，提取结构化 metadata（合同类型/当事方/条款）
- HyDEClient：查询阶段，将问题转换为假设合同条款再 embed（HyDE 技术）

缓存：结果分别存入 .cache/{contextual,hierarchical,meta_extract,hyde}.json。
"""
import hashlib
import json
import time
from pathlib import Path

from lex_rag.chunking import ChunkWindow
from lex_rag.config import ContextualConfig

_PROMPT = """\
<document>
{doc_text}
</document>

Here is a chunk from the above legal contract:
<chunk>
{chunk_text}
</chunk>

In 1-2 sentences, describe where this chunk fits within the contract and what legal topic or obligation it covers. \
This context will be prepended to the chunk to improve search retrieval. \
Reply with only the context sentences, nothing else."""

_DEFAULT_CACHE = Path(".cache/contextual.json")


class ContextualClient:
    def __init__(self, cfg: ContextualConfig, cache_path: Path = _DEFAULT_CACHE):
        from google import genai  # 懒加载：仅在 contextual 开启时才引入，避免污染其他脚本的日志
        self._client = genai.Client(api_key=cfg.api_key)
        self.cfg = cfg
        self.cache_path = cache_path
        self._cache: dict[str, str] = self._load_cache()
        self._min_interval = 60.0 / cfg.rpm_limit  # seconds between calls

    def _load_cache(self) -> dict[str, str]:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8")

    def _cache_key(self, chunk: ChunkWindow) -> str:
        h = hashlib.md5(chunk.text.encode()).hexdigest()[:8]
        return f"{chunk.chunk_id}:{h}"

    def _call_gemini(self, prompt: str) -> str:
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.cfg.model,
                    contents=prompt,
                )
                return resp.text.strip()
            except Exception as exc:
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_sec * (2 ** attempt))
                else:
                    raise RuntimeError(f"Gemini API failed after {attempt + 1} attempts") from exc

    def contextualize(self, doc_text: str, chunks: list[ChunkWindow]) -> list[ChunkWindow]:
        """为一个文档的所有 chunk 生成上下文前缀，返回新的 ChunkWindow 列表。"""
        result: list[ChunkWindow] = []
        dirty = False
        last_call_time = 0.0

        for chunk in chunks:
            key = self._cache_key(chunk)

            if key in self._cache:
                context = self._cache[key]
            else:
                # 限速：两次 API 调用之间保持最小间隔
                elapsed = time.monotonic() - last_call_time
                if elapsed < self._min_interval:
                    time.sleep(self._min_interval - elapsed)

                prompt = _PROMPT.format(doc_text=doc_text, chunk_text=chunk.text)
                context = self._call_gemini(prompt)
                last_call_time = time.monotonic()

                self._cache[key] = context
                dirty = True

            result.append(ChunkWindow(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=f"{context}\n\n{chunk.text}",
                start=chunk.start,
                end=chunk.end,
                parent_chunk_id=chunk.parent_chunk_id,
            ))

        if dirty:
            self._save_cache()

        return result


# ---------------------------------------------------------------------------
# HierarchicalContextualizer
# ---------------------------------------------------------------------------

_SECTION_PROMPT = """\
<document>
{doc_text}
</document>

Here is a section from the above legal contract:
<section>
{section_text}
</section>

In 1-2 sentences, summarize the legal topic and key obligations covered in this section.
This summary will be prepended to smaller chunks within this section to improve search retrieval.
Reply with only the summary sentences, nothing else."""

_HIER_CACHE = Path(".cache/hierarchical.json")


class HierarchicalContextualizer:
    """
    分层上下文：先切 section，每 section 调用一次 Gemini，
    普通 chunk 复用所属 section 的摘要作前缀。
    API 调用量约为 ContextualClient 的 1/10。
    接口与 ContextualClient 相同：contextualize(doc_text, chunks) -> list[ChunkWindow]。
    """

    def __init__(self, cfg: ContextualConfig,
                 cache_path: Path = _HIER_CACHE):
        from google import genai
        self._client = genai.Client(api_key=cfg.api_key)
        self.cfg = cfg
        self.cache_path = cache_path
        self._cache: dict[str, str] = self._load_cache()
        self._min_interval = 60.0 / cfg.rpm_limit

    def _load_cache(self) -> dict[str, str]:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _section_cache_key(self, doc_id: str, section_text: str) -> str:
        h = hashlib.md5(section_text.encode()).hexdigest()[:8]
        return f"hier:{doc_id}:{h}"

    def _cut_sections(self, doc_id: str, text: str) -> list[ChunkWindow]:
        sections: list[ChunkWindow] = []
        i, s_idx = 0, 0
        while i < len(text):
            end = min(i + self.cfg.section_chars, len(text))
            sections.append(ChunkWindow(
                chunk_id=f"{doc_id}#sec{s_idx}",
                doc_id=doc_id,
                text=text[i:end],
                start=i,
                end=end,
            ))
            i += self.cfg.section_chars
            s_idx += 1
        return sections

    def _find_section(self, chunk: ChunkWindow, sections: list[ChunkWindow]) -> ChunkWindow:
        for sec in sections:
            if sec.start <= chunk.start < sec.end:
                return sec
        return sections[-1]

    def _call_gemini(self, prompt: str) -> str:
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.cfg.model, contents=prompt,
                )
                return resp.text.strip()
            except Exception as exc:
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_sec * (2 ** attempt))
                else:
                    raise RuntimeError(f"Gemini API failed after {attempt + 1} attempts") from exc

    def contextualize(self, doc_text: str, chunks: list[ChunkWindow]) -> list[ChunkWindow]:
        if not chunks:
            return []
        doc_id = chunks[0].doc_id
        sections = self._cut_sections(doc_id, doc_text)

        section_summaries: dict[str, str] = {}
        dirty = False
        last_call_time = 0.0
        for sec in sections:
            key = self._section_cache_key(doc_id, sec.text)
            if key in self._cache:
                section_summaries[sec.chunk_id] = self._cache[key]
            else:
                elapsed = time.monotonic() - last_call_time
                if elapsed < self._min_interval:
                    time.sleep(self._min_interval - elapsed)
                prompt = _SECTION_PROMPT.format(doc_text=doc_text, section_text=sec.text)
                summary = self._call_gemini(prompt)
                last_call_time = time.monotonic()
                self._cache[key] = summary
                section_summaries[sec.chunk_id] = summary
                dirty = True

        if dirty:
            self._save_cache()

        result: list[ChunkWindow] = []
        for chunk in chunks:
            sec = self._find_section(chunk, sections)
            summary = section_summaries.get(sec.chunk_id, "")
            result.append(ChunkWindow(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=f"{summary}\n\n{chunk.text}" if summary else chunk.text,
                start=chunk.start,
                end=chunk.end,
                parent_chunk_id=chunk.parent_chunk_id,
            ))
        return result


# ---------------------------------------------------------------------------
# MetadataExtractor
# ---------------------------------------------------------------------------

_META_PROMPT = """\
You are a legal contract analyst. Analyze the following contract excerpt and extract structured metadata.

<contract>
{doc_preview}
</contract>

Extract the following fields. If a field cannot be determined, use null.
Return ONLY valid JSON with no explanation or markdown:

{{
  "contract_type": "<e.g. Software License Agreement, Service Agreement, NDA, etc.>",
  "party_a": "<first party name>",
  "party_b": "<second party name>",
  "effective_date": "<date string or null>",
  "governing_law": "<jurisdiction, e.g. Delaware or New York>",
  "key_clauses": ["<clause type 1>", "<clause type 2>"]
}}

Key clause types to look for (include all that apply):
Limitation of Liability, Indemnification, Non-Compete, Non-Solicitation,
Termination, IP Ownership, Confidentiality, Payment Terms, Warranty, Arbitration"""

_META_CACHE = Path(".cache/meta_extract.json")


class MetadataExtractor:
    """
    每文档调用 Gemini 一次，提取结构化 metadata，存入 doc_meta 表。
    JSON 解析失败时降级为空 meta（不抛异常，不影响主 ingest 流程）。
    """

    def __init__(self, cfg: ContextualConfig, cache_path: Path = _META_CACHE):
        from google import genai
        self._client = genai.Client(api_key=cfg.api_key)
        self.cfg = cfg
        self.cache_path = cache_path
        self._cache: dict[str, dict] = self._load_cache()
        self._min_interval = 60.0 / cfg.rpm_limit
        self._last_call_time = 0.0

    def _load_cache(self) -> dict[str, dict]:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _doc_cache_key(self, doc_id: str, text: str) -> str:
        h = hashlib.md5(text[:500].encode()).hexdigest()[:8]
        return f"meta:{doc_id}:{h}"

    def _empty_meta(self) -> dict:
        return {
            "contract_type": None, "party_a": None, "party_b": None,
            "effective_date": None, "governing_law": None, "key_clauses": [],
        }

    def extract(self, doc_id: str, doc_text: str) -> dict:
        key = self._doc_cache_key(doc_id, doc_text)
        if key in self._cache:
            return self._cache[key]

        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        preview = doc_text[:2000]
        prompt = _META_PROMPT.format(doc_preview=preview)

        meta = self._empty_meta()
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.cfg.model, contents=prompt,
                )
                raw = resp.text.strip()
                self._last_call_time = time.monotonic()
                # 去除可能的 markdown 代码块包装
                if raw.startswith("```"):
                    lines = raw.split("\n")
                    raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
                meta = json.loads(raw)
                break
            except json.JSONDecodeError:
                meta = self._empty_meta()
                break
            except Exception:
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_sec * (2 ** attempt))
                else:
                    meta = self._empty_meta()

        self._cache[key] = meta
        self._save_cache()
        return meta


# ---------------------------------------------------------------------------
# HyDEClient
# ---------------------------------------------------------------------------

_HYDE_PROMPT = """\
Write a short excerpt (2-4 sentences) from a legal contract that would directly answer the following question.
Write only the contract clause text itself, in formal legal language. No explanation, no preamble.

Question: {question}"""

_HYDE_CACHE = Path(".cache/hyde.json")


class HyDEClient:
    """
    查询阶段：将自然语言问题转换为假设合同条款文本，再交由 embedding 模型处理。
    结果缓存在 .cache/hyde.json，相同问题不重复调用 Gemini。
    """

    def __init__(self, cfg: ContextualConfig, cache_path: Path = _HYDE_CACHE):
        from google import genai
        self._client = genai.Client(api_key=cfg.api_key)
        self.cfg = cfg
        self.cache_path = cache_path
        self._cache: dict[str, str] = self._load_cache()
        self._min_interval = 60.0 / cfg.rpm_limit
        self._last_call_time = 0.0

    def _load_cache(self) -> dict[str, str]:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _cache_key(self, question: str) -> str:
        return hashlib.md5(question.encode()).hexdigest()

    def generate(self, question: str) -> str:
        """返回假设合同条款文本；命中缓存时直接返回，不调用 Gemini。"""
        key = self._cache_key(question)
        if key in self._cache:
            return self._cache[key]

        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        prompt = _HYDE_PROMPT.format(question=question)
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.cfg.model, contents=prompt,
                )
                hypo = resp.text.strip()
                self._last_call_time = time.monotonic()
                break
            except Exception:
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_sec * (2 ** attempt))
                else:
                    # 降级：直接返回原始问题，不影响检索流程
                    hypo = question

        self._cache[key] = hypo
        self._save_cache()
        return hypo


# ---------------------------------------------------------------------------
# QueryExpander
# ---------------------------------------------------------------------------

_EXPAND_PROMPT = """\
Generate {n} alternative phrasings of the following legal contract question.
Each rephrasing should approach the same information need from a different angle \
(e.g. different vocabulary, clause-focused, obligation-focused).
Return ONLY a JSON array of strings with exactly {n} items, no explanation.

Question: {question}"""

_EXPAND_CACHE = Path(".cache/query_expand.json")


class QueryExpander:
    """
    查询阶段：将一个问题改写为 N 个变体，用于 Multi-Query 检索。
    结果缓存在 .cache/query_expand.json，相同问题不重复调用 Gemini。
    """

    def __init__(self, cfg: ContextualConfig, n: int = 3,
                 cache_path: Path = _EXPAND_CACHE):
        from google import genai
        self._client = genai.Client(api_key=cfg.api_key)
        self.cfg = cfg
        self.n = n
        self.cache_path = cache_path
        self._cache: dict[str, list[str]] = self._load_cache()
        self._min_interval = 60.0 / cfg.rpm_limit
        self._last_call_time = 0.0

    def _load_cache(self) -> dict[str, list[str]]:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _cache_key(self, question: str, n: int) -> str:
        return hashlib.md5(f"{n}:{question}".encode()).hexdigest()

    def expand(self, question: str) -> list[str]:
        """返回 N 个查询变体（含原始问题）；命中缓存时直接返回。"""
        if self.n <= 1:
            return [question]

        key = self._cache_key(question, self.n)
        if key in self._cache:
            return self._cache[key]

        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        prompt = _EXPAND_PROMPT.format(n=self.n - 1, question=question)
        variants = [question]
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.cfg.model, contents=prompt,
                )
                raw = resp.text.strip()
                self._last_call_time = time.monotonic()
                if raw.startswith("```"):
                    lines = raw.split("\n")
                    raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    variants = [question] + [str(v) for v in parsed]
                break
            except json.JSONDecodeError:
                break
            except Exception:
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_sec * (2 ** attempt))

        self._cache[key] = variants
        self._save_cache()
        return variants
