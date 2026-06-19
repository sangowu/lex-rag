from dataclasses import dataclass
from legal_rag_v1.config import ChunkingConfig
from collections.abc import Iterator
import re

@dataclass
class ChunkWindow:
    chunk_id: str    # 唯一标识，建议格式："{doc_id}#{i}"
    doc_id: str
    text: str
    start: int       # 在原文中的起始字符位置
    end: int         # 在原文中的结束字符位置
    parent_chunk_id: str | None = None  # parent-child 模式：child 指向 parent，parent 为 None

def chunk_fixed(
    doc_id: str,
    text: str,
    chunk_chars: int,
    overlap: int,
) -> Iterator[ChunkWindow]:
    step = chunk_chars - overlap
    i, chunk_index = 0,0
    while i < len(text):
        end = min(i + chunk_chars, len(text))
        chunk_text = text[i:end]
        yield ChunkWindow(chunk_id=f"{doc_id}#{chunk_index}", doc_id=doc_id, text=chunk_text, start=i, end=end)
        i += step
        chunk_index += 1
    
def chunk_recursive(
    doc_id: str,
    text: str,
    chunk_chars: int,
    overlap: int,
) -> Iterator[ChunkWindow]:
    buf = ""
    buf_start = 0
    chunk_index = 0
    for match in re.finditer(r'[^\n]+(?:\n(?!\n)[^\n]*)*', text, re.DOTALL):
        para = match.group()
        para_start = match.start()
        if buf and len(buf) + len(para) > chunk_chars:
            yield ChunkWindow(chunk_id=f"{doc_id}#{chunk_index}", doc_id=doc_id, text=buf, start=buf_start, end=para_start)
            chunk_index += 1
            buf_start = buf_start + len(buf) - overlap
            buf = buf[-overlap:]
        buf += para
    if buf:
        yield ChunkWindow(chunk_id=f"{doc_id}#{chunk_index}", doc_id=doc_id, text=buf, start=buf_start, end=buf_start + len(buf))

def chunk_parent_child(
    doc_id: str,
    text: str,
    parent_chars: int = 1000,
    child_chars: int = 300,
    overlap: int = 50,
) -> tuple[list[ChunkWindow], list[ChunkWindow]]:
    """
    返回 (parents, children)。
    parent 按 parent_chars 无 overlap 切分（避免重叠 span 造成评估歧义）。
    child 在每个 parent 内按 child_chars+overlap 细切，child.parent_chunk_id = parent.chunk_id。
    所有 start/end 均为文档全局字符偏移。
    """
    parents: list[ChunkWindow] = []
    children: list[ChunkWindow] = []
    p_idx = 0
    i = 0
    while i < len(text):
        p_end = min(i + parent_chars, len(text))
        parent = ChunkWindow(
            chunk_id=f"{doc_id}#p{p_idx}",
            doc_id=doc_id,
            text=text[i:p_end],
            start=i,
            end=p_end,
        )
        parents.append(parent)
        c_idx = 0
        j = i
        while j < p_end:
            c_end = min(j + child_chars, p_end)
            children.append(ChunkWindow(
                chunk_id=f"{doc_id}#p{p_idx}c{c_idx}",
                doc_id=doc_id,
                text=text[j:c_end],
                start=j,
                end=c_end,
                parent_chunk_id=parent.chunk_id,
            ))
            if c_end == p_end:
                break
            j += child_chars - overlap
            c_idx += 1
        i += parent_chars
        p_idx += 1
    return parents, children


def chunk_text(
    doc_id: str,
    text: str,
    cfg: ChunkingConfig,
) -> Iterator[ChunkWindow]:
    if cfg.strategy == "fixed":   
        return chunk_fixed(doc_id, text, chunk_chars=cfg.chunk_chars, overlap=cfg.overlap)
    if cfg.strategy == "recursive": 
        return chunk_recursive(doc_id, text, chunk_chars=cfg.chunk_chars, overlap=cfg.overlap)
    else: 
        raise ValueError(f"Unknown chunking strategy: {cfg.strategy}")

if __name__ == "__main__":
    text = "这是第一段。\n这是第二段。\n\n这是第三段。"
    for chunk in chunk_recursive("doc1", text, chunk_chars=10, overlap=2):
        print(chunk)