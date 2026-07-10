from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import re
import requests


@dataclass
class Span:
    start: int
    end:   int
    text:  str


@dataclass
class QAItem:
    id:         str
    question:   str
    doc_id:     str
    answers:    list[str]
    spans:      list[Span]
    has_answer: bool

def _load_cuad_json() -> list:
    # 直接走 HTTP 下载而非 huggingface_hub 的 hf_hub_download：
    # 该数据集用 HF 的 Xet 分块存储后端，hf_hub_download 会走 hf_xet 的下载
    # 协议，在缺少/异常的网络环境下会静默挂起且不抛错、不产生任何输出
    # （曾在 AWS ECS 任务里卡住 2 小时以上）。plain requests.get 直接打
    # resolve 重定向后的 CDN 地址则稳定可用。
    url = "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()["data"]   # 返回 doc 列表

def build_qa_from_hf(docs_dir, qa_path, limit=None):
    docs_dir.mkdir(parents=True, exist_ok=True)
    data = _load_cuad_json()

    items = []
    seen_titles = set()

    for doc in data:
        title = doc["title"].strip()
        safe_title = re.sub(r'[\\/:*?"<>|]', "_", title)

        para = doc["paragraphs"][0]        
        context = para["context"]


        if safe_title not in seen_titles:
            doc_path = docs_dir / f"{safe_title}.txt"
            doc_path.write_text(context, encoding="utf-8")
            seen_titles.add(safe_title)

        for qa in para["qas"]:
            spans = []
            for ans in qa["answers"]:
                ans_text = ans["text"]
                ans_start = ans["answer_start"]
                spans.append(Span(start=ans_start, end=ans_start+len(ans_text), text=ans_text))

            items.append(QAItem(
                id=qa["id"],
                question=qa["question"],
                doc_id=safe_title,
                answers=[s.text for s in spans],
                spans=spans,
                has_answer=len(spans) > 0
            ))

            if limit and len(items) >= limit:
                break
     
        if limit and len(items) >= limit:
            break

    export_qa(items, qa_path)
    return items


def export_qa(items: list[QAItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            row = {
                "id": item.id,
                "question": item.question,
                "doc_id": item.doc_id,
                "answers": item.answers,
                "spans": [{"start": s.start, "end": s.end, "text": s.text} for s in item.spans],
                "has_answer": item.has_answer,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_qa(path: Path) -> list[QAItem]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            spans = [Span(**s) for s in data.get("spans", [])]
            item = QAItem(
                id=data["id"],
                question=data["question"],
                doc_id=data["doc_id"],
                answers=data["answers"],
                spans=spans,
                has_answer=data.get("has_answer", True)
            )
            items.append(item)
    return items