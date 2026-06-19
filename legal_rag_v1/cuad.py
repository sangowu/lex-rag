from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import re
from huggingface_hub import hf_hub_download


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
    local_path = hf_hub_download(
        repo_id="theatticusproject/cuad",
        repo_type="dataset",
        filename="CUAD_v1/CUAD_v1.json",
    )
    with open(local_path, encoding="utf-8") as f:
        return json.load(f)["data"]   # 返回 doc 列表

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