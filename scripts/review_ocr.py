"""
OCR 逐样本对比工具。

每种文档类型取第 1 个样本，运行 OCR 后将 GT 与识别结果并排写入
data/runs/ocr_review/<ts>.md，供人工 review。

用法：
    uv run scripts/review_ocr.py
    uv run scripts/review_ocr.py --model-version vlm

前置：.env 里设置 MINERU_API_TOKEN
"""
from __future__ import annotations

import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import PIL.Image
import editdistance
from dotenv import load_dotenv
from tqdm import tqdm

from lex_rag import ocr

PIL.Image.MAX_IMAGE_PIXELS = None

OUT_DIR = Path("data/runs/ocr_review")

TEXT_CATS = {"text_block", "header", "figure_caption", "table_caption",
             "page_footer", "page_header"}


# ---------------------------------------------------------------------------
# 复用 eval_ocr 的核心函数
# ---------------------------------------------------------------------------

def pil_to_png_bytes(img: PIL.Image.Image) -> bytes:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0
    return editdistance.eval(list(pred), list(gt)) / len(gt)


# ---------------------------------------------------------------------------
# 加载每类第 1 个样本
# ---------------------------------------------------------------------------

def load_one_per_type() -> list[dict]:
    from datasets import load_dataset

    anno_url   = "https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/main/OmniDocBench.json"
    anno_cache = Path("data/omnidocbench_annotations.json")
    anno_cache.parent.mkdir(parents=True, exist_ok=True)
    if not anno_cache.exists():
        print("下载标注文件 ...")
        urllib.request.urlretrieve(anno_url, anno_cache)

    with open(anno_cache, encoding="utf-8") as f:
        annotations: list[dict] = json.load(f)

    print("加载 OmniDocBench 图像 ...")
    ds = load_dataset("opendatalab/OmniDocBench", split="train")

    img_name_to_idx: dict[str, int] = {}
    for i in range(len(ds)):
        img = ds[i]["image"]
        fname = Path(img.filename).name if hasattr(img, "filename") and img.filename else None
        if fname:
            img_name_to_idx[fname] = i

    seen_types: set[str] = set()
    samples = []

    for idx, anno in enumerate(annotations):
        page_info = anno.get("page_info", {})
        attr      = page_info.get("page_attribute", {})
        dtype     = attr.get("data_source", "unknown").lower().replace(" ", "_")

        if dtype in seen_types:
            continue

        gt_parts = []
        for block in anno.get("layout_dets", []):
            if block.get("ignore"):
                continue
            if block.get("category_type") not in TEXT_CATS:
                continue
            text = (block.get("text") or "").strip()
            if text:
                gt_parts.append(text)
        gt_text = " ".join(gt_parts).strip()
        if not gt_text:
            continue

        img_name = Path(page_info.get("image_path", "")).name
        hf_idx   = img_name_to_idx.get(img_name, idx if idx < len(ds) else None)
        if hf_idx is None:
            continue

        pil_img = ds[hf_idx]["image"]
        pil_img.load()

        seen_types.add(dtype)
        samples.append({"image": pil_img, "gt_text": gt_text, "doc_type": dtype,
                         "image_name": img_name})

    print(f"共 {len(samples)} 种类型，每类 1 个样本")
    return samples


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    ocr.add_ocr_args(parser)
    args = parser.parse_args()
    load_dotenv()          # MINERU_API_TOKEN

    samples = load_one_per_type()
    opts = ocr.options_from_args(args)

    lines: list[str] = [
        f"# OCR Review — MinerU 在线 API（model_version={opts.model_version}）",
        f"生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    # 每类只有 1 个样本，总数是个位数，一次性提交成一批即可
    with ocr.MinerUCloudClient(options=opts) as client:
        payload = [(f"{item['doc_type']}_review.png", pil_to_png_bytes(item["image"]))
                   for item in samples]
        try:
            preds = client.parse(payload, progress=lambda n: tqdm.write(f"  已完成 {n} 个"))
        except Exception as e:
            preds = [f"[OCR 失败: {e}]"] * len(samples)

    for item, pred_md in zip(samples, preds):
        gt = item["gt_text"]
        score = cer(pred_md, gt)
        lines += [
            "---",
            f"## {item['doc_type']}",
            f"图像：`{item['image_name']}`　CER: **{score:.4f}**",
            "",
            "### Ground Truth",
            "```",
            gt[:2000] + ("..." if len(gt) > 2000 else ""),
            "```",
            "",
            "### OCR 输出",
            "```",
            pred_md[:2000] + ("..." if len(pred_md) > 2000 else ""),
            "```",
            "",
        ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"{ts}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
