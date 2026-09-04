"""把一份 CUAD 合同渲染成"扫描件"，给 OCR → RAG 那条链造语料。

    uv run scripts/make_scanned_demo.py                      # 默认那份 CENTRACK 合同
    uv run scripts/make_scanned_demo.py --doc-id "..." --out-dir data/scanned_docs

**为什么是合成的。** OmniDocBench 的九类里没有合同（academic_literature /
research_report / book / PPT2PDF / colorful_textbook / magazine / exam_paper /
newspaper / note），拿一张论文扫描件去演法务 RAG，跑通了也说明不了什么。而真实
扫描合同拿不到 ground truth——没有 ground truth 就只能截图说"你看它跑通了"，
答不出"OCR 误差吃掉多少端到端质量"这个唯一值得问的问题。

从 CUAD 原文反向渲染同时给到两样东西：逐字的 ground truth，和这份合同现成的 41
条 CUAD 问答对。代价是**退化是我模拟的，不是真实成像**，所以：

⚠️ **这里量出来的 CER 不能代表真实扫描件**，它只是"这一档模拟退化"下的数字。
真实扫描的失真形态（装订阴影、透印、折痕、非均匀光照、二次复印）这里一个都没有。
可比的是**同一份文本走 OCR 和不走 OCR 的差**，不是 CER 的绝对值。

退化是**定死随机种子**的，重跑得到逐像素相同的图，否则 demo 里的数字下次就对不上。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_DOC = "CENTRACKINTERNATIONALINC_10_29_1999-EX-10.3-WEB SITE HOSTING AGREEMENT"

# A4 @ 200 dpi。再高对 OCR 没有增益，只是让上传变慢。
PAGE_W, PAGE_H = 1654, 2339
MARGIN_X, MARGIN_Y = 150, 170
FONT_SIZE, LINE_H = 26, 38

# SEC 的 EDGAR 文本本来就是等宽排版，用等宽字体渲染是还原而不是风格选择。
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\cour.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/System/Library/Fonts/Courier.ttc",
]


def _load_font() -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, FONT_SIZE)
    raise SystemExit("找不到等宽字体，请在 FONT_CANDIDATES 里加一个本机路径")


def _wrap(text: str, cols: int) -> list[str]:
    """按列宽折行，**保留原文的空行与前导空格**。

    CUAD 原文的缩进是它排版结构的一部分（条款编号靠缩进对齐），
    strip 掉再渲染就等于先毁掉一部分版面信息再去考 OCR。
    """
    out: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            out.append("")
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        pad = " " * min(indent, cols // 3)
        words, line = raw.split(), pad
        for w in words:
            candidate = f"{line}{w} " if line.strip() or line else f"{pad}{w} "
            if len(candidate.rstrip()) > cols and line.strip():
                out.append(line.rstrip())
                line = f"{pad}{w} "
            else:
                line = candidate
        if line.strip():
            out.append(line.rstrip())
    return out


def _render_pages(lines: list[str], font: ImageFont.FreeTypeFont) -> list[Image.Image]:
    per_page = (PAGE_H - 2 * MARGIN_Y) // LINE_H
    pages = []
    for i in range(0, len(lines), per_page):
        img = Image.new("L", (PAGE_W, PAGE_H), 255)
        d = ImageDraw.Draw(img)
        y = MARGIN_Y
        for line in lines[i:i + per_page]:
            d.text((MARGIN_X, y), line, font=font, fill=25)
            y += LINE_H
        # 页脚页码：真实 EDGAR 打印件有，而且它是 OCR 常见的误读来源之一
        d.text((PAGE_W // 2, PAGE_H - 100), str(i // per_page + 1), font=font, fill=60)
        pages.append(img)
    return pages


def _degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    """模拟一次平板扫描：轻微歪斜 + 焦外 + 传感器噪点 + 对比度损失。

    幅度刻意压得保守。目标是"一份能看的复印件"，不是"最坏情况"——
    把退化调到 OCR 明显崩掉很容易，但那样得到的数字只说明我把噪声开大了。
    """
    img = img.rotate(rng.uniform(-0.35, 0.35), resample=Image.BICUBIC,
                     fillcolor=255, expand=False)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 0.7)))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.82, 0.92))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.96, 1.02))

    px = img.load()
    w, h = img.size
    # 逐像素加噪太慢（400 万像素 × 6 页），只在稀疏采样点上加，视觉与 OCR 效果相当
    for _ in range(w * h // 60):
        x, y = rng.randrange(w), rng.randrange(h)
        px[x, y] = max(0, min(255, px[x, y] + rng.randint(-55, 35)))
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="CUAD 合同 → 合成扫描件（PDF）")
    ap.add_argument("--doc-id", default=DEFAULT_DOC)
    ap.add_argument("--docs-dir", default="data/cuad_docs")
    ap.add_argument("--out-dir", default="data/scanned_docs")
    ap.add_argument("--seed", type=int, default=20260904,
                    help="定死种子，重跑得到相同的图；换种子等于换了一份语料")
    args = ap.parse_args()

    src = Path(args.docs_dir) / f"{args.doc_id}.txt"
    if not src.exists():
        raise SystemExit(f"找不到原文：{src}")

    text = src.read_text(encoding="utf-8", errors="replace")
    font = _load_font()
    cols = (PAGE_W - 2 * MARGIN_X) // font.getlength("M")
    pages = _render_pages(_wrap(text, int(cols)), font)

    rng = random.Random(args.seed)
    pages = [_degrade(p, rng).convert("RGB") for p in pages]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 输出**单个多页 PDF**，文件名就是 doc_id：ingest_ocr.py 用 path.stem 当 doc_id，
    # 拆成多张图会把一份合同变成 6 个文档，按 doc_id 检索就对不上 CUAD 的问答对了。
    out = out_dir / f"{args.doc_id}.pdf"
    # PdfImagePlugin 直接查 Image.SAVE["JPEG"]，而这张表要 Image.init() 之后才有内容
    # （平时由 open/save 顺带触发）。少这一行会以 KeyError: 'JPEG' 挂掉。
    Image.init()
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=200.0)

    print(f"{len(pages)} 页 -> {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
