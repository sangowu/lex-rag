"""
MinerU OCR 服务启动脚本（MinerU 3.x）。

MinerU 3.x 自带 mineru-api FastAPI 服务，本脚本为便捷启动封装，
负责设置必要的环境变量后启动服务。

与 start_model_services.py 使用相同端口（1080），两个服务不同时运行。

启动：
    python scripts/start_ocr_service.py
    python scripts/start_ocr_service.py --host 0.0.0.0

健康检查：
    curl http://127.0.0.1:1080/health
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OCR_HOST",   "127.0.0.1")
os.environ.setdefault("OCR_PORT",   "1080")
os.environ.setdefault("OCR_DEVICE", "cuda")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TMPDIR",     "/root/autodl-tmp/tmp")
os.environ.setdefault("MINERU_MODEL_SOURCE", "modelscope")
os.environ.setdefault("MODEL_CACHE_DIR",
                      str(Path.home() / ".cache" / "modelscope" / "hub"))

HOST   = os.getenv("OCR_HOST",   "127.0.0.1")
PORT   = int(os.getenv("OCR_PORT",   "1080"))
DEVICE = os.getenv("OCR_DEVICE", "cuda")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default=HOST)
    parser.add_argument("--port",   type=int, default=PORT)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--vlm",    action="store_true",
                        help="启动时预加载 VLM 模型（支持 --backend vlm 请求，显存占用更高）")
    args = parser.parse_args()

    os.environ["OCR_DEVICE"] = args.device
    print(f"[ocr-service] host={args.host}  port={args.port}  device={args.device}  vlm={args.vlm}")

    cmd = ["mineru-api", "--host", args.host, "--port", str(args.port)]
    if args.vlm:
        cmd.extend(["--enable-vlm-preload", "true"])

    result = subprocess.run(cmd, env=os.environ)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
