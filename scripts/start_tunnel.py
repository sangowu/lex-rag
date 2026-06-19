"""持久 SSH tunnel 脚本 — 手动启动，保持运行直到 Ctrl+C。

使用方式：
  python scripts/start_tunnel.py

tunnel 建立后，config.yaml 里的 provider 改为 direct，
EmbeddingClient 直接连 http://127.0.0.1:6006/v1，无需再管 tunnel。
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from legal_rag_v1.config import load_config


def build_ssh_cmd(cfg) -> list[str]:
    cmd = [
        "ssh",
        "-N",
        "-p", str(cfg.ssh_port),
        "-L", f"{cfg.ssh_local_port}:{cfg.ssh_remote_host}:{cfg.ssh_remote_port}",
        f"{cfg.ssh_user}@{cfg.ssh_host}",
    ]
    if cfg.ssh_key_path:
        cmd[2:2] = ["-i", cfg.ssh_key_path]
    return cmd


def main() -> None:
    cfg = load_config().embedding
    cmd = build_ssh_cmd(cfg)
    print(f"启动 tunnel: {' '.join(cmd)}")
    print(f"本地端口: {cfg.ssh_local_port}  →  远端 {cfg.ssh_remote_host}:{cfg.ssh_remote_port}")
    print("按 Ctrl+C 停止\n")

    while True:
        proc = subprocess.Popen(cmd)
        try:
            proc.wait()  # 阻塞，直到 ssh 进程退出
            print("tunnel 断开，3 秒后重连...")
            time.sleep(3)
        except KeyboardInterrupt:
            print("\n停止 tunnel")
            proc.terminate()
            sys.exit(0)


if __name__ == "__main__":
    main()
