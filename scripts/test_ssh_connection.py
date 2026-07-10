"""手动测试 SSH tunnel 连通性。

用法：
  python scripts/test_ssh_connection.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lex_rag.config import load_config
from lex_rag.ssh_tunnel import get_or_create


def main() -> None:
    cfg = load_config().embedding

    if cfg.provider != "ssh_tunnel":
        print(f"provider={cfg.provider}，无需 SSH tunnel，直接连接 {cfg.base_url}")
        return

    print(f"建立 SSH tunnel: {cfg.ssh_user}@{cfg.ssh_host}:{cfg.ssh_port}")
    print(f"端口映射: 127.0.0.1:{cfg.ssh_local_port} → {cfg.ssh_remote_host}:{cfg.ssh_remote_port}")

    tunnel = get_or_create(
        host=cfg.ssh_host,
        user=cfg.ssh_user,
        ssh_port=cfg.ssh_port,
        key_path=cfg.ssh_key_path,
        local_port=cfg.ssh_local_port,
        remote_host=cfg.ssh_remote_host,
        remote_port=cfg.ssh_remote_port,
    )

    try:
        tunnel.ensure()
        print("SSH tunnel 建立成功，等待 2 秒确认稳定...")
        time.sleep(2)
        if tunnel._proc and tunnel._proc.poll() is None:
            print(f"tunnel 运行中 (pid={tunnel._proc.pid})")
            print(f"现在可以访问: {cfg.base_url}")
        else:
            print("tunnel 已退出，请检查 SSH 配置")
            sys.exit(1)
    except RuntimeError as e:
        print(f"连接失败: {e}")
        sys.exit(1)
    finally:
        tunnel.close()
        print("tunnel 已关闭")


if __name__ == "__main__":
    main()
