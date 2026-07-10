"""SSH tunnel manager — 把远程模型服务映射到本地端口。

用法：
  tunnel = SSHTunnel(host, user, ssh_port, key_path, local_port, remote_host, remote_port)
  tunnel.ensure()   # 建立/复用隧道
  tunnel.close()    # 关闭
"""
import atexit
import subprocess
import time
from dataclasses import dataclass, field

_REGISTRY: dict[tuple, "SSHTunnel"] = {}


@dataclass
class SSHTunnel:
    host: str
    user: str
    ssh_port: int = 22
    key_path: str = ""
    local_port: int = 6006
    remote_host: str = "127.0.0.1"
    remote_port: int = 8000
    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)

    def _cmd(self) -> list[str]:
        cmd = [
            "ssh", "-N",
            "-p", str(self.ssh_port),
            "-L", f"{self.local_port}:{self.remote_host}:{self.remote_port}",
            f"{self.user}@{self.host}",
        ]
        if self.key_path:
            cmd[2:2] = ["-i", self.key_path]
        return cmd

    def ensure(self) -> None:
        """建立隧道；已在运行则直接复用。"""
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            self._cmd(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.6)
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"SSH tunnel failed to start: {' '.join(self._cmd())}"
            )

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()


def get_or_create(
    host: str,
    user: str,
    ssh_port: int = 22,
    key_path: str = "",
    local_port: int = 6006,
    remote_host: str = "127.0.0.1",
    remote_port: int = 8000,
) -> SSHTunnel:
    """获取已有隧道或新建一个，全局复用。"""
    key = (host, user, ssh_port, key_path, local_port, remote_host, remote_port)
    if key not in _REGISTRY:
        _REGISTRY[key] = SSHTunnel(
            host=host,
            user=user,
            ssh_port=ssh_port,
            key_path=key_path,
            local_port=local_port,
            remote_host=remote_host,
            remote_port=remote_port,
        )
    return _REGISTRY[key]


def _close_all() -> None:
    for t in _REGISTRY.values():
        t.close()


atexit.register(_close_all)
