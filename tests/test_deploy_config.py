"""部署配置必须过得了 `serve.py` 自己的启动守卫。

起因：`serve.py` 在 #42 长出了两道启动守卫（非回环地址必须有 `API_KEYS`；
公网不许挂 Gradio），而 `deploy/task-definition.json` 和 `infra/main.tf` 没跟上。
两边都跑 `--host 0.0.0.0`、都没有 `API_KEYS`、都挂着 UI —— **两道守卫全撞**。

症状不会是"部署失败"，而是**容器起来立刻退出、ECS 不停重启它**，看起来像镜像坏了。
而且这个洞在仓库里躺了整整一个版本没人发现，因为 AWS 侧是下线的，没人真跑过。

所以这里不重新实现一遍判断，而是**把任务定义喂给守卫本身**——判据和线上是同一个
函数，改了守卫这里就跟着变，不会出现"测试过了但线上起不来"。

⚠️ `deploy/task-definition.json` 与 `infra/main.tf` 内容重复，**这正是漂移的来源**。
本仓库已经栽过五次"两处各写一份、分叉时无声"，所以下面也钉住两者一致。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.serve import bind_safety_error

ROOT = Path(__file__).resolve().parents[1]
TASK_DEF = ROOT / "deploy" / "task-definition.json"
MAIN_TF = ROOT / "infra" / "main.tf"


@pytest.fixture(scope="module")
def container() -> dict:
    return json.loads(TASK_DEF.read_text(encoding="utf-8"))["containerDefinitions"][0]


def _flag_value(command: list[str], flag: str) -> str | None:
    return command[command.index(flag) + 1] if flag in command else None


def test_the_deployed_command_passes_the_startup_guard(container):
    """**这条就是那个 bug 本身。**

    直接调用线上用的同一个 `bind_safety_error`，参数全部从任务定义里读出来。
    """
    command = container["command"]
    secrets = {s["name"] for s in container.get("secrets", [])}

    err = bind_safety_error(
        _flag_value(command, "--host") or "127.0.0.1",
        auth_enabled="API_KEYS" in secrets,
        ui_mounted="--no-ui" not in command,
        allow_public_ui="--allow-public-ui" in command,
    )
    assert err is None, f"任务定义过不了启动守卫：\n{err}"


def test_api_keys_comes_from_ssm_not_from_a_plain_env_var(container):
    """密钥只能走 `secrets`（ECS agent 去 SSM 取），不能写进 `environment`——
    那里的值会明文出现在任务定义、控制台和 `describe-task-definition` 输出里。"""
    env_names = {e["name"] for e in container.get("environment", [])}
    assert "API_KEYS" not in env_names


def test_the_public_deployment_does_not_mount_the_ui(container):
    """`/ui` 是鉴权豁免路径（浏览器发不出自定义头）。公网挂着它，
    等于把整个合同库开给全世界——而**功能完全正常**，不会有任何报错。"""
    assert "--no-ui" in container["command"]
    assert "--allow-public-ui" not in container["command"]


def test_the_guard_would_have_caught_the_old_configuration():
    """反向确认这条测试有效：把改动撤回去，守卫必须报错。

    没有这一条的话，`bind_safety_error` 哪天变成永远返回 None，
    上面那条会安静地继续通过。
    """
    assert bind_safety_error("0.0.0.0", auth_enabled=False,
                             ui_mounted=True, allow_public_ui=False) is not None


# --- 两份配置必须一致 ---------------------------------------------------------

@pytest.fixture(scope="module")
def main_tf() -> str:
    return MAIN_TF.read_text(encoding="utf-8")


def test_terraform_declares_the_same_secrets(container, main_tf):
    """Terraform 那边所有东西都从 `local.runtime_secrets` 推导（data source、
    容器 secrets、IAM 读取权限），所以只要每个名字在那张表里出现即可。"""
    for name in (s["name"] for s in container["secrets"]):
        assert name in main_tf, f"{name} 在 task-definition.json 里有，infra/main.tf 里没有"


def test_terraform_uses_the_same_flags(main_tf):
    assert '"--no-ui"' in main_tf
