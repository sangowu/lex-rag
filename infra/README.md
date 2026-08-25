# lex-rag 基础设施即代码（Terraform）

把原本手工在 AWS 控制台点出来 / 用 `aws` CLI 敲出来的资源，收敛成可版本化、可 review、可复现的代码。

## 当前纳管范围

| 资源 | Terraform 地址 | 说明 |
|---|---|---|
| CloudWatch 日志组 `/ecs/legal-rag-v1` | `aws_cloudwatch_log_group.app` | 已于 2026-07-31 import 纳管 |
| ECS Task Definition `legal-rag-v1` | `aws_ecs_task_definition.app` | 不可变资源，apply 会产生新 revision（当前 `:3`） |
| IAM roles / SSM 参数 | `data.*` | 只读引用，Terraform 不管理其生命周期 |

**尚未纳管**（后续分批接入）：ECS Service、ALB + Target Group、RDS、EC2 GPU 实例、ECR、安全组、VPC。
分批的原因见下面「为什么不一次全写」。

## 首次接入（已完成，记录备查）

**2026-07-31 完成首次接入**，结果 `1 imported, 1 added, 1 changed, 0 destroyed`：

- 日志组 `retention_in_days` 由 `0`（永不过期，`awslogs-create-group` 隐式创建的默认值）改为 `14`；
  后于 2026-08-01 调整为 `30`（PR #3），理由见该 PR：按排障回溯需求而非成本选值
- Task definition 生成 `legal-rag-v1:3`；**ECS service 未切换，仍运行 `:2`**
- 此后 `terraform plan` 应稳定输出 `No changes`

已经完成的步骤不需要重跑。以下流程供**下次接入新资源**时复用：

```bash
cd infra

# 1. 下载 provider，生成 .terraform.lock.hcl（lock 文件要提交进 git）
terraform init

# 2. 静态检查，不连网，随时可跑
terraform validate
terraform fmt -check -recursive

# 3. 把【已经存在】的资源认领进 state。
#    不做这步，apply 会因为 ResourceAlreadyExistsException 失败——
#    state 里没有记录，Terraform 会以为需要新建。
#
#    新建一个 imports.tf，用 import 块（Terraform 1.5+）：
#
#      import {
#        to = aws_cloudwatch_log_group.app   # 代码里的资源地址
#        id = "/ecs/legal-rag-v1"            # AWS 侧的真实 ID
#      }
#
#    为什么不用老命令 `terraform import <addr> <id>`：
#      老命令直接改 state、不经过 plan，做错了要 `terraform state rm` 手动撤；
#      import 块是代码，会先出现在 plan 里供预览，能进 git、能 review、能进 CI。
#    认领完成后删掉 imports.tf 即可——state 已经记住了对应关系，
#    留着它每次 plan 都会多做一次无意义的存在性检查。

# 4. 存下 plan 再 apply，保证执行的就是你审过的那一份
terraform plan -out=tfplan
#    ★ 确认输出里是 "0 to destroy" 且无 "must be replaced" 才继续
terraform apply tfplan
rm tfplan          # plan 文件含明文，用完即删（.gitignore 已排除）

# 5. 不要只信 Terraform 说成功，用 AWS CLI 独立核实
aws logs describe-log-groups --log-group-name-prefix /ecs/legal-rag-v1 \
    --region eu-west-1 --query 'logGroups[0].retentionInDays'

# 6. 删掉 imports.tf 后重跑，必须是 No changes，才算真正收敛
terraform plan
```

## 日常操作

```bash
terraform fmt      # 格式化，等价于 ruff format
terraform validate # 语法与类型检查，等价于 ruff check，不连网
terraform plan     # 干跑，展示 diff
terraform apply    # 真正执行
terraform show     # 查看当前 state 里记录的资源
terraform state list
```

## 为什么不一次把所有资源都写进来

第一次把存量环境接入 Terraform 时，最大的风险是 **plan 里出现意料之外的 `destroy`**。
手工建的资源往往带着一堆你不知道的默认属性，代码里没写全，Terraform 就认为"要改成我说的样子"，
轻则 in-place 修改，重则 **destroy and recreate**（对 RDS 来说等于删库）。

所以这里的策略是：

1. 先只纳管**破坏后果最轻**的资源（日志组、task definition）。
   task definition 是 append-only 的，apply 出问题最多多一个没人用的 revision。
2. 每接入一个新资源，都先 `terraform import` + `terraform plan`，
   **确认 plan 输出是 `No changes`**，再往下走。
3. RDS、EC2 这类有状态资源放到最后，并且加 `lifecycle { prevent_destroy = true }`。

## 安全须知

- `terraform.tfstate` 含**明文敏感信息**，已在 `.gitignore` 中排除，**绝不可提交**。
- 目前用的是本地 state。多人协作或接入 CI 前，必须迁移到 S3 backend +
  DynamoDB 状态锁，否则两个人同时 apply 会写坏 state。
- 密钥一律走 **SSM Parameter Store**（SecureString + AWS 托管密钥 `alias/aws/ssm`），
  `.tf` 文件里只出现 ARN，不出现明文。选 SSM 而不是 Secrets Manager 的原因是成本：
  标准 SSM 参数免费，Secrets Manager 每个密钥 $0.40/月且服务停机也照收。

## 部署前置：先建好这四个参数

`terraform plan` 会在 data source 阶段就检查它们是否存在，缺一个就报错。

```bash
for kv in   "legal-rag-v1/pg-password:<PG 密码>"   "legal-rag-v1/embed-api-key:<SiliconFlow key>"   "legal-rag-v1/rerank-api-key:<SiliconFlow key，可与上面相同>"   "legal-rag-v1/generate-model-api:<Z.ai key>"
do
  aws ssm put-parameter --name "${kv%%:*}" --value "${kv#*:}"       --type SecureString --overwrite
done
```

> 不要加 `--tier Advanced`（$0.05/个/月），也不要指定自建 KMS 密钥（$1/月）——
> 默认的标准层 + `alias/aws/ssm` 才是免费的。

执行角色 `rag-ecs-execution-role` 需要有 `ssm:GetParameters`（对这四个参数 ARN）
和 `kms:Decrypt`（对 `alias/aws/ssm`）。这两个角色是账号里手工建的，本目录的
Terraform 只引用、不管理其策略；权限缺失的表现是任务启动失败并报
`ResourceInitializationError`。

旧的 Secrets Manager 条目（`legal-rag-v1/pg-password`、`legal-rag-v1/gemini-api-key`）
迁移后可以删掉止损：

```bash
aws secretsmanager delete-secret --secret-id legal-rag-v1/gemini-api-key     --force-delete-without-recovery
```
