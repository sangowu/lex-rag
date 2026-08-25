# ---------------------------------------------------------------------------
# Data sources —— 只读查询，不创建任何东西。
# 用它们代替硬编码，是 Terraform 里最基本的好习惯。
# ---------------------------------------------------------------------------

# 当前 AWS 账号 ID。这样就不用把 569260897196 抄进代码里，
# 换个账号跑同一份代码也能工作。
data "aws_caller_identity" "current" {}

# 当前区域。
data "aws_region" "current" {}

# 已经存在的 IAM role —— 这两个是你之前手工建的。
# 用 data 而不是 resource，表示"我只是引用它，不负责它的生命周期"。
# 好处：terraform destroy 不会把它们删掉。
data "aws_iam_role" "ecs_execution" {
  name = "rag-ecs-execution-role"
}

data "aws_iam_role" "ecs_task" {
  name = "rag-ecs-task-role"
}

# 运行时密钥存在 SSM Parameter Store（SecureString + AWS 托管密钥 alias/aws/ssm）。
#
# 为什么不用 Secrets Manager：标准 SSM 参数免费，Secrets Manager 每个密钥
# $0.40/月，服务停机也照收。ECS 的 secrets.valueFrom 对两者都是原生支持，
# 容器侧完全无感。
#
# with_decryption = false 是刻意的：这里只需要 ARN，关掉解密可以避免把明文
# 拉进 terraform state。用 data 而不是拼字符串，是为了在 plan 阶段就能发现
# 参数不存在，而不是等容器启动失败才知道。
data "aws_ssm_parameter" "runtime" {
  for_each        = local.runtime_secrets
  name            = each.value
  with_decryption = false
}

# ---------------------------------------------------------------------------
# Locals —— 相当于局部变量，用来消除重复。
# ---------------------------------------------------------------------------

locals {
  # 容器运行时需要的密钥：环境变量名 => SSM 参数名。
  # 需要先在 AWS 里建好这些 SecureString 参数，terraform plan 才能通过。
  #
  # 不含 MINERU_API_TOKEN：OCR 是独立的离线脚本，不在 serve.py 的服务路径里。
  # 将来若要在容器内跑 ingest_ocr，把它加进这张表即可。
  # 名字带前导斜杠，与账号里既有的 /proshot/* 命名习惯一致。
  # SSM 的 GetParameter 按名字精确匹配，少一个斜杠就是 ParameterNotFound。
  runtime_secrets = {
    PG_PASSWORD        = "/${var.service_name}/pg-password"
    EMBED_API_KEY      = "/${var.service_name}/embed-api-key"
    RERANK_API_KEY     = "/${var.service_name}/rerank-api-key"
    GENERATE_MODEL_API = "/${var.service_name}/generate-model-api"
  }

  # 拼出完整的 ECR 镜像地址，替代原来硬编码的
  # 569260897196.dkr.ecr.eu-west-1.amazonaws.com/legal-rag-v1:latest
  image_uri = format(
    "%s.dkr.ecr.%s.amazonaws.com/%s:%s",
    data.aws_caller_identity.current.account_id,
    data.aws_region.current.region,
    var.service_name,
    var.image_tag,
  )

  log_group_name = "/ecs/${var.service_name}"
}

# ---------------------------------------------------------------------------
# CloudWatch 日志组
#
# 原来这个日志组是靠 task-definition.json 里的 "awslogs-create-group": "true"
# 隐式创建的 —— 那种方式创建的日志组【永不过期】，会一直累积费用。
# 现在显式声明出来，就能管住保留期。
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "app" {
  name              = local.log_group_name
  retention_in_days = var.log_retention_days
}

# ---------------------------------------------------------------------------
# ECS Task Definition
#
# 对应原来的 deploy/task-definition.json。
# Task definition 是【不可变】的：每次改动 AWS 都会生成一个新 revision
# （:1 → :2 → :3），旧的永远保留。所以改这个文件是安全的，
# 真正的切换发生在 ECS service 指向哪个 revision。
# ---------------------------------------------------------------------------

resource "aws_ecs_task_definition" "app" {
  family                   = var.service_name
  network_mode             = "awsvpc" # Fargate 强制要求
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory

  # 拉镜像、写日志、读 secret 用的角色（ECS agent 用）
  execution_role_arn = data.aws_iam_role.ecs_execution.arn
  # 容器内的应用代码调用 AWS API 时用的角色
  task_role_arn = data.aws_iam_role.ecs_task.arn

  # container_definitions 必须是 JSON 字符串。
  # jsonencode() 让我们能用 HCL 写，避免手写 JSON 时的引号地狱。
  container_definitions = jsonencode([
    {
      name  = var.service_name
      image = local.image_uri

      command = [
        "python", "scripts/serve.py",
        "--host", "0.0.0.0",
        "--port", tostring(var.container_port),
        "--root-path", "/legal-rag",
      ]

      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]

      # 非敏感的普通环境变量。
      # EMBED_API_KEY 原先在这里、值是空串 —— 容器起来后所有 embedding 调用
      # 都会 401，且报错指向 SiliconFlow 而不是配置本身。已挪进 secrets。
      environment = []

      # 敏感值：只传 ARN，明文不进 Terraform，也不进任务定义。
      # ECS agent 启动容器前自己去 SSM 取，注入成环境变量。
      #
      # 前提：执行角色（rag-ecs-execution-role）需要有
      #   ssm:GetParameters  对这些参数 ARN
      #   kms:Decrypt        对 alias/aws/ssm
      # 这两个角色是账号里手工建的（上面用 data 引用），本文件不管它们的策略，
      # 缺权限的表现是任务启动失败并报 ResourceInitializationError。
      secrets = [
        for env_name, param_name in local.runtime_secrets : {
          name      = env_name
          valueFrom = data.aws_ssm_parameter.runtime[env_name].arn
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app.name
          "awslogs-region"        = data.aws_region.current.region
          "awslogs-stream-prefix" = "ecs"
          # 不再需要 awslogs-create-group：日志组现在由上面的资源显式管理。
          # 这里引用 aws_cloudwatch_log_group.app.name 会让 Terraform
          # 自动推导出依赖关系，保证日志组先于 task definition 创建。
        }
      }
    }
  ])
}
