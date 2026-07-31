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

# 已经存在的 Secrets Manager 条目。
# 注意：这里只拿到 ARN，Terraform 完全不会读取密文，
# 密文由 ECS agent 在启动容器时自己去取。
data "aws_secretsmanager_secret" "pg_password" {
  name = "${var.service_name}/pg-password"
}

data "aws_secretsmanager_secret" "gemini_api_key" {
  name = "${var.service_name}/gemini-api-key"
}

# ---------------------------------------------------------------------------
# Locals —— 相当于局部变量，用来消除重复。
# ---------------------------------------------------------------------------

locals {
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

      # 非敏感的普通环境变量
      environment = [
        { name = "EMBED_API_KEY", value = "" },
      ]

      # 敏感值：只传 ARN，明文永远不进 Terraform，也不进任务定义。
      # ECS agent 启动容器前会自己去 Secrets Manager 取，注入成环境变量。
      secrets = [
        {
          name      = "PG_PASSWORD"
          valueFrom = data.aws_secretsmanager_secret.pg_password.arn
        },
        {
          name      = "GEMINI_API_KEY"
          valueFrom = data.aws_secretsmanager_secret.gemini_api_key.arn
        },
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
