variable "aws_region" {
  description = "AWS 区域。lex-rag 现有资源全部在 eu-west-1（爱尔兰）"
  type        = string
  default     = "eu-west-1"
}

variable "service_name" {
  description = "服务名，用作 ECS task family / ECR 仓库名 / 日志组名的统一前缀"
  type        = string
  default     = "legal-rag-v1"
}

variable "image_tag" {
  description = "要部署的容器镜像 tag。生产环境应传入 git commit SHA，不要用 latest（见 README 的说明）"
  type        = string
  default     = "latest"
}

variable "task_cpu" {
  description = "Fargate task 的 CPU 单位。1024 = 1 vCPU"
  type        = string
  default     = "1024"
}

variable "task_memory" {
  description = "Fargate task 的内存（MiB）。注意 Fargate 对 cpu/memory 的组合有硬性限制，1024 CPU 只能配 2048~8192"
  type        = string
  default     = "2048"
}

variable "container_port" {
  description = "容器内监听端口，与 scripts/serve.py 的 --port 保持一致"
  type        = number
  default     = 6800
}

variable "log_retention_days" {
  description = "CloudWatch 日志保留天数。默认永不过期会一直烧钱，个人项目 14 天足够"
  type        = number
  default     = 14
}
