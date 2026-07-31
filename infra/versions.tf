terraform {
  # Terraform CLI 自身的最低版本。低于这个版本会直接报错退出，
  # 避免同事用老版本跑出诡异结果。
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      # ~> 6.0 表示 "允许 6.x，但不允许 7.0"。
      # 大版本升级 provider 经常有破坏性变更，必须显式操作，不能自动跳。
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  # 给这个 provider 创建的所有资源自动打标签。
  # 生产环境的价值：成本分摊报表能按 Project 维度切，
  # 以及一眼看出哪些资源是 Terraform 管的、哪些是人手点的。
  default_tags {
    tags = {
      Project   = "lex-rag"
      ManagedBy = "terraform"
    }
  }
}
