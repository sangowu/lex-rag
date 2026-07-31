output "task_definition_arn" {
  description = "最新一版 task definition 的完整 ARN（带 revision 号）"
  value       = aws_ecs_task_definition.app.arn
}

output "task_definition_revision" {
  description = "revision 号。每次 apply 有变更时会 +1，用来确认部署了哪一版"
  value       = aws_ecs_task_definition.app.revision
}

output "image_uri" {
  description = "本次部署使用的完整镜像地址"
  value       = local.image_uri
}

output "log_group_name" {
  description = "CloudWatch 日志组名，排查问题时直接拿去 aws logs tail"
  value       = aws_cloudwatch_log_group.app.name
}
