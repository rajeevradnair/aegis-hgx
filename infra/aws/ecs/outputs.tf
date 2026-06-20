output "ecs_cluster_name" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.main.name
}

output "default_vpc_id" {
  description = "Default VPC used for this deployment."
  value       = data.aws_vpc.default.id
}

output "default_subnet_ids" {
  description = "Default subnet IDs used for this deployment."
  value       = data.aws_subnets.default.ids
}

output "log_group_name" {
  description = "CloudWatch log group for inference container logs."
  value       = aws_cloudwatch_log_group.inference.name
}

output "alb_dns_name" {
  description = "Public DNS name of the inference load balancer."
  value       = aws_lb.inference.dns_name
}

output "inference_health_url" {
  description = "Health endpoint for the deployed inference service."
  value       = "http://${aws_lb.inference.dns_name}/api/v1/baseline_logistic/health"
}

output "inference_predict_url" {
  description = "Prediction endpoint for the deployed inference service."
  value       = "http://${aws_lb.inference.dns_name}/api/v1/baseline_logistic/predict"
}