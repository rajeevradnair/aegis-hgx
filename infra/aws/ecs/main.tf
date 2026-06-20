locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"
}

resource "aws_cloudwatch_log_group" "inference" {
  name              = "/ecs/${local.name_prefix}-inference"
  retention_in_days = 7
}