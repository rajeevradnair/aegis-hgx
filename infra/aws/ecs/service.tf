resource "aws_ecs_task_definition" "inference" {
  family                   = "${local.name_prefix}-inference"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "inference"
      image     = var.container_image
      essential = true

      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "MODEL_URI"
          value = var.model_uri
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.inference.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "inference"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "inference" {
  name            = "${local.name_prefix}-inference-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.inference.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.inference.arn
    container_name   = "inference"
    container_port   = 8000
  }

  depends_on = [
    aws_lb_listener.http,
    aws_iam_role_policy_attachment.task_execution,
    aws_iam_role_policy_attachment.task_model_read
  ]
}