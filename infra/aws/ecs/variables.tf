variable "aws_region" {
  description = "AWS region for the ECS deployment."
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Project name used for AWS resource naming."
  type        = string
  default     = "aegis-hgx"
}

variable "environment" {
  description = "Deployment environment."
  type        = string
  default     = "dev"
}

variable "container_image" {
  description = "Full ECR image URI for the inference container."
  type        = string
}

variable "model_uri" {
  description = "S3 URI for the trained model artifact."
  type        = string
  default     = "s3://aegis-hgx/artifacts/models/logistic_baseline.joblib"
}