variable "aws_region" {
  description = "AWS region where the existing S3 bucket is located."
  type        = string
  default     = "us-west-2"
}

variable "bucket_name" {
  description = "Existing S3 bucket used by AEGIS-HGX."
  type        = string
  default     = "aegis-hgx"
}

variable "environment" {
  description = "Environment name used in S3 prefixes."
  type        = string
  default     = "dev"
}