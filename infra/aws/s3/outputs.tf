output "bucket_name" {
  description = "Existing S3 bucket used by AEGIS-HGX."
  value       = data.aws_s3_bucket.aegis_hgx.bucket
}

output "bucket_arn" {
  description = "ARN of the existing S3 bucket."
  value       = data.aws_s3_bucket.aegis_hgx.arn
}