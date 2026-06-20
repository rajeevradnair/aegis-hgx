data "aws_s3_bucket" "aegis_hgx" {
  bucket = var.bucket_name
}

locals {
  s3_prefixes = [
    "bronze/",
    "bronze/synthetic/",
    "bronze/synthetic/events/",
    "silver/",
    "silver/synthetic/",
    "silver/synthetic/events/",
    "gold/",
    "gold/features/",
    "gold/features/baseline_logistic/",
    
    "metadata/",
    "metadata/manifests/",
    "metadata/data_quality_reports/",
    "metadata/upload_logs/",
    
    "experiments/",
    "experiments/mlflow-artifacts/",
    
    "configs/",
    "artifacts/",
    "artifacts/models/",
    "artifacts/reports/"
  ]
}

resource "aws_s3_object" "prefixes" {
  for_each = toset(local.s3_prefixes)

  bucket  = data.aws_s3_bucket.aegis_hgx.bucket
  key     = each.value
  content = ""
}