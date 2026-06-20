AEGIS-HGX S3 Data Lake Notes

Objective
Create a production-like AWS S3 data lake layout inside the existing aegis-hgx bucket.

Existing Bucket
s3://aegis-hgx

No new bucket was created. Terraform only manages prefix placeholder objects inside the existing bucket.

Primary S3 Zones

bronze/
Raw or near-raw source data.
Example:
s3://aegis-hgx/bronze/synthetic/events/synthetic_events.csv

silver/
Cleaned, validated, standardized data.
Future example:
s3://aegis-hgx/silver/synthetic/events/validated_events.parquet

gold/
Model-ready or analytics-ready data.
Future example:
s3://aegis-hgx/gold/features/baseline_logistic/features.parquet

metadata/
Data lake bookkeeping.
Examples:
- manifests
- schema versions
- data quality reports
- upload logs

configs/
Cloud copies of configuration snapshots used by training, pipelines, and deployment.

artifacts/
Exported model artifacts and reports.
Examples:
- artifacts/models/
- artifacts/reports/

experiments/
Experiment tracking artifacts.
Future MLflow artifact location:
s3://aegis-hgx/experiments/mlflow-artifacts/

Terraform Files
infra/aws/s3/providers.tf
infra/aws/s3/variables.tf
infra/aws/s3/main.tf
infra/aws/s3/outputs.tf

Terraform Design
data "aws_s3_bucket" "aegis_hgx"
- Looks up the existing bucket.
- Does not create or own the bucket.

aws_s3_object.prefixes
- Creates zero-byte placeholder objects for prefix visibility.
- Makes S3 console display folder-like structure.
- Does not replace or delete the bucket.

Created Prefixes
bronze/
bronze/synthetic/
bronze/synthetic/events/

silver/
silver/synthetic/
silver/synthetic/events/

gold/
gold/features/
gold/features/baseline_logistic/

metadata/
metadata/manifests/
metadata/schema_versions/
metadata/data_quality_reports/
metadata/upload_logs/

configs/

artifacts/
artifacts/models/
artifacts/reports/

experiments/
experiments/mlflow-artifacts/

Verification Commands

Validate Terraform:
cd infra/aws/s3
terraform fmt
terraform validate
terraform plan

Apply Terraform:
terraform apply

Verify top-level prefixes:
aws s3 ls s3://aegis-hgx/

Expected top-level prefixes:
artifacts/
bronze/
configs/
experiments/
gold/
metadata/
silver/

Upload Synthetic Data to Bronze:
aws s3 cp data/processed/synthetic_events.csv s3://aegis-hgx/bronze/synthetic/events/synthetic_events.csv

Verify Bronze Upload:
aws s3 ls s3://aegis-hgx/bronze/synthetic/events/

Upload Manifest:
aws s3 cp metadata/manifests/synthetic_events_manifest.json s3://aegis-hgx/metadata/manifests/synthetic_events_manifest.json

Verify Manifest:
aws s3 ls s3://aegis-hgx/metadata/manifests/

Important First-Principles Notes

S3 buckets are not folders.
S3 stores objects with keys.

Example:
bucket:
aegis-hgx

object key:
bronze/synthetic/events/synthetic_events.csv

Full S3 URI:
s3://aegis-hgx/bronze/synthetic/events/synthetic_events.csv

The folder-like structure is created through prefixes.

Bronze should preserve source evidence.
Silver should contain validated and cleaned data.
Gold should contain model-ready or analytics-ready datasets.

Metadata/manifests record what was uploaded, where it came from, and what it is for.

MLflow Note
Local MLflow still uses:
mlflow/mlflow.db
mlflow/mlruns/

Future cloud MLflow artifacts can use:
s3://aegis-hgx/experiments/mlflow-artifacts/

This separates experiment artifacts from data lake source data.

Cheat Sheet

Bucket:
Top-level S3 storage container.

Object:
One stored file or placeholder in S3.

Key:
The path-like name of an object.

Prefix:
The beginning of an object key. Used like a folder.

Bronze:
Raw source data.

Silver:
Validated and cleaned data.

Gold:
Model-ready data.

Manifest:
Small metadata file describing an uploaded dataset.
