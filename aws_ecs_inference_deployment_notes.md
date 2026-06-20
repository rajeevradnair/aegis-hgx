AEGIS-HGX AWS Inference Deployment Notes

Objective
Deploy the AEGIS-HGX FastAPI inference container to AWS as a production-like service.

Deployment Architecture
Developer machine
-> Docker build
-> Amazon ECR image repository
-> ECS Fargate task
-> ECS service
-> Application Load Balancer
-> Public /health and /predict endpoints

Core AWS Services

Amazon ECR
Stores the Docker image for the inference service.

Amazon ECS
Runs the containerized inference service.

AWS Fargate
Runs ECS containers without managing EC2 servers.

Application Load Balancer
Provides the public HTTP endpoint and forwards traffic to the ECS task.

Target Group
Tracks healthy ECS task IPs and routes traffic to port 8000.

CloudWatch Logs
Stores stdout and stderr output from the running container.

IAM Execution Role
Used by ECS to pull the image from ECR and write logs to CloudWatch.

IAM Task Role
Used by the FastAPI application code to read the model artifact from S3.

S3 Model Artifact
The deployed service loads the trained model from:
s3://aegis-hgx/artifacts/models/logistic_baseline.joblib

Key Runtime Flow
1. ECS starts the inference container.
2. FastAPI lifespan runs at startup.
3. The app reads MODEL_URI from the environment.
4. The app downloads the joblib model from S3 to /tmp/aegis_hgx_model.joblib.
5. joblib loads the sklearn pipeline into memory.
6. /health confirms the model is loaded.
7. /predict accepts validated synthetic cyber event payloads and returns a prediction.

Terraform Files
infra/aws/ecs/providers.tf
infra/aws/ecs/variables.tf
infra/aws/ecs/main.tf
infra/aws/ecs/iam.tf
infra/aws/ecs/networking.tf
infra/aws/ecs/service.tf
infra/aws/ecs/outputs.tf

Important Terraform Resources
aws_ecs_cluster.main
aws_cloudwatch_log_group.inference
aws_iam_role.task_execution
aws_iam_role.task
aws_iam_policy.model_read
aws_security_group.alb
aws_security_group.ecs_tasks
aws_lb.inference
aws_lb_target_group.inference
aws_lb_listener.http
aws_ecs_task_definition.inference
aws_ecs_service.inference

Important Environment Variable
MODEL_URI=s3://aegis-hgx/artifacts/models/logistic_baseline.joblib

Verification Commands

Confirm model artifact exists in S3:
aws s3 ls s3://aegis-hgx/artifacts/models/logistic_baseline.joblib

Build and push image to ECR:
docker build -f Dockerfile.inference -t $IMAGE_URI .
docker push $IMAGE_URI

Validate Terraform:
cd infra/aws/ecs
terraform fmt
terraform validate
terraform plan -var="container_image=$IMAGE_URI"

Apply Terraform:
terraform apply -var="container_image=$IMAGE_URI"

Get health URL:
terraform output -raw inference_health_url

Get prediction URL:
terraform output -raw inference_predict_url

Test health:
curl $(terraform output -raw inference_health_url)

Expected health response:
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/tmp/aegis_hgx_model.joblib"
}

Test prediction:
curl -X POST $(terraform output -raw inference_predict_url) \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_014",
    "host_id": "host_003",
    "process_name": "encoded_powershell",
    "event_type": "privilege_change",
    "source_ip": "10.0.0.12",
    "destination_ip": "203.0.113.18",
    "bytes_in": 500,
    "bytes_out": 95000,
    "event_hour": 2,
    "is_business_hour": false
  }'

Expected prediction response:
{
  "prediction": 0 or 1,
  "classification": "normal" or "suspicious",
  "suspicious_probability": number between 0.0 and 1.0
}

CLI:
aws logs tail /ecs/aegis-hgx-dev-inference --region us-west-2 --since 30m

Live logs:
aws logs tail /ecs/aegis-hgx-dev-inference --region us-west-2 --follow

Useful ECS Debugging Commands

Describe service:
aws ecs describe-services \
  --cluster aegis-hgx-dev-cluster \
  --services aegis-hgx-dev-inference-service \
  --region us-west-2

List running tasks:
aws ecs list-tasks \
  --cluster aegis-hgx-dev-cluster \
  --service-name aegis-hgx-dev-inference-service \
  --region us-west-2

List stopped tasks:
aws ecs list-tasks \
  --cluster aegis-hgx-dev-cluster \
  --service-name aegis-hgx-dev-inference-service \
  --desired-status STOPPED \
  --region us-west-2

Describe a task:
aws ecs describe-tasks \
  --cluster aegis-hgx-dev-cluster \
  --tasks <TASK_ARN> \
  --region us-west-2

Common Failure Patterns

EssentialContainerExited
The container process crashed. Check CloudWatch logs.

CannotPullContainerError
ECS could not pull the image from ECR. Check image URI and execution role.

AccessDenied
The task role cannot access S3 or another AWS service.

Target group unhealthy
The ALB cannot get a 200 response from /health.

No running tasks
Check ECS service events and stopped task reasons.

Security Notes
This milestone uses public HTTP for fast validation.
Production improvements should include:
- HTTPS
- ACM certificate
- custom domain
- authentication
- WAF
- private subnets
- NAT gateway or VPC endpoints
- request IDs
- structured logs
- rate limiting

Cleanup Commands
To avoid AWS charges after testing:
cd infra/aws/ecs
terraform destroy -var="container_image=$IMAGE_URI"
