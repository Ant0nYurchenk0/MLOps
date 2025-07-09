# SageMaker Setup Guide

## Prerequisites

1. **AWS Account**: Ensure you have an AWS account with appropriate permissions
2. **IAM Role**: Create a SageMaker execution role with the following permissions:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess` (or restricted to your specific buckets)
   - `AmazonEC2ContainerRegistryFullAccess` (if using custom Docker images)

## IAM Role Creation

### 1. Create SageMaker Execution Role

```bash
# Create trust policy for SageMaker
cat > sagemaker-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role
aws iam create-role \
  --role-name SageMakerQuoraTrainingRole \
  --assume-role-policy-document file://sagemaker-trust-policy.json

# Attach managed policies
aws iam attach-role-policy \
  --role-name SageMakerQuoraTrainingRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name SageMakerQuoraTrainingRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### 2. Get Role ARN

```bash
aws iam get-role --role-name SageMakerQuoraTrainingRole --query 'Role.Arn' --output text
```

## Configuration

### 1. Update docker-compose.yml

```yaml
environment:
  USE_SAGEMAKER: "true"
  SAGEMAKER_ROLE_ARN: "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerQuoraTrainingRole"
  SAGEMAKER_INSTANCE_TYPE: "ml.m5.large"  # Or ml.c5.xlarge for CPU-optimized
  AWS_DEFAULT_REGION: "us-east-1"
  # Add AWS credentials if not using IAM instance profile
  AWS_ACCESS_KEY_ID: "your_access_key"
  AWS_SECRET_ACCESS_KEY: "your_secret_key"
```

### 2. Instance Types

Common CPU-optimized instance types for training:
- `ml.m5.large`: 2 vCPUs, 8 GB RAM - Good for small models
- `ml.m5.xlarge`: 4 vCPUs, 16 GB RAM - Recommended for this workload
- `ml.c5.xlarge`: 4 vCPUs, 8 GB RAM - CPU-optimized
- `ml.c5.2xlarge`: 8 vCPUs, 16 GB RAM - Larger CPU-optimized

### 3. AWS Credentials

#### Option 1: Environment Variables (for testing)
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

#### Option 2: IAM Instance Profile (recommended for production)
If running on EC2, attach an IAM role with SageMaker permissions to the instance.

#### Option 3: AWS CLI Configuration
```bash
aws configure
```

## Usage

### 1. Enable SageMaker Training

Set the environment variable in your deployment:
```bash
USE_SAGEMAKER=true
```

### 2. Monitor Training Jobs

#### Via AWS Console
1. Go to SageMaker console
2. Navigate to Training Jobs
3. Find your job (named `quora-training-YYYYMMDD-HHMMSS`)

#### Via AWS CLI
```bash
# List training jobs
aws sagemaker list-training-jobs --sort-by CreationTime --sort-order Descending

# Get job details
aws sagemaker describe-training-job --training-job-name quora-training-YYYYMMDD-HHMMSS

# Get logs
aws logs describe-log-streams --log-group-name /aws/sagemaker/TrainingJobs/quora-training-YYYYMMDD-HHMMSS
```

### 3. Cost Optimization

1. **Right-size instances**: Start with `ml.m5.large` and scale up if needed
2. **Use Spot instances**: Add `use_spot_instances=True` to the estimator
3. **Set max_run**: Limit training time to prevent runaway costs
4. **Monitor usage**: Set up CloudWatch alerts for cost thresholds

## Troubleshooting

### Common Issues

1. **Role permissions**: Ensure the SageMaker role has access to S3 and ECR
2. **Network access**: SageMaker needs internet access to download dependencies
3. **Container limits**: Increase max_run if training takes longer than expected
4. **Data access**: Ensure your data sources are accessible from SageMaker

### Logs

Training logs are available in:
- Airflow task logs (job submission and status)
- CloudWatch logs (`/aws/sagemaker/TrainingJobs/job-name`)
- SageMaker console training job details

## Security Best Practices

1. **Least privilege**: Only grant necessary permissions to the SageMaker role
2. **Network isolation**: Use VPC endpoints for SageMaker if possible
3. **Encrypt data**: Enable encryption at rest and in transit
4. **Secrets management**: Use AWS Secrets Manager for sensitive data like database passwords

## Cost Estimation

Approximate costs for training (us-east-1):
- `ml.m5.large`: ~$0.115/hour
- `ml.m5.xlarge`: ~$0.230/hour
- `ml.c5.xlarge`: ~$0.192/hour

For a 30-minute training job on `ml.m5.xlarge`: ~$0.115