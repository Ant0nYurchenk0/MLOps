"""
SageMaker training job configuration and execution
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.pytorch import PyTorch

logger = logging.getLogger(__name__)

class SageMakerTrainingJob:
    """Handles SageMaker training job creation and execution"""
    
    def __init__(self, role_arn: str, region: str = None):
        """
        Initialize SageMaker training job
        
        Args:
            role_arn: IAM role ARN for SageMaker execution
            region: AWS region (defaults to environment or us-east-1)
        """
        self.role_arn = role_arn
        self.region = region or os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        self.session = sagemaker.Session(boto_session=boto3.Session(region_name=self.region))
        self.bucket = self.session.default_bucket()
        
        logger.info(f"Initialized SageMaker training job in region: {self.region}")
        logger.info(f"Using S3 bucket: {self.bucket}")
    
    def create_training_job(self, 
                          training_image: str = None,
                          instance_type: str = 'ml.m5.large',
                          instance_count: int = 1,
                          max_run_seconds: int = 3600,
                          environment_vars: Dict[str, str] = None) -> str:
        """
        Create and submit SageMaker training job
        
        Args:
            training_image: Docker image URI (optional, will use default PyTorch)
            instance_type: EC2 instance type for training
            instance_count: Number of instances
            max_run_seconds: Maximum training time in seconds
            environment_vars: Environment variables for training
            
        Returns:
            Training job name
        """
        
        # Generate unique job name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"quora-training-{timestamp}"
        
        # Default environment variables
        default_env = {
            'DB_HOST': os.environ.get('DB_HOST', 'localhost'),
            'DB_PORT': os.environ.get('DB_PORT', '5432'),
            'DB_NAME': os.environ.get('DB_NAME', 'quora_moderation'),
            'DB_USER': os.environ.get('DB_USER', 'mluser'),
            'DB_PASSWORD': os.environ.get('DB_PASSWORD', 'mlpass'),
            'MINIO_ENDPOINT': os.environ.get('MINIO_ENDPOINT', 'http://localhost:9000'),
            'MINIO_USERNAME': os.environ.get('MINIO_USERNAME', 'minio'),
            'MINIO_PASSWORD': os.environ.get('MINIO_PASSWORD', 'minio123'),
            'MINIO_BUCKET_NAME': os.environ.get('MINIO_BUCKET_NAME', 'embeddings'),
            'EMBEDDINGS_BUCKET': os.environ.get('EMBEDDINGS_BUCKET', 'embeddings'),
            'MODEL_BUCKET': os.environ.get('MODEL_BUCKET', 'models'),
            'MLFLOW_TRACKING_URI': os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
            'PREPROCESSOR_LINK': os.environ.get('PREPROCESSOR_LINK', 'http://localhost:5001'),
            'GLOVE_EMBED_KEY': os.environ.get('GLOVE_EMBED_KEY', 'glove.840B.300d.txt'),
            'WIKI_EMBED_KEY': os.environ.get('WIKI_EMBED_KEY', 'wiki-news-300d-1M.vec'),
            'PARA_EMBED_KEY': os.environ.get('PARA_EMBED_KEY', 'paragram_300_sl999.txt'),
            'EPOCHS': os.environ.get('EPOCHS', '2'),
            'BATCH_SIZE': os.environ.get('BATCH_SIZE', '256'),
        }
        
        # Merge with provided environment variables
        if environment_vars:
            default_env.update(environment_vars)
        
        # Create estimator
        if training_image:
            # Use custom Docker image
            estimator = Estimator(
                image_uri=training_image,
                role=self.role_arn,
                instance_count=instance_count,
                instance_type=instance_type,
                environment=default_env,
                max_run=max_run_seconds,
                sagemaker_session=self.session
            )
        else:
            # Use PyTorch framework
            estimator = PyTorch(
                entry_point='sagemaker_train.py',
                source_dir='.',
                role=self.role_arn,
                instance_count=instance_count,
                instance_type=instance_type,
                framework_version='1.12',
                py_version='py38',
                environment=default_env,
                max_run=max_run_seconds,
                sagemaker_session=self.session
            )
        
        logger.info(f"Created SageMaker estimator for job: {job_name}")
        logger.info(f"Instance type: {instance_type}")
        logger.info(f"Instance count: {instance_count}")
        
        # Submit training job
        try:
            estimator.fit(job_name=job_name, wait=False)
            logger.info(f"Successfully submitted SageMaker training job: {job_name}")
            return job_name
            
        except Exception as e:
            logger.error(f"Failed to submit SageMaker training job: {str(e)}")
            raise
    
    def wait_for_training_job(self, job_name: str, check_interval: int = 30) -> Dict[str, Any]:
        """
        Wait for training job to complete and return status
        
        Args:
            job_name: Name of the training job
            check_interval: Seconds between status checks
            
        Returns:
            Training job description
        """
        
        sm_client = boto3.client('sagemaker', region_name=self.region)
        
        logger.info(f"Waiting for training job {job_name} to complete...")
        
        while True:
            try:
                response = sm_client.describe_training_job(TrainingJobName=job_name)
                status = response['TrainingJobStatus']
                
                logger.info(f"Training job {job_name} status: {status}")
                
                if status == 'Completed':
                    logger.info(f"Training job {job_name} completed successfully")
                    return response
                elif status == 'Failed':
                    failure_reason = response.get('FailureReason', 'Unknown')
                    logger.error(f"Training job {job_name} failed: {failure_reason}")
                    raise Exception(f"Training job failed: {failure_reason}")
                elif status == 'Stopped':
                    logger.warning(f"Training job {job_name} was stopped")
                    return response
                
                # Wait before checking again
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error checking training job status: {str(e)}")
                raise
    
    def get_training_job_logs(self, job_name: str) -> str:
        """
        Get training job logs from CloudWatch
        
        Args:
            job_name: Name of the training job
            
        Returns:
            Training job logs
        """
        
        logs_client = boto3.client('logs', region_name=self.region)
        log_group = f'/aws/sagemaker/TrainingJobs/{job_name}'
        
        try:
            response = logs_client.describe_log_streams(logGroupName=log_group)
            log_streams = response['logStreams']
            
            all_logs = []
            for stream in log_streams:
                stream_name = stream['logStreamName']
                events = logs_client.get_log_events(
                    logGroupName=log_group,
                    logStreamName=stream_name
                )
                
                for event in events['events']:
                    all_logs.append(event['message'])
            
            return '\n'.join(all_logs)
            
        except Exception as e:
            logger.error(f"Error retrieving logs for job {job_name}: {str(e)}")
            return f"Error retrieving logs: {str(e)}"

def run_sagemaker_training(role_arn: str, 
                         instance_type: str = 'ml.m5.large',
                         wait_for_completion: bool = True,
                         environment_vars: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Convenience function to run SageMaker training job
    
    Args:
        role_arn: IAM role ARN for SageMaker
        instance_type: EC2 instance type
        wait_for_completion: Whether to wait for job completion
        environment_vars: Additional environment variables
        
    Returns:
        Training job result
    """
    
    trainer = SageMakerTrainingJob(role_arn)
    
    # Submit training job
    job_name = trainer.create_training_job(
        instance_type=instance_type,
        environment_vars=environment_vars
    )
    
    result = {
        'job_name': job_name,
        'status': 'InProgress'
    }
    
    if wait_for_completion:
        # Wait for completion
        job_description = trainer.wait_for_training_job(job_name)
        result['status'] = job_description['TrainingJobStatus']
        result['job_description'] = job_description
        
        # Get logs
        result['logs'] = trainer.get_training_job_logs(job_name)
    
    return result