#!/usr/bin/env python3
"""
SageMaker training script entry point
This script adapts the existing training code to work with SageMaker's training environment
"""

import json
import logging
import os
import sys
from pathlib import Path

# Configure logging for SageMaker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def setup_sagemaker_environment():
    """Setup environment variables for SageMaker training"""
    
    # SageMaker specific paths
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    train_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    
    # Set model output path for SageMaker
    os.environ['LOCAL_MODEL_PATH'] = model_dir
    
    # Training parameters from environment
    epochs = int(os.environ.get('EPOCHS', '2'))
    batch_size = int(os.environ.get('BATCH_SIZE', '256'))
    
    logger.info(f"SageMaker Model Dir: {model_dir}")
    logger.info(f"SageMaker Train Dir: {train_dir}")
    logger.info(f"Training epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    return model_dir, train_dir, epochs, batch_size

def main():
    """Main training function for SageMaker"""
    logger.info("Starting SageMaker training job")
    
    # Setup SageMaker environment
    model_dir, train_dir, epochs, batch_size = setup_sagemaker_environment()
    
    # Log environment variables for debugging
    logger.info("Environment variables:")
    for key, value in os.environ.items():
        if any(prefix in key for prefix in ['DB_', 'MINIO_', 'MLFLOW_', 'SM_', 'AWS_']):
            logger.info(f"  {key}: {value}")
    
    # Import and run the main training function
    try:
        from train import main as train_main
        logger.info("Successfully imported training module")
        
        # Run the training
        result = train_main()
        logger.info("Training completed successfully")
        
        # Create a simple model metadata file for SageMaker
        model_metadata = {
            "training_job_name": os.environ.get('SM_TRAINING_JOB_NAME', 'unknown'),
            "model_artifacts_path": model_dir,
            "training_parameters": {
                "epochs": epochs,
                "batch_size": batch_size
            },
            "environment": {
                "python_version": sys.version,
                "sagemaker_region": os.environ.get('AWS_DEFAULT_REGION', 'unknown')
            }
        }
        
        metadata_path = Path(model_dir) / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {metadata_path}")
        return result
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()