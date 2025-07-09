from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import logging
import sys

def run_training():
    """Call retraining microservice endpoint"""
    import requests
    
    try:
        logger.info("Calling retraining microservice...")
        
        # Call the retraining service
        response = requests.post("http://retraining:8002/train", timeout=7200)  # 2 hour timeout
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Training completed: {result}")
            return result
        else:
            logger.error(f"Training failed with status {response.status_code}: {response.text}")
            raise Exception(f"Training service returned {response.status_code}: {response.text}")
            
    except Exception as e:
        logger.error(f"Failed to call training service: {str(e)}")
        raise

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    # If a run fails, retry up to 1 time after 5 minutes
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retrain_quora_models",
    default_args=default_args,
    description="Retrain Quora Moderation models daily using new DB data",
    schedule_interval="0 0 * * *",  # daily at midnight UTC
    start_date=datetime(2025, 6, 1),
    catchup=False,
    tags=["quora", "retrain"],
) as dag:

    # Add debug logging and better error handling
    logger.info("Creating retrain_quora_models DAG")
    
    # The `bash_command` below assumes that inside the Airflow container,
    # your code lives under `/opt/airflow/app`.
    # Adjust the path to your training script if it lives somewhere else.
    retrain_task = PythonOperator(
        task_id='run_train_models_py',
        python_callable=run_training,
        execution_timeout=timedelta(hours=2),  # Allow 2 hours for training
        retries=2,  # Increase retries
        retry_delay=timedelta(minutes=10),  # Longer retry delay
    )

    retrain_task
