from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import logging
import sys

def run_training():
    """Wrapper function that handles the import at runtime"""
    # Add the training module to Python path (it's in /opt/airflow/app/training)
    sys.path.insert(0, '/opt/airflow/app/training')
    
    # Import the main function from training directory
    from train import main
    
    # Execute the training
    return main()

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
    )

    retrain_task
