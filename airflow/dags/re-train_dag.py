from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

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

    # The `bash_command` below assumes that inside the Airflow container,
    # your code lives under `/opt/airflow/app`.
    # Adjust the path to your training script if it lives somewhere else.
    retrain_task = BashOperator(
        task_id="run_train_models_py",
        bash_command=("python /opt/airflow/app/training/train.py"),
    )

    retrain_task
