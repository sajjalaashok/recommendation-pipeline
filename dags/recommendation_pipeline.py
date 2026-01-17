
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import os

# Definition of default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Determine the project root
# If running in Docker as per our instructions, the project is mounted at /opt/airflow/project_root
# Otherwise, we assume standard relative structure
if os.path.exists("/opt/airflow/project_root"):
    PROJECT_ROOT = "/opt/airflow/project_root"
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define virtual environment path inside the project root
VENV_PATH = f"{PROJECT_ROOT}/airflow_env"
PYTHON_BIN = f"{VENV_PATH}/bin/python"

with DAG(
    'recommendation_system_pipeline',
    default_args=default_args,
    description='End-to-end recommendation system pipeline',
    schedule=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['recommendation', 'mlops'],
) as dag:

    # Task 1: Check if raw data exists
    validate_data = BashOperator(
        task_id='validate_data_availability',
        bash_command=f'test -f {PROJECT_ROOT}/raw_zone/*transaction*.csv || (echo "Transaction data missing" && exit 1)',
        cwd=PROJECT_ROOT
    )

    # Task 2: Setup Virtual Environment & Install Dependencies
    # We create a venv if it doesn't exist, then install requirements.
    # This solves the permission error by writing to our mounted project dir instead of system paths.
    setup_env = BashOperator(
        task_id='setup_environment',
        bash_command=f'''
        if [ ! -d "{VENV_PATH}" ]; then
            python -m venv {VENV_PATH}
        fi
        {PYTHON_BIN} -m pip install --upgrade pip
        {PYTHON_BIN} -m pip install mlflow scikit-learn pandas
        ''',
        cwd=PROJECT_ROOT
    )

    # Task 3: Data Transformation (Using Venv Python)
    transform_data = BashOperator(
        task_id='transform_data',
        bash_command=f'{PYTHON_BIN} transform.py',
        cwd=PROJECT_ROOT
    )

    # Task 4: Feature Store Creation (Using Venv Python)
    build_features = BashOperator(
        task_id='build_feature_store',
        bash_command=f'{PYTHON_BIN} feature_store.py',
        cwd=PROJECT_ROOT
    )

    # Task 5: Model Training (Using Venv Python)
    train_model = BashOperator(
        task_id='train_model',
        bash_command=f'{PYTHON_BIN} train.py',
        cwd=PROJECT_ROOT
    )

    # Task Dependencies
    validate_data >> setup_env >> transform_data >> build_features >> train_model
