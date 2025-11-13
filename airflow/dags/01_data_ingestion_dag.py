"""
DAG 1: Data Ingestion Pipeline
Ingests data from various sources into PostgreSQL
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'sensordrift',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

def start_kafka_producer(**context):
    """Start Kafka producer to stream sensor data"""
    logger.info("Starting Kafka producer for data ingestion")
    import subprocess
    import sys

    try:
        # Start producer in batch mode
        result = subprocess.run([
            sys.executable,
            '/opt/airflow/project/streaming/kafka_producer.py',
            '--kafka-servers', 'kafka:29092',
            '--batch', '1000',  # Stream 1000 samples
            '--topic', 'sensor-readings'
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logger.info(f"✅ Kafka producer completed successfully")
            logger.info(result.stdout)
        else:
            logger.error(f"❌ Kafka producer failed: {result.stderr}")
            raise Exception(f"Kafka producer failed: {result.stderr}")

    except Exception as e:
        logger.error(f"Error starting Kafka producer: {e}")
        raise

def start_kafka_consumer(**context):
    """Start Kafka consumer to store data in PostgreSQL"""
    logger.info("Starting Kafka consumer for data storage")
    import subprocess
    import sys
    import signal
    import time

    try:
        # Start consumer process
        process = subprocess.Popen([
            sys.executable,
            '/opt/airflow/project/streaming/kafka_consumer.py',
            '--kafka-servers', 'kafka:29092',
            '--topic', 'sensor-readings'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Let it run for 60 seconds to consume messages
        time.sleep(60)

        # Stop consumer
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)

        logger.info(f"✅ Kafka consumer completed")
        logger.info(stdout)

        if stderr and 'Error' in stderr:
            logger.warning(f"Consumer warnings: {stderr}")

    except Exception as e:
        logger.error(f"Error running Kafka consumer: {e}")
        raise

def verify_ingestion(**context):
    """Verify data was ingested successfully"""
    logger.info("Verifying data ingestion")
    import psycopg2

    try:
        conn = psycopg2.connect(
            host='postgres',
            port=5432,
            database='gas_monitoring',
            user='sensordrift',
            password='sensordrift_password'
        )

        cursor = conn.cursor()

        # Count records ingested in last hour
        cursor.execute("""
            SELECT COUNT(*) as count,
                   MIN(timestamp) as earliest,
                   MAX(timestamp) as latest
            FROM raw_data.sensor_readings
            WHERE ingestion_timestamp > NOW() - INTERVAL '1 hour'
        """)

        result = cursor.fetchone()
        count, earliest, latest = result

        logger.info(f"✅ Ingestion verified: {count} records")
        logger.info(f"   Time range: {earliest} to {latest}")

        if count == 0:
            logger.warning("⚠️  No records ingested in last hour")

        cursor.close()
        conn.close()

        # Push metrics to XCom
        context['task_instance'].xcom_push(key='records_ingested', value=count)

        return count

    except Exception as e:
        logger.error(f"Error verifying ingestion: {e}")
        raise

def log_pipeline_execution(**context):
    """Log pipeline execution to monitoring table"""
    logger.info("Logging pipeline execution")
    import psycopg2

    try:
        # Get ingestion count from previous task
        records_count = context['task_instance'].xcom_pull(
            task_ids='verify_ingestion',
            key='records_ingested'
        ) or 0

        conn = psycopg2.connect(
            host='postgres',
            port=5432,
            database='gas_monitoring',
            user='sensordrift',
            password='sensordrift_password'
        )

        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO monitoring.pipeline_executions (
                pipeline_name, dag_id, task_id, status,
                records_processed, execution_metadata
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            'Data Ingestion',
            context['dag'].dag_id,
            'data_ingestion_complete',
            'success',
            records_count,
            '{"source": "kafka", "batch_mode": true}'
        ))

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"✅ Pipeline execution logged")

    except Exception as e:
        logger.error(f"Error logging execution: {e}")
        # Don't fail the pipeline for logging errors
        pass

# Define DAG
with DAG(
    dag_id='01_data_ingestion_pipeline',
    default_args=default_args,
    description='Ingests sensor data from Kafka stream into PostgreSQL',
    schedule_interval='*/30 * * * *',  # Every 30 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'kafka', 'postgresql'],
) as dag:

    # Task 1: Start Kafka Producer
    task_produce_data = PythonOperator(
        task_id='produce_sensor_data',
        python_callable=start_kafka_producer,
        execution_timeout=timedelta(minutes=10),
    )

    # Task 2: Start Kafka Consumer
    task_consume_data = PythonOperator(
        task_id='consume_and_store_data',
        python_callable=start_kafka_consumer,
        execution_timeout=timedelta(minutes=5),
    )

    # Task 3: Verify Ingestion
    task_verify = PythonOperator(
        task_id='verify_ingestion',
        python_callable=verify_ingestion,
    )

    # Task 4: Log Execution
    task_log = PythonOperator(
        task_id='log_pipeline_execution',
        python_callable=log_pipeline_execution,
    )

    # Define task dependencies
    task_produce_data >> task_consume_data >> task_verify >> task_log
