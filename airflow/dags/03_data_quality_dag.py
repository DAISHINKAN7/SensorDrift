"""
DAG 3: Data Quality Monitoring
Monitors data quality and generates alerts
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'sensordrift',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

def check_data_completeness(**context):
    """Check for data completeness"""
    logger.info("Checking data completeness")
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

        # Check completeness of recent data
        cursor.execute("""
            SELECT
                COUNT(*) as total_records,
                SUM(CASE WHEN quality_check_passed THEN 1 ELSE 0 END) as passed_records,
                SUM(missing_values_count) as total_missing_values,
                ROUND(AVG(CASE WHEN quality_check_passed THEN 100.0 ELSE 0.0 END), 2) as completeness_pct
            FROM processed_data.cleaned_sensor_readings
            WHERE timestamp > NOW() - INTERVAL '1 hour'
        """)

        result = cursor.fetchone()
        total, passed, missing, completeness = result

        logger.info(f"Data Completeness: {completeness}%")
        logger.info(f"  Total records: {total}")
        logger.info(f"  Passed QC: {passed}")
        logger.info(f"  Missing values: {missing}")

        # Log metric
        cursor.execute("""
            INSERT INTO monitoring.data_quality_metrics (
                table_name, metric_name, metric_value,
                threshold_min, is_within_threshold, details
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            'cleaned_sensor_readings',
            'data_completeness',
            completeness,
            95.0,  # Threshold
            completeness >= 95.0,
            f'{{"total": {total}, "passed": {passed}, "missing": {missing}}}'
        ))

        conn.commit()
        cursor.close()
        conn.close()

        # Push to XCom
        context['task_instance'].xcom_push(key='completeness', value=completeness)

        return completeness

    except Exception as e:
        logger.error(f"Error checking completeness: {e}")
        raise

def check_data_freshness(**context):
    """Check data freshness (how recent is the data)"""
    logger.info("Checking data freshness")
    import psycopg2
    from datetime import datetime

    try:
        conn = psycopg2.connect(
            host='postgres',
            port=5432,
            database='gas_monitoring',
            user='sensordrift',
            password='sensordrift_password'
        )

        cursor = conn.cursor()

        # Get latest timestamp
        cursor.execute("""
            SELECT MAX(timestamp) as latest_timestamp
            FROM raw_data.sensor_readings
        """)

        latest = cursor.fetchone()[0]

        if latest:
            age_minutes = (datetime.now(latest.tzinfo) - latest).total_seconds() / 60
            is_fresh = age_minutes < 60  # Fresh if less than 60 minutes old

            logger.info(f"Latest data: {latest} ({age_minutes:.1f} minutes ago)")

            # Log metric
            cursor.execute("""
                INSERT INTO monitoring.data_quality_metrics (
                    table_name, metric_name, metric_value,
                    threshold_max, is_within_threshold, details
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                'sensor_readings',
                'data_freshness_minutes',
                age_minutes,
                60.0,
                is_fresh,
                f'{{"latest_timestamp": "{latest}"}}'
            ))

            conn.commit()
        else:
            logger.warning("No data found in sensor_readings table")
            age_minutes = None

        cursor.close()
        conn.close()

        return age_minutes

    except Exception as e:
        logger.error(f"Error checking freshness: {e}")
        raise

def check_data_consistency(**context):
    """Check data consistency (valid ranges, no corruption)"""
    logger.info("Checking data consistency")
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

        # Check for valid gas labels
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN gas_label BETWEEN 0 AND 5 THEN 1 ELSE 0 END) as valid_labels,
                ROUND(AVG(CASE WHEN gas_label BETWEEN 0 AND 5 THEN 100.0 ELSE 0.0 END), 2) as consistency_pct
            FROM processed_data.cleaned_sensor_readings
            WHERE timestamp > NOW() - INTERVAL '1 hour'
        """)

        result = cursor.fetchone()
        total, valid, consistency = result

        logger.info(f"Data Consistency: {consistency}%")
        logger.info(f"  Valid labels: {valid}/{total}")

        # Log metric
        cursor.execute("""
            INSERT INTO monitoring.data_quality_metrics (
                table_name, metric_name, metric_value,
                threshold_min, is_within_threshold, details
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            'cleaned_sensor_readings',
            'data_consistency',
            consistency,
            98.0,
            consistency >= 98.0,
            f'{{"total": {total}, "valid": {valid}}}'
        ))

        conn.commit()
        cursor.close()
        conn.close()

        return consistency

    except Exception as e:
        logger.error(f"Error checking consistency: {e}")
        raise

def generate_quality_report(**context):
    """Generate summary quality report"""
    logger.info("Generating data quality summary report")
    import psycopg2

    try:
        # Get metrics from previous tasks
        completeness = context['task_instance'].xcom_pull(
            task_ids='check_completeness',
            key='completeness'
        ) or 0

        conn = psycopg2.connect(
            host='postgres',
            port=5432,
            database='gas_monitoring',
            user='sensordrift',
            password='sensordrift_password'
        )

        cursor = conn.cursor()

        # Get recent metrics
        cursor.execute("""
            SELECT metric_name, metric_value, is_within_threshold
            FROM monitoring.data_quality_metrics
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            ORDER BY timestamp DESC
        """)

        metrics = cursor.fetchall()

        logger.info("=" * 60)
        logger.info("DATA QUALITY REPORT")
        logger.info("=" * 60)

        for metric_name, value, within_threshold in metrics:
            status = "✅ PASS" if within_threshold else "❌ FAIL"
            logger.info(f"{metric_name}: {value:.2f} {status}")

        logger.info("=" * 60)

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"Error generating quality report: {e}")
        raise

# Define DAG
with DAG(
    dag_id='03_data_quality_monitoring',
    default_args=default_args,
    description='Monitors data quality and generates quality reports',
    schedule_interval='0 * * * *',  # Every hour
    start_date=days_ago(1),
    catchup=False,
    tags=['data-quality', 'monitoring', 'alerts'],
) as dag:

    task_completeness = PythonOperator(
        task_id='check_completeness',
        python_callable=check_data_completeness,
    )

    task_freshness = PythonOperator(
        task_id='check_freshness',
        python_callable=check_data_freshness,
    )

    task_consistency = PythonOperator(
        task_id='check_consistency',
        python_callable=check_data_consistency,
    )

    task_report = PythonOperator(
        task_id='generate_quality_report',
        python_callable=generate_quality_report,
    )

    # Dependencies
    [task_completeness, task_freshness, task_consistency] >> task_report
