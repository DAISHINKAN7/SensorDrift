"""
DAG 2: Data Transformation Pipeline
Cleans, validates, and transforms raw sensor data
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
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

def clean_raw_data(**context):
    """Clean and validate raw sensor readings"""
    logger.info("Starting data cleaning process")
    import psycopg2
    import numpy as np

    try:
        conn = psycopg2.connect(
            host='postgres',
            port=5432,
            database='gas_monitoring',
            user='sensordrift',
            password='sensordrift_password'
        )

        cursor = conn.cursor()

        # Get unprocessed raw data from last hour
        cursor.execute("""
            SELECT reading_id, timestamp, gas_type, gas_label,
                   f1, f2, f3, f4, f5, f6, f7, f8
            FROM raw_data.sensor_readings
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            AND reading_id NOT IN (
                SELECT original_reading_id
                FROM processed_data.cleaned_sensor_readings
                WHERE original_reading_id IS NOT NULL
            )
            ORDER BY timestamp DESC
            LIMIT 10000
        """)

        raw_records = cursor.fetchall()
        logger.info(f"Processing {len(raw_records)} raw records")

        if len(raw_records) == 0:
            logger.info("No new records to process")
            cursor.close()
            conn.close()
            return 0

        # Process each record
        cleaned_count = 0
        for record in raw_records:
            reading_id = record[0]
            timestamp = record[1]
            gas_type = record[2]
            gas_label = record[3]
            features = list(record[4:12])  # First 8 features

            # Data quality checks
            has_nulls = any(f is None for f in features)
            is_outlier = False

            if not has_nulls:
                # Simple outlier detection
                feature_array = np.array(features)
                mean_val = np.mean(feature_array)
                std_val = np.std(feature_array)

                if std_val > 0:
                    z_scores = np.abs((feature_array - mean_val) / std_val)
                    is_outlier = np.any(z_scores > 3)

                # Calculate derived features
                concentration_ppm = float(np.mean(features[:4]))  # Simplified
                risk_score = min(100.0, concentration_ppm / 10.0)
                anomaly_score = float(np.max(z_scores)) if std_val > 0 else 0.0

                # Check threshold (example thresholds)
                thresholds = {
                    'Ammonia': 25, 'Acetaldehyde': 50, 'Acetone': 750,
                    'Ethanol': 1000, 'Ethylene': 100, 'Toluene': 50
                }
                threshold = thresholds.get(gas_type, 100)
                threshold_exceeded = concentration_ppm > threshold

                # Insert cleaned record
                cursor.execute("""
                    INSERT INTO processed_data.cleaned_sensor_readings (
                        original_reading_id, timestamp, gas_type, gas_label,
                        feature_1, feature_2, feature_3, feature_4,
                        feature_5, feature_6, feature_7, feature_8,
                        concentration_ppm, risk_score, anomaly_score,
                        threshold_exceeded, is_outlier, is_duplicate,
                        missing_values_count, quality_check_passed
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                              %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    reading_id, timestamp, gas_type, gas_label,
                    *features, concentration_ppm, risk_score, anomaly_score,
                    threshold_exceeded, is_outlier, False, 0, True
                ))

                cleaned_count += 1

        conn.commit()
        logger.info(f"✅ Cleaned {cleaned_count} records")

        cursor.close()
        conn.close()

        # Push to XCom
        context['task_instance'].xcom_push(key='cleaned_records', value=cleaned_count)

        return cleaned_count

    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        raise

def detect_duplicates(**context):
    """Detect and mark duplicate readings"""
    logger.info("Detecting duplicate readings")
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

        # Mark duplicates (same timestamp + gas + features)
        cursor.execute("""
            WITH duplicates AS (
                SELECT reading_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY timestamp, gas_label,
                                      feature_1, feature_2, feature_3
                           ORDER BY reading_id
                       ) as row_num
                FROM processed_data.cleaned_sensor_readings
                WHERE timestamp > NOW() - INTERVAL '1 hour'
            )
            UPDATE processed_data.cleaned_sensor_readings
            SET is_duplicate = TRUE
            WHERE reading_id IN (
                SELECT reading_id FROM duplicates WHERE row_num > 1
            )
        """)

        duplicates_marked = cursor.rowcount
        conn.commit()

        logger.info(f"✅ Marked {duplicates_marked} duplicate records")

        cursor.close()
        conn.close()

        return duplicates_marked

    except Exception as e:
        logger.error(f"Error detecting duplicates: {e}")
        raise

def calculate_aggregates(**context):
    """Calculate hourly aggregates"""
    logger.info("Calculating hourly aggregates")
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

        # Insert hourly statistics
        cursor.execute("""
            INSERT INTO analytics.hourly_statistics (
                hour_timestamp, gas_type,
                total_readings, avg_concentration, max_concentration,
                min_concentration, std_concentration,
                threshold_exceedances, anomaly_count, avg_risk_score
            )
            SELECT
                date_trunc('hour', timestamp) as hour_timestamp,
                gas_type,
                COUNT(*) as total_readings,
                AVG(concentration_ppm) as avg_concentration,
                MAX(concentration_ppm) as max_concentration,
                MIN(concentration_ppm) as min_concentration,
                STDDEV(concentration_ppm) as std_concentration,
                SUM(CASE WHEN threshold_exceeded THEN 1 ELSE 0 END) as threshold_exceedances,
                SUM(CASE WHEN anomaly_score > 2.0 THEN 1 ELSE 0 END) as anomaly_count,
                AVG(risk_score) as avg_risk_score
            FROM processed_data.cleaned_sensor_readings
            WHERE timestamp > NOW() - INTERVAL '2 hours'
            AND quality_check_passed = TRUE
            AND is_duplicate = FALSE
            GROUP BY date_trunc('hour', timestamp), gas_type
            ON CONFLICT DO NOTHING
        """)

        aggregates_created = cursor.rowcount
        conn.commit()

        logger.info(f"✅ Created {aggregates_created} hourly aggregates")

        cursor.close()
        conn.close()

        return aggregates_created

    except Exception as e:
        logger.error(f"Error calculating aggregates: {e}")
        raise

def log_transformation(**context):
    """Log transformation pipeline execution"""
    logger.info("Logging transformation pipeline")
    import psycopg2

    try:
        cleaned_count = context['task_instance'].xcom_pull(
            task_ids='clean_and_validate_data',
            key='cleaned_records'
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
                pipeline_name, dag_id, status, records_processed
            ) VALUES (%s, %s, %s, %s)
        """, (
            'Data Transformation',
            context['dag'].dag_id,
            'success',
            cleaned_count
        ))

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"✅ Transformation pipeline logged")

    except Exception as e:
        logger.error(f"Error logging transformation: {e}")
        pass

# Define DAG
with DAG(
    dag_id='02_data_transformation_pipeline',
    default_args=default_args,
    description='Transforms raw sensor data into clean, processed data',
    schedule_interval='*/15 * * * *',  # Every 15 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['transformation', 'cleaning', 'validation'],
) as dag:

    task_clean = PythonOperator(
        task_id='clean_and_validate_data',
        python_callable=clean_raw_data,
    )

    task_duplicates = PythonOperator(
        task_id='detect_duplicates',
        python_callable=detect_duplicates,
    )

    task_aggregates = PythonOperator(
        task_id='calculate_aggregates',
        python_callable=calculate_aggregates,
    )

    task_log = PythonOperator(
        task_id='log_transformation',
        python_callable=log_transformation,
    )

    # Dependencies
    task_clean >> task_duplicates >> task_aggregates >> task_log
