"""
Kafka Consumer for Sensor Data
Consumes from Kafka and stores in PostgreSQL
"""

import json
import logging
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorDataConsumer:
    """Consumes sensor data from Kafka and stores in PostgreSQL"""

    def __init__(self, kafka_servers='localhost:9092', topic='sensor-readings'):
        self.topic = topic
        self.consumer = None
        self.db_conn = None

        # Database connection parameters
        self.db_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'gas_monitoring'),
            'user': os.getenv('POSTGRES_USER', 'sensordrift'),
            'password': os.getenv('POSTGRES_PASSWORD', 'sensordrift_password')
        }

        # Connect to services
        self.connect_kafka(kafka_servers)
        self.connect_database()

    def connect_kafka(self, kafka_servers):
        """Connect to Kafka with retry logic"""
        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                self.consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=kafka_servers,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    auto_offset_reset='earliest',
                    enable_auto_commit=True,
                    group_id='sensor-consumer-group',
                    max_poll_records=500
                )
                logger.info(f"✅ Connected to Kafka topic '{self.topic}'")
                return
            except Exception as e:
                logger.warning(f"Kafka connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

    def connect_database(self):
        """Connect to PostgreSQL"""
        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                self.db_conn = psycopg2.connect(**self.db_params)
                self.db_conn.autocommit = False
                logger.info(f"✅ Connected to PostgreSQL at {self.db_params['host']}")
                return
            except Exception as e:
                logger.warning(f"Database connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

    def insert_reading(self, message):
        """Insert sensor reading into PostgreSQL"""
        try:
            cursor = self.db_conn.cursor()

            # Extract features
            features = message.get('features', {})
            feature_values = [features.get(f'f{i}') for i in range(1, 129)]

            # Prepare insert query
            insert_query = """
                INSERT INTO raw_data.sensor_readings (
                    timestamp, batch_id, gas_type, gas_label,
                    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                    f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
                    f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
                    f31, f32, f33, f34, f35, f36, f37, f38, f39, f40,
                    f41, f42, f43, f44, f45, f46, f47, f48, f49, f50,
                    f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
                    f61, f62, f63, f64, f65, f66, f67, f68, f69, f70,
                    f71, f72, f73, f74, f75, f76, f77, f78, f79, f80,
                    f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
                    f91, f92, f93, f94, f95, f96, f97, f98, f99, f100,
                    f101, f102, f103, f104, f105, f106, f107, f108, f109, f110,
                    f111, f112, f113, f114, f115, f116, f117, f118, f119, f120,
                    f121, f122, f123, f124, f125, f126, f127, f128,
                    data_source, data_quality_score
                ) VALUES (
                    %s, %s, %s, %s,
                    """ + ', '.join(['%s'] * 128) + """,
                    %s, %s
                )
            """

            # Prepare values
            values = [
                message.get('timestamp'),
                message.get('batch_id'),
                message.get('gas_type'),
                message.get('gas_label'),
            ] + feature_values + [
                message.get('data_source'),
                100.0  # Default quality score
            ]

            cursor.execute(insert_query, values)
            self.db_conn.commit()
            cursor.close()

            return True

        except Exception as e:
            logger.error(f"Error inserting reading: {e}")
            self.db_conn.rollback()
            return False

    def consume_messages(self, batch_size=100):
        """Consume messages from Kafka and store in database"""
        logger.info(f"🚀 Starting to consume messages (batch_size={batch_size})")

        messages_processed = 0
        messages_failed = 0
        batch = []

        try:
            for message in self.consumer:
                try:
                    data = message.value

                    # Insert into database
                    if self.insert_reading(data):
                        messages_processed += 1

                        if messages_processed % 10 == 0:
                            logger.info(f"📊 Processed: {messages_processed} | Failed: {messages_failed}")
                    else:
                        messages_failed += 1

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    messages_failed += 1
                    continue

        except KeyboardInterrupt:
            logger.info("\n⏹️  Consumer stopped by user")
        finally:
            logger.info(f"📈 Summary: Processed={messages_processed}, Failed={messages_failed}")
            self.close()

    def close(self):
        """Close connections"""
        if self.consumer:
            self.consumer.close()
            logger.info("🔌 Kafka consumer closed")

        if self.db_conn:
            self.db_conn.close()
            logger.info("🔌 Database connection closed")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Kafka Sensor Data Consumer')
    parser.add_argument('--kafka-servers', default='localhost:9092', help='Kafka bootstrap servers')
    parser.add_argument('--topic', default='sensor-readings', help='Kafka topic')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')

    args = parser.parse_args()

    try:
        consumer = SensorDataConsumer(
            kafka_servers=args.kafka_servers,
            topic=args.topic
        )

        consumer.consume_messages(batch_size=args.batch_size)

    except Exception as e:
        logger.error(f"❌ Consumer error: {e}")
        raise


if __name__ == '__main__':
    main()
