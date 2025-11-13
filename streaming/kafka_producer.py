"""
Kafka Producer for Sensor Data Streaming
Reads from UCI Gas Sensor Dataset and streams to Kafka
"""

import json
import time
import pandas as pd
import numpy as np
from kafka import KafkaProducer
from kafka.errors import KafkaError
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorDataProducer:
    """Streams sensor data from UCI dataset to Kafka"""

    def __init__(self, kafka_servers='localhost:9092', topic='sensor-readings'):
        self.topic = topic
        self.producer = None
        self.connect_kafka(kafka_servers)

        # Load dataset
        self.real_data_path = os.getenv('REAL_DATA_PATH', 'data/combined_sensor_data_clean.csv')
        self.synth_data_path = os.getenv('SYNTH_DATA_PATH', 'data/realistic_synthetic_gas_300k.csv')

        # Gas mapping
        self.gas_mapping = {
            1: "Ethanol",
            2: "Ethylene",
            3: "Ammonia",
            4: "Acetaldehyde",
            5: "Acetone",
            6: "Toluene"
        }

        # Load data
        self.load_data()

    def connect_kafka(self, kafka_servers):
        """Connect to Kafka with retry logic"""
        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=kafka_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    acks='all',
                    retries=3,
                    max_in_flight_requests_per_connection=1
                )
                logger.info(f"✅ Connected to Kafka at {kafka_servers}")
                return
            except Exception as e:
                logger.warning(f"Kafka connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Failed to connect to Kafka after all retries")
                    raise

    def load_data(self):
        """Load sensor data from CSV files"""
        try:
            # Try to load real data
            if os.path.exists(self.real_data_path):
                self.real_data = pd.read_csv(self.real_data_path)
                logger.info(f"✅ Loaded real data: {len(self.real_data)} samples")
            else:
                logger.warning(f"Real data not found at {self.real_data_path}")
                self.real_data = None

            # Try to load synthetic data
            if os.path.exists(self.synth_data_path):
                self.synth_data = pd.read_csv(self.synth_data_path)
                logger.info(f"✅ Loaded synthetic data: {len(self.synth_data)} samples")
            else:
                logger.warning(f"Synthetic data not found at {self.synth_data_path}")
                self.synth_data = None

            # Determine which dataset to use
            if self.real_data is not None:
                self.data = self.real_data
                self.data_source = "real"
            elif self.synth_data is not None:
                self.data = self.synth_data
                self.data_source = "synthetic"
            else:
                raise FileNotFoundError("No dataset found!")

            logger.info(f"Using {self.data_source} dataset with {len(self.data)} samples")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def prepare_message(self, row, index):
        """Prepare sensor reading message"""
        # Extract features
        feature_cols = [f'f{i}' for i in range(1, 129)]
        features = {}

        for col in feature_cols:
            if col in row.index:
                val = row[col]
                features[col] = float(val) if pd.notna(val) else None

        # Get gas info
        gas_label = int(row['label']) if 'label' in row.index else None
        gas_type = self.gas_mapping.get(gas_label, "Unknown")

        # Create message
        message = {
            'reading_id': f"{self.data_source}_{index}_{int(time.time())}",
            'timestamp': datetime.utcnow().isoformat(),
            'gas_type': gas_type,
            'gas_label': gas_label,
            'features': features,
            'data_source': self.data_source,
            'batch_id': f"batch_{int(index / 1000)}",
            'metadata': {
                'producer': 'kafka_producer',
                'version': '1.0',
                'dataset_index': int(index)
            }
        }

        return message

    def send_message(self, message):
        """Send message to Kafka"""
        try:
            future = self.producer.send(self.topic, message)
            record_metadata = future.get(timeout=10)
            return True
        except KafkaError as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def stream_continuous(self, interval=2.0, loop=True):
        """Stream data continuously"""
        logger.info(f"🚀 Starting continuous streaming (interval={interval}s, loop={loop})")

        index = 0
        total_sent = 0

        try:
            while True:
                # Get current row
                if index >= len(self.data):
                    if loop:
                        logger.info("↻ Restarting from beginning of dataset")
                        index = 0
                    else:
                        logger.info("✅ Reached end of dataset")
                        break

                row = self.data.iloc[index]

                # Prepare and send message
                message = self.prepare_message(row, index)

                if self.send_message(message):
                    total_sent += 1
                    if total_sent % 10 == 0:
                        logger.info(f"📊 Sent {total_sent} messages | Current: {message['gas_type']}")

                index += 1
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("\n⏹️  Streaming stopped by user")
        finally:
            logger.info(f"📈 Total messages sent: {total_sent}")
            self.close()

    def stream_batch(self, batch_size=100):
        """Stream a batch of data"""
        logger.info(f"📦 Streaming batch of {batch_size} samples")

        total_sent = 0

        for index in range(min(batch_size, len(self.data))):
            row = self.data.iloc[index]
            message = self.prepare_message(row, index)

            if self.send_message(message):
                total_sent += 1

        logger.info(f"✅ Batch complete: {total_sent}/{batch_size} messages sent")
        return total_sent

    def close(self):
        """Close Kafka producer"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("🔌 Kafka producer closed")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Kafka Sensor Data Producer')
    parser.add_argument('--kafka-servers', default='localhost:9092', help='Kafka bootstrap servers')
    parser.add_argument('--topic', default='sensor-readings', help='Kafka topic')
    parser.add_argument('--interval', type=float, default=2.0, help='Streaming interval in seconds')
    parser.add_argument('--batch', type=int, help='Send batch of N samples and exit')
    parser.add_argument('--loop', action='store_true', help='Loop continuously through dataset')

    args = parser.parse_args()

    try:
        producer = SensorDataProducer(
            kafka_servers=args.kafka_servers,
            topic=args.topic
        )

        if args.batch:
            producer.stream_batch(args.batch)
        else:
            producer.stream_continuous(
                interval=args.interval,
                loop=args.loop
            )

    except Exception as e:
        logger.error(f"❌ Producer error: {e}")
        raise


if __name__ == '__main__':
    main()
