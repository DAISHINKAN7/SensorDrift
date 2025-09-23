import pandas as pd
import numpy as np
import os
import time
import threading
from datetime import datetime
import json
import queue
from collections import deque, Counter
import random
import asyncio
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, stddev, when, lit, current_timestamp, struct
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("GasSensorStreamingDashboard") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Define schema for the synthetic gas sensor data
schema = StructType([
    StructField("label", StringType(), False),
    *[StructField(f"f{i}", DoubleType(), True) for i in range(1, 129)],
    StructField("timestamp", TimestampType(), True),
    StructField("batch_id", StringType(), True)
])

# Alternative 1: Simple Python-based Streaming (No Java required)
class PythonStreamingProcessor:    
    def __init__(self, csv_path, batch_size=50, processing_interval=2.0):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.processing_interval = processing_interval
        self.data = None
        self.current_index = 0
        self.is_streaming = False
        self.batch_queue = queue.Queue()
        
        # Analytics
        self.total_processed = 0
        self.gas_counter = Counter()
        self.batch_stats = []
        
    def load_data(self):
        """Load the CSV data"""
        print(f"📂 Loading data from: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(self.data):,} samples")
        print(f"🏷️ Gas types: {list(self.data['label'].unique())}")
        return self.data
    
    def generate_batch(self):
        """Generate a batch of samples"""
        if self.current_index >= len(self.data):
            self.current_index = 0
            print("🔄 Dataset loop completed")
        
        batch_data = []
        for _ in range(self.batch_size):
            if self.current_index >= len(self.data):
                break
                
            sample = self.data.iloc[self.current_index].copy()
            sample['timestamp'] = datetime.now().isoformat()
            sample['batch_id'] = len(self.batch_stats)
            
            # Add noise
            feature_cols = [col for col in sample.index if col.startswith('f')]
            for col in feature_cols:
                noise = np.random.normal(0, abs(sample[col]) * 0.001)
                sample[col] += noise
            
            batch_data.append(sample.to_dict())
            self.current_index += 1
        
        return batch_data
    
    def process_batch(self, batch):
        """Process a batch of data"""
        if not batch:
            return
        
        batch_df = pd.DataFrame(batch)
        batch_time = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n{'='*70}")
        print(f"🕐 Batch #{len(self.batch_stats)} at {batch_time} | Size: {len(batch)} samples")
        print(f"{'='*70}")
        
        # Gas distribution
        gas_dist = batch_df['label'].value_counts()
        print("🏷️ Gas Distribution:")
        for gas, count in gas_dist.items():
            percentage = (count / len(batch)) * 100
            self.gas_counter[gas] += count
            print(f"   {gas:<15}: {count:>3} ({percentage:>5.1f}%)")
        
        # Feature statistics
        print("\n📊 Feature Statistics (F1-F5):")
        feature_stats = batch_df.groupby('label')[['f1', 'f2', 'f3', 'f4', 'f5']].agg(['mean', 'std'])
        print(feature_stats.round(2))
        
        # Anomaly detection
        print("\n🚨 Anomaly Detection:")
        anomalies = []
        
        # Define thresholds based on your data characteristics
        for _, row in batch_df.iterrows():
            anomaly_score = 0
            reasons = []
            
            # Check extreme values
            if row['f1'] > 80000 or row['f1'] < 5000:
                anomaly_score += 1
                reasons.append("F1_extreme")
            
            if row['f2'] > 8 or row['f2'] < 0.5:
                anomaly_score += 1
                reasons.append("F2_extreme")
            
            if anomaly_score > 0:
                anomalies.append({
                    'gas': row['label'],
                    'timestamp': row['timestamp'],
                    'score': anomaly_score,
                    'reasons': reasons,
                    'f1': row['f1'],
                    'f2': row['f2']
                })
        
        if anomalies:
            print(f"   Found {len(anomalies)} anomalies:")
            for anomaly in anomalies[:3]:  # Show first 3
                print(f"   ⚠️ {anomaly['gas']} | Score: {anomaly['score']} | F1: {anomaly['f1']:.1f} | Reasons: {anomaly['reasons']}")
        else:
            print("   ✅ No anomalies detected")
        
        # Running totals
        self.total_processed += len(batch)
        print(f"\n📈 Running Totals:")
        print(f"   Total processed: {self.total_processed:,} samples")
        print(f"   Overall gas distribution:")
        total_gases = sum(self.gas_counter.values())
        for gas, count in self.gas_counter.most_common():
            percentage = (count / total_gases) * 100
            print(f"     {gas:<15}: {count:>6,} ({percentage:>5.1f}%)")
        
        # Store batch statistics
        self.batch_stats.append({
            'timestamp': batch_time,
            'size': len(batch),
            'gas_distribution': dict(gas_dist),
            'anomalies': len(anomalies)
        })
    
    def start_streaming(self, duration=None):
        """Start the streaming process"""
        if not hasattr(self, 'data') or self.data is None:
            self.load_data()
        
        self.is_streaming = True
        start_time = time.time()
        
        print(f"🚀 Starting Python-based streaming...")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"⏱️ Processing interval: {self.processing_interval}s")
        if duration:
            print(f"⏰ Duration: {duration} seconds")
        else:
            print("⏰ Duration: Infinite (Ctrl+C to stop)")
        
        try:
            while self.is_streaming:
                batch_start = time.time()
                
                # Generate and process batch
                batch = self.generate_batch()
                self.process_batch(batch)
                
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    print(f"\n⏰ Streaming duration ({duration}s) completed!")
                    break
                
                # Wait for next batch
                elapsed = time.time() - batch_start
                sleep_time = max(0, self.processing_interval - elapsed)
                if sleep_time > 0:
                    print(f"⏸️ Waiting {sleep_time:.1f}s for next batch...")
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print(f"\n🛑 Streaming stopped by user")
        finally:
            self.is_streaming = False
            self.show_final_summary()
    
    def show_final_summary(self):
        """Show final streaming summary"""
        print(f"\n{'='*70}")
        print("📋 FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"Total batches processed: {len(self.batch_stats)}")
        print(f"Total samples processed: {self.total_processed:,}")
        
        if self.batch_stats:
            avg_batch_size = sum(b['size'] for b in self.batch_stats) / len(self.batch_stats)
            total_anomalies = sum(b['anomalies'] for b in self.batch_stats)
            print(f"Average batch size: {avg_batch_size:.1f}")
            print(f"Total anomalies detected: {total_anomalies}")
        
        print(f"\nFinal gas distribution:")
        total = sum(self.gas_counter.values())
        for gas, count in self.gas_counter.most_common():
            percentage = (count / total) * 100
            print(f"  {gas:<15}: {count:>6,} ({percentage:>5.1f}%)")

# Alternative 2: Fix Spark Setup
class SparkFixHelper:
    """Helper to fix Spark/Java issues"""
    
    @staticmethod
    def check_java():
        """Check if Java is installed"""
        import subprocess
        try:
            result = subprocess.run(['java', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Java is installed")
                print(result.stderr.split('\n')[0])
                return True
            else:
                print("❌ Java not found")
                return False
        except FileNotFoundError:
            print("❌ Java not found")
            return False
    
    @staticmethod
    def install_java_instructions():
        """Show Java installation instructions"""
        print("\n🔧 To fix Spark issues, install Java:")
        print("\nFor macOS:")
        print("  brew install openjdk@11")
        print('  echo \'export PATH="/usr/local/opt/openjdk@11/bin:$PATH"\' >> ~/.zshrc')
        print("  source ~/.zshrc")
        
        print("\nFor Ubuntu/Debian:")
        print("  sudo apt update")
        print("  sudo apt install openjdk-11-jdk")
        
        print("\nFor Windows:")
        print("  Download from https://adoptopenjdk.net/")
        print("  Or use: winget install Microsoft.OpenJDK.11")
        
        print("\nAfter installing Java, restart your terminal and try again.")
    
    @staticmethod
    def find_free_port(start_port=9999):
        """Find a free port"""
        import socket
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        return None

# Alternative 3: Async Streaming (Modern Python)
class AsyncStreamingProcessor:
    """Async-based streaming using Python asyncio"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.current_index = 0
        
    async def load_data(self):
        """Async data loading"""
        print(f"📂 Loading data asynchronously...")
        # In real scenario, this could be from database, API, etc.
        self.data = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(self.data):,} samples")
        return self.data
    
    async def generate_sample(self):
        """Generate a single sample asynchronously"""
        if self.current_index >= len(self.data):
            self.current_index = 0
        
        sample = self.data.iloc[self.current_index].copy()
        sample['timestamp'] = datetime.now().isoformat()
        
        # Add noise
        feature_cols = [col for col in sample.index if col.startswith('f')]
        for col in feature_cols:
            noise = np.random.normal(0, abs(sample[col]) * 0.001)
            sample[col] += noise
        
        self.current_index += 1
        return sample.to_dict()
    
    async def process_sample(self, sample):
        """Process individual sample"""
        # Simulate processing time
        await asyncio.sleep(0.01)
        
        # Simple processing
        return {
            'processed_at': datetime.now().isoformat(),
            'gas': sample['label'],
            'sensor_sum': sum(sample[f'f{i}'] for i in range(1, 6)),
            'anomaly': sample['f1'] > 50000 or sample['f1'] < 10000
        }
    
    async def stream_async(self, rate=10.0, duration=60):
        """Async streaming"""
        await self.load_data()
        
        interval = 1.0 / rate
        start_time = time.time()
        processed = 0
        gas_counter = Counter()
        anomalies = 0
        
        print(f"🚀 Starting async streaming at {rate} samples/sec for {duration}s")
        
        try:
            while (time.time() - start_time) < duration:
                sample = await self.generate_sample()
                result = await self.process_sample(sample)
                
                processed += 1
                gas_counter[result['gas']] += 1
                if result['anomaly']:
                    anomalies += 1
                
                # Show progress every 50 samples
                if processed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate_actual = processed / elapsed
                    print(f"📊 Processed: {processed} | Rate: {rate_actual:.1f}/sec | "
                          f"Anomalies: {anomalies} | Current: {result['gas']}")
                
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("🛑 Async streaming stopped")
        
        print(f"\n✅ Async streaming completed:")
        print(f"   Processed: {processed} samples")
        print(f"   Anomalies: {anomalies}")
        print(f"   Gas distribution: {dict(gas_counter)}")

# PySpark Streaming Dashboard
class PySparkStreamingDashboard:
    def __init__(self, csv_path, batch_size=50, processing_interval=3.0):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.processing_interval = processing_interval
        self.spark = spark
        self.total_processed = 0
        self.gas_counter = Counter()
        self.batch_stats = []

    def load_data(self):
        """Load the CSV data into a Spark DataFrame"""
        print(f"📂 Loading data from: {self.csv_path}")
        df = self.spark.read.csv(self.csv_path, schema=schema, header=True)
        print(f"✅ Loaded {df.count():,} samples")
        print(f"🏷️ Gas types: {df.select('label').distinct().collect()}")
        return df

    def simulate_stream(self):
        """Simulate streaming by reading the CSV file in batches"""
        # Load data as a static DataFrame
        static_df = self.load_data()
        
        # Create a streaming DataFrame by simulating batches
        # For demonstration, we'll use a temporary directory to write batches
        temp_dir = "/tmp/gas_sensor_stream"
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # Split data into batches
        total_rows = static_df.count()
        num_batches = max(1, total_rows // self.batch_size)
        batch_dfs = static_df.randomSplit([1.0 / num_batches] * num_batches, seed=42)

        # Write batches to temporary files to simulate streaming
        for i, batch_df in enumerate(batch_dfs):
            batch_df.write.mode("overwrite").csv(f"{temp_dir}/batch_{i}")

        # Create a streaming DataFrame
        stream_df = self.spark.readStream.schema(schema).csv(temp_dir)
        return stream_df

    def process_stream(self, stream_df):
        """Process the streaming DataFrame with anomaly detection"""
        # Add timestamp and batch_id
        stream_df = stream_df.withColumn("timestamp", current_timestamp()) \
                            .withColumn("batch_id", lit(str(len(self.batch_stats))))

        # Anomaly detection
        anomaly_condition = (col("f1") > 80000) | (col("f1") < 5000) | \
                           (col("f2") > 8) | (col("f2") < 0.5)
        stream_df = stream_df.withColumn("anomaly_score", 
                                        when(anomaly_condition, 1).otherwise(0)) \
                           .withColumn("anomaly_reasons",
                                       when(col("f1") > 80000, "F1_high")
                                       .when(col("f1") < 5000, "F1_low")
                                       .when(col("f2") > 8, "F2_high")
                                       .when(col("f2") < 0.5, "F2_low"))

        # Aggregate statistics
        gas_dist = stream_df.groupBy("batch_id", "label").agg(
            count("*").alias("count"),
            (count("*") / self.batch_size * 100).alias("percentage")
        )

        feature_stats = stream_df.groupBy("batch_id", "label").agg(
            mean("f1").alias("f1_mean"), stddev("f1").alias("f1_std"),
            mean("f2").alias("f2_mean"), stddev("f2").alias("f2_std"),
            mean("f3").alias("f3_mean"), stddev("f3").alias("f3_std"),
            mean("f4").alias("f4_mean"), stddev("f4").alias("f4_std"),
            mean("f5").alias("f5_mean"), stddev("f5").alias("f5_std")
        )

        anomalies = stream_df.filter(col("anomaly_score") > 0) \
                           .select("batch_id", "label", "timestamp", "anomaly_score", 
                                   "anomaly_reasons", "f1", "f2")

        # Write streams to console
        gas_dist_query = gas_dist.writeStream \
            .outputMode("complete") \
            .format("console") \
            .option("truncate", False) \
            .trigger(processingTime=f"{self.processing_interval} seconds") \
            .start()

        feature_stats_query = feature_stats.writeStream \
            .outputMode("complete") \
            .format("console") \
            .option("truncate", False) \
            .trigger(processingTime=f"{self.processing_interval} seconds") \
            .start()

        anomalies_query = anomalies.writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", False) \
            .trigger(processingTime=f"{self.processing_interval} seconds") \
            .start()

        # Update running totals
        def update_totals(batch_df, batch_id):
            batch_count = batch_df.count()
            self.total_processed += batch_count
            gas_counts = batch_df.groupBy("label").count().collect()
            for row in gas_counts:
                self.gas_counter[row["label"]] += row["count"]
            anomaly_count = batch_df.filter(col("anomaly_score") > 0).count()
            self.batch_stats.append({
                "batch_id": batch_id,
                "size": batch_count,
                "anomalies": anomaly_count
            })

        # Process each micro-batch
        stream_df.writeStream \
            .foreachBatch(update_totals) \
            .outputMode("append") \
            .trigger(processingTime=f"{self.processing_interval} seconds") \
            .start()

        return [gas_dist_query, feature_stats_query, anomalies_query]

    def start_streaming(self, duration=120):
        """Start the PySpark streaming dashboard"""
        print(f"🚀 Starting PySpark-based streaming dashboard...")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"⏱️ Processing interval: {self.processing_interval}s")
        print(f"⏰ Duration: {duration} seconds")

        stream_df = self.simulate_stream()
        queries = self.process_stream(stream_df)

        # Wait for duration or interruption
        start_time = time.time()
        try:
            while any(query.isActive() for query in queries):
                if (time.time() - start_time) >= duration:
                    print(f"\n⏰ Streaming duration ({duration}s) completed!")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🛑 Streaming stopped by user")
        finally:
            for query in queries:
                query.stop()
            self.show_final_summary()

    def show_final_summary(self):
        """Show final streaming summary"""
        print(f"\n{'='*70}")
        print("📋 FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"Total batches processed: {len(self.batch_stats)}")
        print(f"Total samples processed: {self.total_processed:,}")
        
        if self.batch_stats:
            avg_batch_size = sum(b["size"] for b in self.batch_stats) / len(self.batch_stats)
            total_anomalies = sum(b["anomalies"] for b in self.batch_stats)
            print(f"Average batch size: {avg_batch_size:.1f}")
            print(f"Total anomalies detected: {total_anomalies}")
        
        print(f"\nFinal gas distribution:")
        total = sum(self.gas_counter.values())
        for gas, count in self.gas_counter.most_common():
            percentage = (count / total) * 100
            print(f"  {gas:<15}: {count:>6,} ({percentage:>5.1f}%)")

# MAIN USAGE FUNCTIONS

def run_python_streaming(csv_path, duration=120):
    """Run Python-based streaming (NO JAVA REQUIRED)"""
    print("🐍 Starting Python-based streaming (no Java required)")
    
    processor = PythonStreamingProcessor(
        csv_path=csv_path,
        batch_size=30,           # Process 30 samples per batch
        processing_interval=3.0  # New batch every 3 seconds
    )
    
    processor.start_streaming(duration=duration)

def run_async_streaming(csv_path):
    """Run async streaming"""
    processor = AsyncStreamingProcessor(csv_path)
    asyncio.run(processor.stream_async(rate=15.0, duration=90))

def run_pyspark_streaming(csv_path, duration=120):
    """Run PySpark-based streaming dashboard"""
    print("🔥 Starting PySpark-based streaming dashboard")
    
    dashboard = PySparkStreamingDashboard(
        csv_path=csv_path,
        batch_size=30,
        processing_interval=3.0
    )
    
    dashboard.start_streaming(duration=duration)

def check_spark_setup():
    """Check if Spark can work"""
    helper = SparkFixHelper()
    
    if not helper.check_java():
        helper.install_java_instructions()
        return False
    
    free_port = helper.find_free_port()
    if free_port:
        print(f"✅ Free port found: {free_port}")
    else:
        print("❌ No free ports available")
    
    return True

# READY TO USE
csv_file_path = "/Users/kshitijnavale/Desktop/sensor data/synthetic_gas_1M.csv"

print("🎯 Multiple Streaming Options Available:")
print("1. Python Streaming:")
print("   run_python_streaming(csv_file_path, duration=120)")
print("2. Async Streaming:")
print("   run_async_streaming(csv_file_path)")
print("3. PySpark Streaming Dashboard:")
print("   run_pyspark_streaming(csv_file_path, duration=120)")
print("4. Check Spark Setup:")
print("   check_spark_setup()")

print(f"\nYour file: {csv_file_path}")
print(f"File size: {499998:,} samples")

# Uncomment to run immediately:
run_pyspark_streaming(csv_file_path, duration=120)

# Clean up Spark session
spark.stop()