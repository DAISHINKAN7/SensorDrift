import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime
import json
import queue
from collections import deque, Counter
import random
import asyncio


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
print("   run_python_streaming(csv_file_path, duration=120)")

print("\n2. Async Streaming:")
print("   run_async_streaming(csv_file_path)")

print("\n3. Check Spark Setup:")
print("   check_spark_setup()")

print(f"\nYour file: {csv_file_path}")
print(f"File size: {499998:,} samples")

# Uncomment to run immediately:
run_python_streaming(csv_file_path, duration=120)