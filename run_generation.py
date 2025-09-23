#!/usr/bin/env python3
"""
Simple script to generate 1M synthetic gas sensor samples
"""

from generate_10M_gas_data import generate_10_million_samples

# Configuration
DATA_DIR = "/Users/kshitijnavale/Desktop/sensor data/Dataset"
OUTPUT_FILE = "synthetic_gas_10M.csv"
BATCH_SIZE = 10000  # Adjust based on your RAM

if __name__ == "__main__":
    print("🚀 Starting 1M Gas Sensor Data Generation...")
    
    try:
        result_file = generate_10_million_samples(
            data_dir=DATA_DIR,
            save_path=OUTPUT_FILE,
            batch_size=BATCH_SIZE
        )
        
        print(f"\n✅ SUCCESS! Generated file: {result_file}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check your data directory and try again.")
