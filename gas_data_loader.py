import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_gas_sensor_data(data_dir):
    """
    Load gas sensor data from .dat files in the specified directory
    
    Args:
        data_dir: Directory containing batch*.dat files
    
    Returns:
        DataFrame with features f1-f128 and label column
    """
    data_dir = Path(data_dir)
    all_data = []
    
    # Find all batch*.dat files
    dat_files = sorted(data_dir.glob("batch*.dat"))
    
    if not dat_files:
        raise FileNotFoundError(f"No batch*.dat files found in {data_dir}")
    
    print(f"Found {len(dat_files)} data files:")
    for file in dat_files:
        print(f"  {file}")
    
    for file_path in dat_files:
        print(f"Loading {file_path.name}...")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        batch_data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 2:
                continue
                
            # First part is the label
            label = int(parts[0])
            
            # Parse feature:value pairs
            features = {}
            for part in parts[1:]:
                if ':' in part:
                    feature_idx, value = part.split(':')
                    features[int(feature_idx)] = float(value)
            
            # Create row with all 128 features
            row = {'label': label}
            for i in range(1, 129):  # f1 to f128
                row[f'f{i}'] = features.get(i, 0.0)
            
            batch_data.append(row)
        
        if batch_data:
            batch_df = pd.DataFrame(batch_data)
            all_data.append(batch_df)
            print(f"  Loaded {len(batch_data)} samples")
    
    if not all_data:
        raise ValueError("No valid data found in any files")
    
    # Combine all batches
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\nTotal samples loaded: {len(combined_df)}")
    print(f"Features: f1 to f128 ({len([col for col in combined_df.columns if col.startswith('f')])} features)")
    print(f"Labels: {sorted(combined_df['label'].unique())}")
    print(f"Label distribution:")
    for label, count in combined_df['label'].value_counts().sort_index().items():
        print(f"  Label {label}: {count} samples")
    
    return combined_df

# Test the loader
if __name__ == "__main__":
    data_dir = "/Users/kshitijnavale/Desktop/sensor data/Dataset"
    df = load_gas_sensor_data(data_dir)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Sample data:")
    print(df.head())
