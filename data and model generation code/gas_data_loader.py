import pandas as pd
import os
import glob
import numpy as np

def load_gas_sensor_data(data_directory):
    """
    Load gas sensor data from directory containing batch files.
    
    Handles various formats including space-separated .dat files
    with format: feature1 feature2 ... feature128 gas_label
    
    Args:
        data_directory: Path to the directory containing gas sensor data files
        
    Returns:
        pd.DataFrame: Combined dataframe with columns ['f1', 'f2', ..., 'f128', 'label']
    """
    
    print(f"📂 Loading gas sensor data from: {data_directory}")
    
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Directory not found: {data_directory}")
    
    all_data = []
    
    # Find .dat, .csv, or .txt files
    file_patterns = ['*.dat', '*.csv', '*.txt']
    files_found = []
    
    for pattern in file_patterns:
        files_found.extend(glob.glob(os.path.join(data_directory, pattern)))
    
    if not files_found:
        raise FileNotFoundError(f"No data files found in {data_directory}")
    
    print(f"   Found {len(files_found)} data files")
    
    for file_path in sorted(files_found):
        print(f"   📄 Loading: {os.path.basename(file_path)}")
        
        try:
            # Read file as raw text first to inspect format
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            
            # Count how many space-separated values
            parts = first_line.split()
            n_cols = len(parts)
            
            if n_cols < 129:
                print(f"      ⚠️ Warning: Only {n_cols} columns found, expected 129. Skipping...")
                continue
            
            # Read the file with proper parsing
            # Format is: f1 f2 ... f128 label
            df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python')
            
            # Check if last column might have the format "128:value" or similar
            if df.shape[1] >= 129:
                # Standard case: 128 features + 1 label
                feature_cols = df.iloc[:, :128]
                label_col = df.iloc[:, 128]
                
                # If label column contains strings with ":", extract the gas number
                if label_col.dtype == 'object':
                    # Try to extract integer from strings like "1", "2", or "128:-2.65"
                    def extract_label(x):
                        x_str = str(x).strip()
                        # If it contains ':', take the part before ':'
                        if ':' in x_str:
                            # This is likely feature 128 value, not the label
                            # The label should be after this
                            return None
                        try:
                            return int(float(x_str))
                        except:
                            return None
                    
                    labels = label_col.apply(extract_label)
                    
                    # If we got Nones, the format might be different
                    # Try: last column is actually the label
                    if labels.isna().all() or labels.isna().any():
                        # Maybe the format is: f1 f2 ... f128 label (with f128 potentially having format issues)
                        # Let's try to use the LAST column as label
                        if df.shape[1] > 129:
                            label_col = df.iloc[:, -1]  # Use very last column
                            feature_cols = df.iloc[:, :128]
                        
                        # Try to parse last column as integer
                        try:
                            labels = pd.to_numeric(label_col, errors='coerce').astype('Int64')
                            # Remove rows where label couldn't be parsed
                            valid_mask = ~labels.isna()
                            if valid_mask.sum() < len(labels) * 0.5:
                                print(f"      ⚠️ Warning: Could not parse labels for most rows. Trying alternate format...")
                                # Maybe each line is: f1 f2 ... f128 label
                                # And we need to re-read with exactly 129 columns
                                df = pd.read_csv(file_path, sep=r'\s+', header=None, 
                                               engine='python', usecols=range(129))
                                feature_cols = df.iloc[:, :128]
                                labels = df.iloc[:, 128].astype(int)
                            else:
                                feature_cols = feature_cols[valid_mask].reset_index(drop=True)
                                labels = labels[valid_mask].reset_index(drop=True)
                        except Exception as e:
                            print(f"      ❌ Error parsing labels: {e}. Skipping file...")
                            continue
                else:
                    labels = label_col.astype(int)
                
                # Create dataframe
                result_df = pd.DataFrame(feature_cols.values, columns=[f'f{i}' for i in range(1, 129)])
                result_df['label'] = labels
                
                # Remove any rows with NaN labels
                result_df = result_df.dropna(subset=['label'])
                result_df['label'] = result_df['label'].astype(int)
                
                # Validate labels are in expected range (1-6 for gas types)
                valid_labels = result_df['label'].isin([1, 2, 3, 4, 5, 6])
                if valid_labels.sum() < len(result_df) * 0.5:
                    print(f"      ⚠️ Warning: Many invalid labels. Unique values: {result_df['label'].unique()}")
                
                result_df = result_df[valid_labels]
                
                if len(result_df) == 0:
                    print(f"      ⚠️ No valid samples after filtering. Skipping...")
                    continue
                
                all_data.append(result_df)
                print(f"      ✅ Loaded {len(result_df)} samples")
            
            else:
                print(f"      ⚠️ Unexpected format: {df.shape[1]} columns. Skipping...")
                continue
                
        except Exception as e:
            print(f"      ❌ Error loading {os.path.basename(file_path)}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_data:
        raise ValueError(f"No valid data files could be loaded from {data_directory}")
    
    # Combine all data
    print(f"\n🔗 Combining all data...")
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Ensure label is integer
    combined_data['label'] = combined_data['label'].astype(int)
    
    print(f"\n✅ Successfully loaded {len(combined_data):,} total samples")
    print(f"📊 Distribution:")
    for label in sorted(combined_data['label'].unique()):
        count = (combined_data['label'] == label).sum()
        print(f"   Gas {label}: {count:,} samples")
    
    return combined_data


def load_gas_sensor_data_from_csv(csv_path):
    """
    Load gas sensor data from a single CSV file.
    
    Args:
        csv_path: Path to CSV file with columns [f1, f2, ..., f128, label]
        
    Returns:
        pd.DataFrame: Dataframe with gas sensor data
    """
    print(f"📂 Loading data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate columns
    expected_cols = [f'f{i}' for i in range(1, 129)] + ['label']
    
    if not all(col in df.columns for col in expected_cols):
        print(f"⚠️ Warning: Missing some expected columns")
        print(f"   Found columns: {list(df.columns)[:5]}... (showing first 5)")
    
    print(f"✅ Loaded {len(df):,} samples")
    print(f"📊 Distribution:")
    for label in sorted(df['label'].unique()):
        count = (df['label'] == label).sum()
        print(f"   Gas {label}: {count:,} samples")
    
    return df


if __name__ == "__main__":
    # Test the loader
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        
        if os.path.isdir(data_path):
            data = load_gas_sensor_data(data_path)
        elif os.path.isfile(data_path):
            data = load_gas_sensor_data_from_csv(data_path)
        else:
            print(f"❌ Invalid path: {data_path}")
            sys.exit(1)
        
        print(f"\n📈 Data shape: {data.shape}")
        print(f"📋 First few rows:")
        print(data.head())
        print(f"\n📊 Summary statistics:")
        print(data[['f1', 'f2', 'f3', 'label']].describe())
        
    else:
        print("Usage: python gas_data_loader.py <path_to_data_directory_or_csv>")
        print("\nThis script loads gas sensor data from:")
        print("  - Directory with .dat/.csv/.txt files")
        print("  - Directory with subdirectories (one per gas type)")
        print("  - Single CSV file")