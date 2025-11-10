import pandas as pd
import numpy as np
import time
import os
import gc
from gas_data_loader import load_gas_sensor_data
from scipy.ndimage import gaussian_filter1d

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")
    TQDM_AVAILABLE = False

class GasSensor1MGenerator:
    def __init__(self, original_data):
        self.original_data = original_data
        self.feature_columns = [f'f{i}' for i in range(1, 129)]  # f1 to f128
        self.n_features = 128
        self.gas_types = original_data['label'].unique()
        self.gas_stats = {}
        
        print(f"Initialized generator with {len(original_data)} samples")
        print(f"Gas types: {sorted(self.gas_types)}")
        print(f"Features: {self.n_features}")
        
        self._calculate_gas_statistics()
        
    def _calculate_gas_statistics(self):
        print("\nCalculating statistics for each gas type...")
        
        for gas in self.gas_types:
            gas_data = self.original_data[self.original_data['label'] == gas]
            features = gas_data[self.feature_columns].values
            
            self.gas_stats[gas] = {
                'count': len(gas_data),
                'mean': features.mean(axis=0),
                'std': features.std(axis=0),
                'cov': np.cov(features.T)
            }
            
            print(f"  Gas {gas}: {len(gas_data)} samples")

    def generate_1M_samples(self, batch_size=10000, save_path='synthetic_gas_1M.csv'):
        total_samples = 1_000_000
        
        # Calculate imbalanced synthetic counts based on original proportions
        sorted_gases = sorted(self.gas_types)
        original_counts = np.array([self.gas_stats[gas]['count'] for gas in sorted_gases])
        total_original = original_counts.sum()
        proportions = original_counts / total_original
        synthetic_counts = np.round(proportions * total_samples).astype(int)
        diff = total_samples - synthetic_counts.sum()
        if diff != 0:
            idx = np.argmax(original_counts)
            synthetic_counts[idx] += diff
        
        print(f"\n{'='*60}")
        print(f"GENERATING 1 MILLION SYNTHETIC SAMPLES WITH IMBALANCE AND NOISE")
        print(f"{'='*60}")
        print(f"Batch size: {batch_size:,}")
        print(f"Memory optimization: Enabled")
        
        for i, gas in enumerate(sorted_gases):
            print(f"  Planned samples for Gas {gas}: {synthetic_counts[i]:,}")
        
        start_time = time.time()
        temp_files = []
        
        noise_level = 1.0  # Multiplier for Gaussian white noise std (1.0 makes it very noisy by doubling variance)
        filter_sigma = 2.0  # Sigma for Gaussian filter (smoothing)
        
        for gas_idx, gas in enumerate(sorted_gases):
            samples_for_gas = synthetic_counts[gas_idx]
            print(f"\nGenerating {samples_for_gas:,} samples for Gas {gas} ({gas_idx + 1}/{len(sorted_gases)})...")
            stats = self.gas_stats[gas]
            
            # Add regularization to covariance matrix
            cov_matrix = stats['cov'] + 0.01 * np.eye(self.n_features)
            
            remaining_samples = samples_for_gas
            temp_file = f'temp_gas_{gas}_{int(time.time())}.csv'
            temp_files.append(temp_file)
            first_batch = True
            
            if TQDM_AVAILABLE:
                pbar = tqdm(total=samples_for_gas, desc=f"Gas {gas}")
            
            while remaining_samples > 0:
                current_batch = min(batch_size, remaining_samples)
                
                # Generate batch from multivariate normal
                batch_data = np.random.multivariate_normal(
                    mean=stats['mean'],
                    cov=cov_matrix,
                    size=current_batch
                )
                
                # Apply Gaussian filter (smoothing) to each feature vector
                batch_data = gaussian_filter1d(batch_data, sigma=filter_sigma, axis=1)
                
                # Add Gaussian white noise to make data as noisy as possible
                noise = np.random.normal(0, noise_level * stats['std'], batch_data.shape)
                batch_data += noise
                
                # Create DataFrame
                batch_df = pd.DataFrame(batch_data, columns=self.feature_columns)
                batch_df['label'] = gas
                
                # Save batch
                batch_df.to_csv(temp_file, mode='a', header=first_batch, index=False)
                first_batch = False
                
                # Clean up memory
                del batch_data, batch_df
                gc.collect()
                
                remaining_samples -= current_batch
                
                if TQDM_AVAILABLE:
                    pbar.update(current_batch)
            
            if TQDM_AVAILABLE:
                pbar.close()
            
            print(f"  Completed: {samples_for_gas:,} samples for Gas {gas}")
        
        print(f"\nCombining temporary files into final dataset...")
        
        # Combine all temporary files
        combined_data = []
        for i, temp_file in enumerate(temp_files):
            print(f"  Reading temporary file {i+1}/{len(temp_files)}: {temp_file}")
            temp_df = pd.read_csv(temp_file)
            combined_data.append(temp_df)
            
        print("  Concatenating all data...")
        final_dataset = pd.concat(combined_data, ignore_index=True)
        
        print("  Shuffling dataset...")
        final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"  Saving final dataset to {save_path}...")
        final_dataset.to_csv(save_path, index=False)
        
        # Clean up temporary files
        print("  Cleaning up temporary files...")
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Calculate file size
        file_size_mb = os.path.getsize(save_path) / (1024**2)
        file_size_gb = file_size_mb / 1024
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print("1 MILLION SAMPLES GENERATED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Total samples: {len(final_dataset):,}")
        print(f"File size: {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")
        print(f"Time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Samples per minute: {len(final_dataset) / (total_time / 60):,.0f}")
        print(f"Final file: {save_path}")
        
        self._validate_dataset(final_dataset)
        
        return save_path

    def _validate_dataset(self, synthetic_dataset):
        print(f"\n{'='*40}")
        print("DATASET VALIDATION")
        print(f"{'='*40}")
        
        print(f"Dataset shape: {synthetic_dataset.shape}")
        print(f"Total samples: {len(synthetic_dataset):,}")
        
        print(f"\nLabel distribution:")
        label_counts = synthetic_dataset['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            percentage = (count / len(synthetic_dataset)) * 100
            print(f"  Gas {label}: {count:,} ({percentage:.1f}%)")
        
        # Sample validation
        print(f"\nFeature statistics validation (using 10K sample):")
        sample_size = min(10000, len(synthetic_dataset))
        sample_synth = synthetic_dataset.sample(n=sample_size, random_state=42)
        
        for gas in sorted(self.gas_types):
            orig_gas = self.original_data[self.original_data['label'] == gas][self.feature_columns]
            synth_gas = sample_synth[sample_synth['label'] == gas][self.feature_columns]
            
            if len(synth_gas) > 0:
                orig_mean = orig_gas.mean().mean()
                synth_mean = synth_gas.mean().mean()
                orig_std = orig_gas.std().mean()
                synth_std = synth_gas.std().mean()
                
                print(f"  Gas {gas}:")
                print(f"    Mean - Original: {orig_mean:.3f}, Synthetic: {synth_mean:.3f}")
                print(f"    Std  - Original: {orig_std:.3f}, Synthetic: {synth_std:.3f}")

def generate_1_million_samples(data_dir, save_path='synthetic_gas_1M.csv', batch_size=10000):
    """
    Generate exactly 1 million synthetic gas sensor samples
    
    Args:
        data_dir: Directory containing batch*.dat files
        save_path: Path to save the final CSV file
        batch_size: Batch size for memory optimization
    
    Returns:
        Path to the generated file
    """
    print("Starting 1 Million Gas Sensor Sample Generation Process...")
    print(f"Expected file size: ~0.5-0.8 GB")
    print(f"Estimated time: 1-3 minutes")
    
    # Load original data
    print("\nLoading original data...")
    df = load_gas_sensor_data(data_dir)
    
    # Generate synthetic data
    generator = GasSensor1MGenerator(df)
    final_file = generator.generate_1M_samples(
        batch_size=batch_size,
        save_path=save_path
    )
    
    return final_file

if __name__ == "__main__":
    # Configuration
    data_directory = "/Users/kshitijnavale/Desktop/sensor data/Dataset"
    output_file = "synthetic_gas_1M.csv"
    batch_size = 10000  # Adjust based on your RAM (10000=~2GB, 20000=~4GB)
    
    # Generate 1M samples
    result_file = generate_1_million_samples(
        data_dir=data_directory,
        save_path=output_file,
        batch_size=batch_size
    )
    
    print(f"\n🎉 SUCCESS! 1M dataset saved to: {result_file}")
    
    print("""
MEMORY OPTIMIZATION TIPS:
- Current batch_size={}: Uses ~1-2GB RAM
- Increase to 20000-50000 if you have 8GB+ RAM (faster generation)
- Decrease to 5000 if you have limited RAM
- The script automatically cleans up memory and temporary files
""".format(batch_size))