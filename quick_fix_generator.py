import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load real data to match its statistics
real_df = pd.read_csv('/Users/kshitijnavale/Desktop/sensor data/data/combined_sensor_data_clean.csv')
feature_cols = [f'f{i}' for i in range(1, 129)]
real_X = real_df[feature_cols].values
real_y = real_df['label'].values

print("Creating better synthetic data...")

# Calculate real data statistics per class
class_stats = {}
for class_id in range(1, 7):
    mask = real_y == class_id
    class_data = real_X[mask]
    class_stats[class_id] = {
        'mean': class_data.mean(axis=0),
        'std': class_data.std(axis=0),
        'count': mask.sum()
    }

# Generate synthetic data matching real class distribution
np.random.seed(42)
synthetic_data = []
synthetic_labels = []

# Match real data percentages from chart (scaled to 300k total)
samples_per_class = {
    1: 55200,   # 18.4%
    2: 63000,   # 21.0%
    3: 35400,   # 11.8%
    4: 41700,   # 13.9%
    5: 64800,   # 21.6%
    6: 39900    # 13.3%
}

for class_id in range(1, 7):
    stats = class_stats[class_id]
    samples = samples_per_class[class_id]
    
    # Generate base data matching real statistics
    class_data = np.random.normal(
        loc=stats['mean'],
        scale=stats['std'] * 2,  # Slightly more variation
        size=(samples, 128)
    )
    
    # Add some controlled noise for realism
    noise = np.random.normal(0, stats['std'] * 0.5, (samples, 128))
    class_data += noise
    
    synthetic_data.append(class_data)
    synthetic_labels.extend([class_id] * samples)

# Combine all classes
synthetic_X = np.vstack(synthetic_data)
synthetic_y = np.array(synthetic_labels)

# Shuffle
indices = np.random.permutation(len(synthetic_y))
synthetic_X = synthetic_X[indices]
synthetic_y = synthetic_y[indices]

# Create DataFrame
synthetic_df = pd.DataFrame(synthetic_X, columns=feature_cols)
synthetic_df['label'] = synthetic_y

# Save
synthetic_df.to_csv('realistic_synthetic_gas_300k.csv', index=False)

print(f"Generated {len(synthetic_df)} samples")
print(f"Value range: [{synthetic_X.min():.1f}, {synthetic_X.max():.1f}]")
print(f"Std: {synthetic_X.std():.1f}")
print("Class distribution:")
for class_id in range(1, 7):
    count = (synthetic_y == class_id).sum()
    print(f"  Class {class_id}: {count}")
