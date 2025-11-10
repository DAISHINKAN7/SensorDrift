import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ============================================================================
# DIAGNOSTIC SUITE - Find Why Accuracy Is So Low
# ============================================================================

class SensorDataDiagnostics:
    """
    Comprehensive diagnostics to find root cause of poor model performance
    """
    
    def __init__(self, X_data, y_labels, data_name="Dataset"):
        """
        Parameters:
        -----------
        X_data : np.array
            Sensor data (samples, timesteps, features) or (samples, features)
        y_labels : np.array
            Labels for classification
        data_name : str
            Name for this dataset (e.g., "Real", "Synthetic")
        """
        self.X = X_data
        self.y = y_labels
        self.name = data_name
        
    def check_data_shapes(self):
        """Check basic data properties"""
        print("="*80)
        print(f"DATA SHAPE DIAGNOSTICS - {self.name}")
        print("="*80)
        
        print(f"\n✓ X shape: {self.X.shape}")
        print(f"✓ y shape: {self.y.shape}")
        print(f"✓ Number of samples: {len(self.X)}")
        
        if len(self.X.shape) == 3:
            print(f"✓ Timesteps: {self.X.shape[1]}")
            print(f"✓ Features: {self.X.shape[2]}")
        elif len(self.X.shape) == 2:
            print(f"✓ Features: {self.X.shape[1]}")
            print("⚠ WARNING: Data is 2D, not time series!")
        
        # Check for NaN/Inf
        nan_count = np.isnan(self.X).sum()
        inf_count = np.isinf(self.X).sum()
        
        if nan_count > 0:
            print(f"\n❌ CRITICAL: {nan_count} NaN values found!")
        if inf_count > 0:
            print(f"\n❌ CRITICAL: {inf_count} Inf values found!")
        
        if nan_count == 0 and inf_count == 0:
            print("\n✓ No NaN or Inf values")
        
        # Value ranges
        print(f"\nValue Ranges:")
        print(f"  Min: {self.X.min():.4f}")
        print(f"  Max: {self.X.max():.4f}")
        print(f"  Mean: {self.X.mean():.4f}")
        print(f"  Std: {self.X.std():.4f}")
        
    def check_labels(self):
        """Analyze label distribution and quality"""
        print("\n" + "="*80)
        print(f"LABEL DIAGNOSTICS - {self.name}")
        print("="*80)
        
        # Basic stats
        unique_labels = np.unique(self.y)
        label_counts = Counter(self.y)
        
        print(f"\n✓ Number of unique classes: {len(unique_labels)}")
        print(f"✓ Label range: {self.y.min()} to {self.y.max()}")
        print(f"✓ Expected classes for 6 gases: 0-5")
        
        if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
            print("\n⚠ WARNING: Labels are not sequential (0,1,2,...)!")
            print(f"   Found labels: {unique_labels}")
        
        # Class balance
        print("\nClass Distribution:")
        print("-" * 50)
        total = len(self.y)
        for label, count in sorted(label_counts.items()):
            percentage = (count / total) * 100
            bar = "█" * int(percentage / 2)
            print(f"  Class {label}: {count:4d} ({percentage:5.2f}%) {bar}")
        
        # Check imbalance
        counts = list(label_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        
        print(f"\n{'='*50}")
        print(f"Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3:
            print("❌ CRITICAL: Severe class imbalance!")
            print("   Recommendation: Use class weights or resampling")
        elif imbalance_ratio > 1.5:
            print("⚠ WARNING: Moderate class imbalance")
        else:
            print("✓ Classes are reasonably balanced")
        
        # Visualize
        plt.figure(figsize=(10, 5))
        labels_list = sorted(label_counts.keys())
        counts_list = [label_counts[l] for l in labels_list]
        
        plt.bar(labels_list, counts_list, color='steelblue', alpha=0.8)
        plt.xlabel('Class Label', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(f'{self.name} - Class Distribution', fontsize=14, fontweight='bold')
        plt.xticks(labels_list)
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for i, (label, count) in enumerate(zip(labels_list, counts_list)):
            percentage = (count / total) * 100
            plt.text(label, count, f'{percentage:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.name}_class_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def check_feature_quality(self, gas_names=None):
        """Check if features have signal or just noise"""
        print("\n" + "="*80)
        print(f"FEATURE QUALITY DIAGNOSTICS - {self.name}")
        print("="*80)
        
        if gas_names is None:
            if len(self.X.shape) == 3:
                gas_names = [f'Feature_{i}' for i in range(self.X.shape[2])]
            else:
                gas_names = [f'Feature_{i}' for i in range(self.X.shape[1])]
        
        # Flatten if 3D
        if len(self.X.shape) == 3:
            X_flat = self.X.reshape(-1, self.X.shape[2])
        else:
            X_flat = self.X
        
        print("\nPer-Feature Statistics:")
        print("-" * 80)
        
        feature_stats = []
        for i, name in enumerate(gas_names):
            feature = X_flat[:, i]
            
            stats_dict = {
                'Feature': name,
                'Mean': feature.mean(),
                'Std': feature.std(),
                'Min': feature.min(),
                'Max': feature.max(),
                'Range': feature.max() - feature.min(),
                'Zeros_%': (feature == 0).sum() / len(feature) * 100
            }
            feature_stats.append(stats_dict)
        
        df_stats = pd.DataFrame(feature_stats)
        print(df_stats.to_string(index=False))
        
        # Check for dead features
        print("\n" + "="*50)
        dead_features = df_stats[df_stats['Std'] < 0.01]
        if len(dead_features) > 0:
            print("❌ CRITICAL: Dead/constant features detected!")
            print(dead_features[['Feature', 'Std']])
        else:
            print("✓ All features have variation")
        
        # Check for features that are mostly zeros
        mostly_zero = df_stats[df_stats['Zeros_%'] > 90]
        if len(mostly_zero) > 0:
            print("\n⚠ WARNING: Features that are mostly zeros:")
            print(mostly_zero[['Feature', 'Zeros_%']])
        
    def check_temporal_structure(self):
        """Check if time series data has temporal structure"""
        if len(self.X.shape) != 3:
            print("\n⚠ Skipping temporal diagnostics - data is not 3D time series")
            return
        
        print("\n" + "="*80)
        print(f"TEMPORAL STRUCTURE DIAGNOSTICS - {self.name}")
        print("="*80)
        
        # Sample a few sequences and check if they vary
        n_samples = min(5, len(self.X))
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(15, 3*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        temporal_variance = []
        
        for i in range(n_samples):
            sample = self.X[i]  # Shape: (timesteps, features)
            
            # Calculate variance across time for each feature
            time_variance = np.var(sample, axis=0)
            temporal_variance.append(time_variance.mean())
            
            # Plot
            axes[i].plot(sample)
            axes[i].set_title(f'Sample {i} (Label: {self.y[i]}) - Temporal Variance: {time_variance.mean():.4f}')
            axes[i].set_xlabel('Timestep')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.name}_temporal_structure.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        avg_temporal_var = np.mean(temporal_variance)
        print(f"\nAverage Temporal Variance: {avg_temporal_var:.4f}")
        
        if avg_temporal_var < 0.01:
            print("❌ CRITICAL: Very low temporal variance - data may be constant!")
        elif avg_temporal_var < 0.1:
            print("⚠ WARNING: Low temporal variance - weak temporal patterns")
        else:
            print("✓ Data has temporal structure")
        
    def check_class_separability(self, gas_names=None):
        """Check if classes are separable in feature space"""
        print("\n" + "="*80)
        print(f"CLASS SEPARABILITY DIAGNOSTICS - {self.name}")
        print("="*80)
        
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Flatten if 3D
        if len(self.X.shape) == 3:
            X_flat = self.X.reshape(len(self.X), -1)
        else:
            X_flat = self.X
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flat)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"\nPCA Explained Variance: {pca.explained_variance_ratio_.sum():.4f}")
        print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f}")
        print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f}")
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y, 
                            cmap='tab10', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Class')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        plt.title(f'{self.name} - Class Separability (PCA)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.name}_class_separability.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Calculate class separation metric
        from scipy.spatial.distance import cdist
        
        unique_classes = np.unique(self.y)
        class_centers = []
        
        for cls in unique_classes:
            class_mask = self.y == cls
            class_center = X_pca[class_mask].mean(axis=0)
            class_centers.append(class_center)
        
        class_centers = np.array(class_centers)
        
        # Average distance between class centers
        if len(class_centers) > 1:
            center_distances = cdist(class_centers, class_centers)
            np.fill_diagonal(center_distances, np.nan)
            avg_separation = np.nanmean(center_distances)
            
            print(f"\nAverage Class Separation: {avg_separation:.4f}")
            
            if avg_separation < 0.5:
                print("❌ CRITICAL: Classes are very close - hard to separate!")
            elif avg_separation < 2.0:
                print("⚠ WARNING: Classes have low separation")
            else:
                print("✓ Classes are reasonably separated")
        
    def run_full_diagnostics(self, gas_names=None):
        """Run complete diagnostic suite"""
        print("\n" + "🔍"*40)
        print(f"COMPLETE DIAGNOSTICS: {self.name}")
        print("🔍"*40 + "\n")
        
        self.check_data_shapes()
        self.check_labels()
        self.check_feature_quality(gas_names)
        self.check_temporal_structure()
        self.check_class_separability(gas_names)
        
        print("\n" + "="*80)
        print("DIAGNOSTICS COMPLETE")
        print("="*80)


def compare_real_vs_synthetic(real_X, real_y, synth_X, synth_y, gas_names=None):
    """Compare real and synthetic datasets side by side"""
    print("\n" + "🔄"*40)
    print("REAL vs SYNTHETIC COMPARISON")
    print("🔄"*40 + "\n")
    
    # Run diagnostics on both
    print("\n" + "📊"*40)
    print("REAL DATA DIAGNOSTICS")
    print("📊"*40)
    real_diag = SensorDataDiagnostics(real_X, real_y, "Real")
    real_diag.run_full_diagnostics(gas_names)
    
    print("\n" + "🤖"*40)
    print("SYNTHETIC DATA DIAGNOSTICS")
    print("🤖"*40)
    synth_diag = SensorDataDiagnostics(synth_X, synth_y, "Synthetic")
    synth_diag.run_full_diagnostics(gas_names)
    
    # Direct comparison
    print("\n" + "="*80)
    print("DIRECT COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n📏 Shape Comparison:")
    print(f"  Real:      {real_X.shape}")
    print(f"  Synthetic: {synth_X.shape}")
    shape_match = real_X.shape == synth_X.shape
    print(f"  Match: {'✓' if shape_match else '❌'}")
    
    print(f"\n🏷️  Label Comparison:")
    real_unique = len(np.unique(real_y))
    synth_unique = len(np.unique(synth_y))
    print(f"  Real classes:      {real_unique}")
    print(f"  Synthetic classes: {synth_unique}")
    print(f"  Match: {'✓' if real_unique == synth_unique else '❌'}")
    
    print(f"\n📊 Value Range Comparison:")
    print(f"  Real:      [{real_X.min():.4f}, {real_X.max():.4f}]")
    print(f"  Synthetic: [{synth_X.min():.4f}, {synth_X.max():.4f}]")
    
    # Scale difference
    real_scale = real_X.std()
    synth_scale = synth_X.std()
    scale_ratio = synth_scale / real_scale
    print(f"\n📐 Scale Comparison:")
    print(f"  Real std:      {real_scale:.4f}")
    print(f"  Synthetic std: {synth_scale:.4f}")
    print(f"  Ratio:         {scale_ratio:.4f}")
    
    if scale_ratio < 0.8 or scale_ratio > 1.2:
        print("  ⚠ WARNING: Significant scale mismatch!")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("SENSOR DATA DIAGNOSTICS SUITE")
    print("="*80)
    print("\nThis will help identify why your accuracy is so low (16-22%)")
    print("\nLoading your data...")
    
    # REPLACE THESE PATHS WITH YOUR ACTUAL DATA FILES
    real_data_path = '/Users/kshitijnavale/Desktop/sensor data/data/combined_sensor_data_clean.csv'  # YOUR FILE
    synth_data_path = '/Users/kshitijnavale/Desktop/sensor data/realistic_synthetic_gas_300k.csv'  # REALISTIC DISTRIBUTION
    
    # Load data (modify based on your format)
    # Example for CSV with columns: timestamp, gas1, gas2, ..., gas6, label
    try:
        real_df = pd.read_csv(real_data_path)
        synth_df = pd.read_csv(synth_data_path)
        
        # Use the actual feature column names
        feature_cols = [f'f{i}' for i in range(1, 129)]  # f1 to f128
        
        # Extract features and labels
        real_X = real_df[feature_cols].values
        real_y = real_df['label'].values
        
        synth_X = synth_df[feature_cols].values
        synth_y = synth_df['label'].values
        
        print(f"✓ Real data loaded: {real_X.shape}")
        print(f"✓ Synthetic data loaded: {synth_X.shape}")
        
        # Run diagnostics
        compare_real_vs_synthetic(real_X, real_y, synth_X, synth_y, feature_cols)
        
    except FileNotFoundError:
        print("\n⚠ Data files not found. Running with dummy data for demonstration...")
        print("   Please update the file paths with your actual data.\n")
        
        # Dummy data for demonstration
        np.random.seed(42)
        real_X = np.random.randn(500, 100, 6)
        real_y = np.random.randint(0, 6, 500)
        synth_X = np.random.randn(2000, 100, 6)
        synth_y = np.random.randint(0, 6, 2000)
        
        compare_real_vs_synthetic(real_X, real_y, synth_X, synth_y,
                                gas_names=["Ethanol", "Ethylene", "Ammonia", 
                                         "Acetaldehyde", "Acetone", "Toluene"])