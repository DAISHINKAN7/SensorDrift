import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class SimpleTimeSeriesAnalyzer:
    """
    Simple time series analyzer for gas sensor data
    Focus: drift, anomalies, trends
    """
    
    def __init__(self, csv_path, name="Dataset"):
        """Load data"""
        print(f"\n📂 Loading {name}...")
        self.data = pd.read_csv(csv_path)
        self.name = name
        self.features = [f'f{i}' for i in range(1, 129)]
        print(f"✅ Loaded {len(self.data):,} samples with {len(self.features)} sensors")
    
    def analyze_single_sensor(self, sensor_num=1, window=20):
        """
        Analyze one sensor completely
        
        Args:
            sensor_num: Which sensor (1-128)
            window: Window size for rolling stats
        """
        sensor_col = f'f{sensor_num}'
        values = self.data[sensor_col].values
        
        print(f"\n{'='*70}")
        print(f"🔍 ANALYZING SENSOR: {sensor_col}")
        print(f"{'='*70}")
        
        # 1. Basic Statistics
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   Mean:     {np.mean(values):.4e}")
        print(f"   Std Dev:  {np.std(values):.4e}")
        print(f"   Min:      {np.min(values):.4e}")
        print(f"   Max:      {np.max(values):.4e}")
        print(f"   Range:    {np.max(values) - np.min(values):.4e}")
        
        # 2. Drift Detection
        print(f"\n📈 DRIFT ANALYSIS:")
        indices = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(indices, values)
        print(f"   Drift Rate:   {slope:.4e} units/sample")
        print(f"   R-squared:    {r_value**2:.4f}")
        print(f"   P-value:      {p_value:.4f}")
        if p_value < 0.05:
            print(f"   ✅ Significant drift detected")
        else:
            print(f"   ⚠️  No significant drift")
        
        # 3. Stationarity Test
        print(f"\n🔄 STATIONARITY TEST:")
        adf_result = adfuller(values[:10000], autolag='AIC')  # Use first 10k samples
        print(f"   ADF Statistic: {adf_result[0]:.4f}")
        print(f"   P-value:       {adf_result[1]:.4f}")
        if adf_result[1] < 0.05:
            print(f"   ✅ Data is stationary")
        else:
            print(f"   ⚠️  Data is non-stationary (has trends/drift)")
        
        # 4. Anomaly Detection (Z-score)
        print(f"\n🚨 ANOMALY DETECTION:")
        z_scores = np.abs(stats.zscore(values))
        anomalies = z_scores > 3
        anomaly_pct = (np.sum(anomalies) / len(values)) * 100
        print(f"   Anomalies:    {np.sum(anomalies):,} samples ({anomaly_pct:.2f}%)")
        
        # 5. Rolling Statistics
        rolling_mean = pd.Series(values).rolling(window=window, min_periods=1).mean()
        rolling_std = pd.Series(values).rolling(window=window, min_periods=1).std()
        
        # Store results
        results = {
            'sensor': sensor_col,
            'values': values,
            'mean': np.mean(values),
            'std': np.std(values),
            'drift_rate': slope,
            'drift_pvalue': p_value,
            'is_stationary': adf_result[1] < 0.05,
            'anomaly_pct': anomaly_pct,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'anomalies': anomalies
        }
        
        return results
    
    def visualize_sensor(self, sensor_num=1, sample_size=10000):
        """
        Create visualization for one sensor
        
        Args:
            sensor_num: Which sensor to visualize (1-128)
            sample_size: Number of samples to visualize (for speed)
        """
        print(f"\n📊 Creating visualization for f{sensor_num}...")
        
        # Analyze first (but only use subset for visualization)
        results = self.analyze_single_sensor(sensor_num)
        
        # Sample data for visualization if too large
        values = results['values']
        if len(values) > sample_size:
            print(f"   Using {sample_size:,} samples for visualization...")
            indices_sample = np.linspace(0, len(values)-1, sample_size, dtype=int)
            values_plot = values[indices_sample]
            indices = indices_sample
            rolling_mean_plot = results['rolling_mean'].iloc[indices_sample]
            rolling_std_plot = results['rolling_std'].iloc[indices_sample]
            anomalies_plot = results['anomalies'][indices_sample]
        else:
            values_plot = values
            indices = np.arange(len(values))
            rolling_mean_plot = results['rolling_mean']
            rolling_std_plot = results['rolling_std']
            anomalies_plot = results['anomalies']
        
        # Create 2x2 plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Sensor f{sensor_num} - Time Series Analysis ({self.name})', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Raw signal with rolling mean
        ax = axes[0, 0]
        # Create continuous index for plotting
        plot_indices = np.arange(len(values_plot))
        ax.plot(plot_indices, values_plot, alpha=0.4, linewidth=0.5, label='Raw Signal')
        
        # Only plot rolling stats where not NaN
        valid_roll = ~np.isnan(rolling_mean_plot)
        if valid_roll.sum() > 0:
            valid_indices = plot_indices[valid_roll]
            ax.plot(valid_indices, rolling_mean_plot[valid_roll], 
                   color='red', linewidth=2, label='Rolling Mean')
            ax.fill_between(valid_indices, 
                           (rolling_mean_plot - rolling_std_plot)[valid_roll],
                           (rolling_mean_plot + rolling_std_plot)[valid_roll],
                           alpha=0.2, color='red', label='±1 Std')
        ax.set_title('Raw Signal with Rolling Statistics')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Sensor Value')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Plot 2: Anomalies
        ax = axes[0, 1]
        ax.plot(plot_indices, values_plot, alpha=0.6, color='blue')
        anomaly_indices_plot = np.where(anomalies_plot)[0]
        if len(anomaly_indices_plot) > 0:
            ax.scatter(plot_indices[anomaly_indices_plot], values_plot[anomaly_indices_plot], 
                      color='red', s=5, alpha=0.7, label=f'Anomalies ({results["anomaly_pct"]:.1f}%)')
        ax.set_title('Anomaly Detection')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Sensor Value')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Plot 3: Distribution
        ax = axes[1, 0]
        ax.hist(values, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(results['mean'], color='red', linestyle='--', linewidth=2, 
                  label=f"Mean: {results['mean']:.2e}")
        ax.set_title('Value Distribution')
        ax.set_xlabel('Sensor Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        # Plot 4: Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary = f"""
        SUMMARY
        {'='*40}
        
        Mean:           {results['mean']:.4e}
        Std Dev:        {results['std']:.4e}
        
        Drift Rate:     {results['drift_rate']:.4e}
        Drift P-value:  {results['drift_pvalue']:.4f}
        Stationary:     {'Yes ✅' if results['is_stationary'] else 'No ⚠️'}
        
        Anomalies:      {results['anomaly_pct']:.2f}%
        
        Health:         {'Good 🟢' if results['anomaly_pct'] < 5 else 'Fair 🟡' if results['anomaly_pct'] < 10 else 'Poor 🔴'}
        
        Note: Values shown in scientific notation
              due to large magnitude
        """
        
        ax.text(0.1, 0.5, summary, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        # Save
        filename = f'{self.name.lower().replace(" ", "_")}_sensor_f{sensor_num}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {filename}")
        plt.close()
        
        return results
    
    def compare_multiple_sensors(self, sensor_list=[1, 2, 3, 4, 5]):
        """
        Compare multiple sensors side by side
        
        Args:
            sensor_list: List of sensor numbers to compare
        """
        print(f"\n📊 Comparing {len(sensor_list)} sensors...")
        
        comparison_data = []
        
        for sensor_num in sensor_list:
            results = self.analyze_single_sensor(sensor_num)
            comparison_data.append({
                'Sensor': f'f{sensor_num}',
                'Mean': results['mean'],
                'Std Dev': results['std'],
                'Drift Rate': results['drift_rate'],
                'Anomaly %': results['anomaly_pct'],
                'Stationary': 'Yes' if results['is_stationary'] else 'No'
            })
        
        df = pd.DataFrame(comparison_data)
        
        print(f"\n{'='*70}")
        print("SENSOR COMPARISON TABLE")
        print(f"{'='*70}")
        print(df.to_string(index=False))
        
        # Save to CSV
        filename = f'{self.name.lower().replace(" ", "_")}_comparison.csv'
        df.to_csv(filename, index=False)
        print(f"\n✅ Saved comparison to: {filename}")
        
        return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Your file paths
    synthetic_path = "/Users/kshitijnavale/Desktop/sensor data/better_synthetic_gas_300k.csv"
    real_path = "/Users/kshitijnavale/Desktop/sensor data/data/combined_sensor_data_clean.csv"
    
    print("\n" + "="*70)
    print("🚀 GAS SENSOR TIME SERIES ANALYSIS")
    print("="*70)
    
    # ========================================
    # STEP 1: Analyze Synthetic Data
    # ========================================
    print("\n\n" + "="*70)
    print("STEP 1: SYNTHETIC DATA (Noisy)")
    print("="*70)
    
    synthetic = SimpleTimeSeriesAnalyzer(synthetic_path, name="Synthetic")
    
    # Analyze sensor 1 in detail
    syn_results = synthetic.visualize_sensor(sensor_num=1)
    
    # Compare first 5 sensors
    syn_comparison = synthetic.compare_multiple_sensors([1, 2, 3, 4, 5])
    
    # ========================================
    # STEP 2: Analyze Real Data
    # ========================================
    print("\n\n" + "="*70)
    print("STEP 2: REAL DATA (Clean)")
    print("="*70)
    
    real = SimpleTimeSeriesAnalyzer(real_path, name="Real")
    
    # Analyze sensor 1 in detail
    real_results = real.visualize_sensor(sensor_num=1)
    
    # Compare first 5 sensors
    real_comparison = real.compare_multiple_sensors([1, 2, 3, 4, 5])
    
    # ========================================
    # STEP 3: Compare Synthetic vs Real
    # ========================================
    print("\n\n" + "="*70)
    print("COMPARISON: SYNTHETIC vs REAL")
    print("="*70)
    
    print(f"\n📈 DRIFT COMPARISON (Sensor f1):")
    print(f"   Synthetic: {syn_results['drift_rate']:.6f}")
    print(f"   Real:      {real_results['drift_rate']:.6f}")
    print(f"   Ratio:     {abs(syn_results['drift_rate']/real_results['drift_rate']) if real_results['drift_rate'] != 0 else 'N/A':.2f}x")
    
    print(f"\n🚨 ANOMALY COMPARISON (Sensor f1):")
    print(f"   Synthetic: {syn_results['anomaly_pct']:.2f}%")
    print(f"   Real:      {real_results['anomaly_pct']:.2f}%")
    print(f"   Difference: {syn_results['anomaly_pct'] - real_results['anomaly_pct']:.2f}%")
    
    print(f"\n📊 STATIONARITY:")
    print(f"   Synthetic: {'Stationary ✅' if syn_results['is_stationary'] else 'Non-stationary ⚠️'}")
    print(f"   Real:      {'Stationary ✅' if real_results['is_stationary'] else 'Non-stationary ⚠️'}")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("   • synthetic_sensor_f1.png")
    print("   • real_sensor_f1.png")
    print("   • synthetic_comparison.csv")
    print("   • real_comparison.csv")
    print("\n💡 Next: You can analyze any sensor by calling:")
    print("   synthetic.visualize_sensor(sensor_num=10)")
    print("   real.visualize_sensor(sensor_num=10)")