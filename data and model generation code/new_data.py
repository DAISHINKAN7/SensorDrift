import pandas as pd
import numpy as np
import time
import os
import gc

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("Warning: tqdm not available. Progress bars disabled.")
    TQDM_AVAILABLE = False

class StandaloneCatastrophicGenerator:
    """
    Generates catastrophically noisy gas sensor data WITHOUT needing original data.
    Uses realistic ranges based on typical MOX gas sensor characteristics.
    """
    
    def __init__(self, balance_strategy="custom"):
        self.n_features = 128
        self.balance_strategy = balance_strategy
        
        # Define 6 gas types with REALISTIC sensor response characteristics
        self.gas_configs = {
            1: {'name': 'Ethanol', 'base_range': (20000, 80000), 'volatility': 15000},
            2: {'name': 'Ammonia', 'base_range': (30000, 90000), 'volatility': 18000},
            3: {'name': 'Acetone', 'base_range': (25000, 70000), 'volatility': 12000},
            4: {'name': 'Acetaldehyde', 'base_range': (15000, 60000), 'volatility': 10000},
            5: {'name': 'Toluene', 'base_range': (35000, 100000), 'volatility': 20000},
            6: {'name': 'Ethylene', 'base_range': (40000, 110000), 'volatility': 22000},
        }
        
        # CATASTROPHIC noise configuration
        self.noise_config = {
            'label_noise_rate': 0.22,            # 22% wrong labels
            'class_overlap_ratio': 0.85,         # 85% overlap
            'signal_to_noise_ratio': 0.15,       # 15% signal
            
            'spike_rate': 0.12,                  # 12% spikes
            'spike_magnitude_range': (5, 25),    # 5-25x multiplier
            'burst_error_rate': 0.08,            # 8% burst errors
            'baseline_shift_rate': 0.15,         # 15% baseline shifts
            'baseline_shift_magnitude': 15000,   # Absolute shift
            
            'sensor_dropout_rate': 0.10,         # 10% dropout
            'stuck_sensor_rate': 0.12,           # 12% stuck
            'drift_magnitude': 0.08,             # 8% drift per feature
            
            'saturation_rate': 0.25,             # 25% saturation
            'dead_zone_rate': 0.15,              # 15% dead zones
            'non_linear_distortion': 0.30,       # 30% non-linear
            
            'temporal_correlation': 0.75,        # Strong temporal correlation
            'periodic_interference_prob': 0.60,  # 60% periodic noise
        }
        
        print(f"🔥 Standalone Catastrophic Generator 🔥")
        print(f"🎯 Target: 55-60% accuracy with VISIBLE chaos")
        print(f"🧪 Gases: {len(self.gas_configs)} types")
        print(f"📏 Features: {self.n_features}")
        
        self._create_gas_statistics()
    
    def _create_gas_statistics(self):
        """Create realistic statistics for each gas type"""
        print(f"\n🧮 Creating realistic gas sensor statistics...")
        
        # Calculate global stats for overlap
        all_means = []
        all_stds = []
        
        for gas_id, config in self.gas_configs.items():
            base_min, base_max = config['base_range']
            volatility = config['volatility']
            
            # Create realistic per-feature means (varying across sensors)
            feature_means = np.random.uniform(base_min, base_max, self.n_features)
            feature_stds = np.random.uniform(volatility * 0.5, volatility * 1.5, self.n_features)
            
            all_means.append(feature_means)
            all_stds.append(feature_stds)
        
        # Global statistics
        global_mean = np.mean(all_means, axis=0)
        global_std = np.mean(all_stds, axis=0)
        
        # Create blended statistics per gas (for overlap)
        self.gas_stats = {}
        overlap = self.noise_config['class_overlap_ratio']
        
        for gas_id, config in self.gas_configs.items():
            base_min, base_max = config['base_range']
            volatility = config['volatility']
            
            # Original stats
            orig_mean = np.random.uniform(base_min, base_max, self.n_features)
            orig_std = np.random.uniform(volatility * 0.5, volatility * 1.5, self.n_features)
            
            # Blend with global for overlap
            blended_mean = overlap * global_mean + (1 - overlap) * orig_mean
            blended_std = overlap * global_std + (1 - overlap) * orig_std
            
            # Increase std for noise
            noisy_std = blended_std * (1 / self.noise_config['signal_to_noise_ratio'])
            
            self.gas_stats[gas_id] = {
                'name': config['name'],
                'mean': blended_mean,
                'std': noisy_std,
                'original_mean': orig_mean,
                'original_std': orig_std,
                'base_range': config['base_range'],
            }
            
            print(f"   Gas {gas_id} ({config['name']}): "
                  f"range={config['base_range']}, overlap={overlap*100:.0f}%")
    
    def _get_target_distribution(self, total_samples=1_000_000):
        """Get target number of samples per gas"""
        if self.balance_strategy == "custom":
            distribution = {
                1: 300_000, 2: 300_000, 3: 10_000,
                4: 10_000, 5: 300_000, 6: 80_000
            }
        elif self.balance_strategy == "balanced":
            samples_per_gas = total_samples // len(self.gas_configs)
            distribution = {gas_id: samples_per_gas for gas_id in self.gas_configs.keys()}
        else:  # proportional
            distribution = {
                1: 250_000, 2: 250_000, 3: 50_000,
                4: 50_000, 5: 250_000, 6: 150_000
            }
        
        # Adjust to exact total
        actual_total = sum(distribution.values())
        if actual_total != total_samples:
            max_gas = max(distribution, key=distribution.get)
            distribution[max_gas] += (total_samples - actual_total)
        
        print(f"\n🎯 Target distribution:")
        for gas_id in sorted(distribution.keys()):
            count = distribution[gas_id]
            pct = (count / total_samples) * 100
            name = self.gas_stats[gas_id]['name']
            print(f"   Gas {gas_id} ({name}): {count:,} ({pct:.1f}%)")
        
        return distribution
    
    def _apply_visible_catastrophic_noise(self, data, gas_id, labels):
        """Apply visible catastrophic noise"""
        noisy_data = data.copy()
        noisy_labels = labels.copy()
        n_samples, n_features = data.shape
        stats = self.gas_stats[gas_id]
        
        print(f"     🔥 Applying catastrophic noise to {stats['name']}...")
        
        # 1. LABEL NOISE
        n_mislabel = int(n_samples * self.noise_config['label_noise_rate'])
        if n_mislabel > 0:
            mislabel_idx = np.random.choice(n_samples, n_mislabel, replace=False)
            other_gases = [g for g in self.gas_configs.keys() if g != gas_id]
            for idx in mislabel_idx:
                noisy_labels[idx] = np.random.choice(other_gases)
        
        # 2. BASE GAUSSIAN NOISE
        base_noise = np.random.normal(0, stats['std'] * 0.3, (n_samples, n_features))
        noisy_data += base_noise
        
        # 3. VISIBLE SPIKES
        spike_mask = np.random.random((n_samples, n_features)) < self.noise_config['spike_rate']
        if spike_mask.sum() > 0:
            spike_min, spike_max = self.noise_config['spike_magnitude_range']
            spike_mults = np.random.uniform(spike_min, spike_max, spike_mask.sum())
            spike_signs = np.random.choice([-1, 1], spike_mask.sum())
            noisy_data[spike_mask] *= (spike_signs * spike_mults)
        
        # 4. BURST ERRORS
        n_bursts = max(1, int(n_samples * self.noise_config['burst_error_rate'] / 100))
        for _ in range(n_bursts):
            burst_start = np.random.randint(0, max(1, n_samples - 100))
            burst_len = np.random.randint(50, min(150, n_samples - burst_start))
            burst_features = np.random.choice(n_features, np.random.randint(20, 60), replace=False)
            
            burst_type = np.random.choice(['scale', 'shift', 'zero'])
            for feat in burst_features:
                if burst_type == 'scale':
                    noisy_data[burst_start:burst_start+burst_len, feat] *= np.random.uniform(0.1, 5.0)
                elif burst_type == 'shift':
                    noisy_data[burst_start:burst_start+burst_len, feat] += np.random.uniform(-20000, 20000)
                else:  # zero
                    noisy_data[burst_start:burst_start+burst_len, feat] = np.random.uniform(-1000, 1000)
        
        # 5. DRAMATIC BASELINE SHIFTS (like real sensors)
        n_shifts = max(3, int(n_samples * self.noise_config['baseline_shift_rate'] / 100))
        for _ in range(n_shifts):
            shift_point = np.random.randint(100, n_samples - 100)
            # Much larger shifts that affect rolling mean visibly
            shift_magnitude = np.random.uniform(-stats['mean'] * 0.4, stats['mean'] * 0.4)
            noisy_data[shift_point:] += shift_magnitude
        
        # 6. DRAMATIC TEMPORAL DRIFT (like real sensors)
        time_vector = np.linspace(0, 1, n_samples)
        
        # Multiple strong drift components for dramatic rolling mean changes
        for feat_idx in range(n_features):
            # Strong linear drift
            linear_drift = np.random.uniform(-0.8, 0.8) * time_vector * stats['mean'][feat_idx]
            
            # Multiple sinusoidal drifts (temperature cycles, daily patterns, etc.)
            sine_drift = np.zeros(n_samples)
            for _ in range(np.random.randint(2, 4)):  # 2-3 sine components
                freq = np.random.uniform(0.3, 4.0)
                phase = np.random.uniform(0, 2*np.pi)
                amplitude = np.random.uniform(0.2, 0.6) * stats['mean'][feat_idx]
                sine_drift += amplitude * np.sin(freq * time_vector + phase)
            
            # Strong random walk drift
            walk_steps = np.random.normal(0, stats['std'][feat_idx] * 0.03, n_samples)
            walk_drift = np.cumsum(walk_steps)
            
            # Exponential drift (sensor aging)
            exp_drift = np.random.uniform(-0.3, 0.3) * stats['mean'][feat_idx] * (np.exp(time_vector * 2) - 1)
            
            # Combine all drifts for dramatic effect
            total_drift = linear_drift + sine_drift + walk_drift + exp_drift
            noisy_data[:, feat_idx] += total_drift
        
        # 7. SENSOR DROPOUTS
        dropout_mask = np.random.random((n_samples, n_features)) < self.noise_config['sensor_dropout_rate']
        if dropout_mask.sum() > 0:
            noisy_data[dropout_mask] = np.random.uniform(-1000, 1000, dropout_mask.sum())
        
        # 8. STUCK SENSORS
        for feat_idx in range(n_features):
            if np.random.random() < self.noise_config['stuck_sensor_rate']:
                stuck_value = np.random.uniform(10000, 100000)
                stuck_samples = np.random.choice(n_samples, n_samples//3, replace=False)
                noisy_data[stuck_samples, feat_idx] = stuck_value
        
        # 9. SATURATION
        for feat_idx in range(n_features):
            if np.random.random() < self.noise_config['saturation_rate']:
                sat_high = np.random.uniform(100000, 150000)
                sat_low = np.random.uniform(0, 5000)
                noisy_data[:, feat_idx] = np.clip(noisy_data[:, feat_idx], sat_low, sat_high)
        
        # 10. DEAD ZONES
        for feat_idx in range(n_features):
            if np.random.random() < self.noise_config['dead_zone_rate']:
                dead_center = np.random.uniform(20000, 80000)
                dead_width = np.random.uniform(5000, 15000)
                mask = np.abs(noisy_data[:, feat_idx] - dead_center) < dead_width
                if mask.sum() > 0:
                    noisy_data[mask, feat_idx] = dead_center + np.random.normal(0, 100, mask.sum())
        
        # 11. NON-LINEAR DISTORTIONS
        for feat_idx in range(n_features):
            if np.random.random() < self.noise_config['non_linear_distortion']:
                x = noisy_data[:, feat_idx]
                x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
                
                distortion = np.random.choice(['square', 'sqrt', 'sigmoid'])
                if distortion == 'square':
                    x_norm = x_norm ** 2
                elif distortion == 'sqrt':
                    x_norm = np.sqrt(x_norm)
                else:  # sigmoid
                    x_norm = 1 / (1 + np.exp(-10 * (x_norm - 0.5)))
                
                noisy_data[:, feat_idx] = x_norm * (x.max() - x.min()) + x.min()
        
        # 12. ENHANCED TEMPORAL CORRELATION (for realistic rolling stats)
        if self.noise_config['temporal_correlation'] > 0:
            # Strong temporal correlation with varying alpha
            for feat_idx in range(n_features):
                # Variable correlation strength over time
                base_alpha = self.noise_config['temporal_correlation']
                
                for i in range(1, n_samples):
                    # Add some randomness to correlation strength
                    alpha = base_alpha + np.random.normal(0, 0.1)
                    alpha = np.clip(alpha, 0.3, 0.9)  # Keep reasonable bounds
                    
                    # Apply temporal correlation with noise
                    correlation_noise = np.random.normal(0, stats['std'][feat_idx] * 0.05)
                    noisy_data[i, feat_idx] = (alpha * noisy_data[i-1, feat_idx] + 
                                               (1 - alpha) * noisy_data[i, feat_idx] + 
                                               correlation_noise)
        
        # 13. PERIODIC INTERFERENCE
        if np.random.random() < self.noise_config['periodic_interference_prob']:
            freq = np.random.choice([30, 50, 60, 100])
            t = np.linspace(0, freq, n_samples)
            interference = np.sin(2 * np.pi * t).reshape(-1, 1)
            interference_strength = np.random.uniform(3000, 8000)
            noisy_data += interference * interference_strength
        
        # 14. PRESERVE CHAOS - Only clip extreme outliers, keep spikes visible
        # Calculate dynamic bounds based on actual data distribution
        q1, q99 = np.percentile(noisy_data, [1, 99])
        extreme_low = q1 - 5 * (q99 - q1)  # Allow 5x range below Q1
        extreme_high = q99 + 10 * (q99 - q1)  # Allow 10x range above Q99
        
        # Only clip truly extreme values that would break systems
        noisy_data = np.clip(noisy_data, extreme_low, extreme_high)
        
        return noisy_data, noisy_labels
    
    def generate_1M_samples(self, batch_size=5000, save_path='catastrophic_gas_1M.csv'):
        """Generate 1M catastrophically noisy samples"""
        total_samples = 1_000_000
        target_distribution = self._get_target_distribution(total_samples)
        
        print(f"\n{'='*80}")
        print(f"🔥 GENERATING 1M CATASTROPHIC SAMPLES 🔥")
        print(f"{'='*80}")
        
        start_time = time.time()
        temp_files = []
        
        for gas_id in sorted(self.gas_configs.keys()):
            target_samples = target_distribution[gas_id]
            stats = self.gas_stats[gas_id]
            
            print(f"\n🔥 Generating {target_samples:,} samples for Gas {gas_id} ({stats['name']})...")
            
            remaining = target_samples
            temp_file = f'temp_gas_{gas_id}_{int(time.time())}.csv'
            temp_files.append(temp_file)
            first_batch = True
            
            if TQDM_AVAILABLE:
                pbar = tqdm(total=target_samples, desc=f"Gas {gas_id}")
            
            while remaining > 0:
                current_batch = min(batch_size, remaining)
                
                # Generate base data
                batch_data = np.random.normal(
                    loc=stats['mean'],
                    scale=stats['std'],
                    size=(current_batch, self.n_features)
                )
                
                batch_labels = np.full(current_batch, gas_id)
                
                # Apply catastrophic noise
                batch_data, batch_labels = self._apply_visible_catastrophic_noise(
                    batch_data, gas_id, batch_labels
                )
                
                # Save
                batch_df = pd.DataFrame(batch_data, columns=[f'f{i}' for i in range(1, 129)])
                batch_df['label'] = batch_labels
                batch_df.to_csv(temp_file, mode='a', header=first_batch, index=False)
                first_batch = False
                
                del batch_data, batch_df, batch_labels
                gc.collect()
                
                remaining -= current_batch
                if TQDM_AVAILABLE:
                    pbar.update(current_batch)
            
            if TQDM_AVAILABLE:
                pbar.close()
            
            print(f"     ✅ Completed")
        
        # Combine files
        print(f"\n📁 Combining files...")
        combined_data = []
        for i, temp_file in enumerate(temp_files):
            print(f"   📖 Reading file {i+1}/{len(temp_files)}")
            combined_data.append(pd.read_csv(temp_file))
        
        print("   🔗 Concatenating...")
        final_dataset = pd.concat(combined_data, ignore_index=True)
        
        print("   🔀 Shuffling...")
        final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   💾 Saving to {save_path}...")
        final_dataset.to_csv(save_path, index=False)
        
        # Cleanup
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        file_size_mb = os.path.getsize(save_path) / (1024**2)
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"🔥 COMPLETE! 🔥")
        print(f"{'='*80}")
        print(f"📊 Samples: {len(final_dataset):,}")
        print(f"💾 Size: {file_size_mb:.2f} MB")
        print(f"⏱️ Time: {elapsed:.2f}s")
        print(f"📂 File: {save_path}")
        
        # Quality check
        print(f"\n📈 QUALITY CHECK:")
        for feat in ['f1', 'f10', 'f50']:
            values = final_dataset[feat].values
            print(f"   {feat}: mean={values.mean():.0f}, std={values.std():.0f}, "
                  f"range=[{values.min():.0f}, {values.max():.0f}]")
        
        print(f"\n🎯 EXPECTED:")
        print(f"   ✅ Values: 10K-150K range")
        print(f"   ✅ Anomaly rate: 10-15%")
        print(f"   ✅ Model accuracy: 55-60%")
        
        return save_path


if __name__ == "__main__":
    print("""
🔥🔥🔥 STANDALONE CATASTROPHIC GENERATOR 🔥🔥🔥
============================================================

NO ORIGINAL DATA NEEDED!

This generates realistic gas sensor data with:
✅ 22% wrong labels (max accuracy: 78%)
✅ 85% class overlap (very hard!)
✅ Visible spikes and anomalies (10-15%)
✅ Realistic value ranges (10K-150K)
✅ Burst errors, drift, baseline shifts
✅ Right-skewed distributions

EXPECTED: 55-60% model accuracy
""")
    
    generator = StandaloneCatastrophicGenerator(balance_strategy="custom")
    
    output_file = generator.generate_1M_samples(
        batch_size=5000,
        save_path='catastrophic_gas_1M.csv'
    )
    
    print(f"\n🎉 Done! Train your model on: {output_file}")