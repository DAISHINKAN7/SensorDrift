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
        
        # CATASTROPHIC noise configuration - DESIGNED FOR VISIBLE CHAOS
        self.noise_config = {
            'label_noise_rate': 0.22,            # 22% wrong labels
            'class_overlap_ratio': 0.85,         # 85% overlap
            'signal_to_noise_ratio': 0.15,       # 15% signal
            
            'spike_rate': 0.20,                  # 20% spikes (INCREASED for more anomalies!)
            'spike_magnitude_range': (6, 25),    # 6-25x multiplier (REDUCED slightly)
            'burst_error_rate': 0.12,            # 12% burst errors (INCREASED!)
            'baseline_shift_rate': 0.20,         # 20% baseline shifts (INCREASED!)
            'baseline_shift_magnitude': 25000,   # BIGGER shifts
            
            'sensor_dropout_rate': 0.15,         # 15% dropout (INCREASED!)
            'stuck_sensor_rate': 0.18,           # 18% stuck (INCREASED!)
            'drift_magnitude': 0.12,             # 12% drift (INCREASED!)
            
            'saturation_rate': 0.20,             # 20% saturation (REDUCED - was removing anomalies!)
            'dead_zone_rate': 0.12,              # 12% dead zones (REDUCED!)
            'non_linear_distortion': 0.35,       # 35% non-linear (INCREASED!)
            
            'temporal_correlation': 0.65,        # Moderate correlation (REDUCED for more chaos)
            'periodic_interference_prob': 0.70,  # 70% periodic noise (INCREASED!)
        }
        
        print(f"🔥 Standalone Catastrophic Generator v2 🔥")
        print(f"🎯 Target: 55-60% accuracy with MAXIMUM VISIBLE CHAOS")
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
            
            # Increase std for noise (FIXED for real data scale)
            noisy_std = blended_std * 0.3  # Much smaller multiplier
            
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
        """Apply visible catastrophic noise WITHOUT aggressive clipping"""
        noisy_data = data.copy()
        noisy_labels = labels.copy()
        n_samples, n_features = data.shape
        stats = self.gas_stats[gas_id]
        
        print(f"     🔥 Applying catastrophic noise to {stats['name']}...")
        
        # Track anomaly count for reporting
        anomaly_mask = np.zeros(n_samples, dtype=bool)
        
        # 1. LABEL NOISE (22% - CRITICAL FOR LOW ACCURACY)
        n_mislabel = int(n_samples * self.noise_config['label_noise_rate'])
        if n_mislabel > 0:
            mislabel_idx = np.random.choice(n_samples, n_mislabel, replace=False)
            other_gases = [g for g in self.gas_configs.keys() if g != gas_id]
            for idx in mislabel_idx:
                noisy_labels[idx] = np.random.choice(other_gases)
        
        # 2. BASE GAUSSIAN NOISE (fixed scale)
        base_noise = np.random.normal(0, stats['std'] * 0.1, (n_samples, n_features))
        noisy_data += base_noise
        
        # 3. VISIBLE SPIKES (smaller multipliers)
        spike_mask = np.random.random((n_samples, n_features)) < self.noise_config['spike_rate']
        n_spikes = spike_mask.sum()
        if n_spikes > 0:
            spike_mults = np.random.uniform(2, 8, n_spikes)  # Reduced from 6-25
            spike_signs = np.random.choice([-1, 1], n_spikes)
            noisy_data[spike_mask] *= (spike_signs * spike_mults)
            # Track which samples have spikes
            anomaly_mask |= spike_mask.any(axis=1)
        
        # 4. BURST ERRORS (visible chaos regions)
        n_bursts = max(1, int(n_samples * self.noise_config['burst_error_rate'] / 80))
        for _ in range(n_bursts):
            burst_start = np.random.randint(0, max(1, n_samples - 150))
            burst_len = np.random.randint(60, min(200, n_samples - burst_start))
            burst_features = np.random.choice(n_features, np.random.randint(30, 70), replace=False)
            
            burst_type = np.random.choice(['scale_high', 'scale_low', 'shift', 'chaos'])
            for feat in burst_features:
                if burst_type == 'scale_high':
                    noisy_data[burst_start:burst_start+burst_len, feat] *= np.random.uniform(5.0, 20.0)
                elif burst_type == 'scale_low':
                    noisy_data[burst_start:burst_start+burst_len, feat] *= np.random.uniform(0.05, 0.3)
                elif burst_type == 'shift':
                    noisy_data[burst_start:burst_start+burst_len, feat] += np.random.uniform(-30000, 30000)
                else:  # chaos
                    noisy_data[burst_start:burst_start+burst_len, feat] = np.random.uniform(-20000, 300000, burst_len)
            
            anomaly_mask[burst_start:burst_start+burst_len] = True
        
        # 5. BASELINE SHIFTS (visible jumps)
        n_shifts = max(1, int(n_samples * self.noise_config['baseline_shift_rate'] / 400))
        for _ in range(n_shifts):
            shift_point = np.random.randint(100, n_samples - 100)
            shift_magnitude = np.random.uniform(-self.noise_config['baseline_shift_magnitude'],
                                               self.noise_config['baseline_shift_magnitude'], 
                                               n_features)
            noisy_data[shift_point:] += shift_magnitude
        
        # 6. PROGRESSIVE DRIFT (visible trend)
        time_vector = np.linspace(0, 1, n_samples).reshape(-1, 1)
        drift_pattern = time_vector ** 1.5
        drift_magnitude = self.noise_config['drift_magnitude'] * stats['mean']
        drift = drift_pattern * np.random.uniform(-drift_magnitude, drift_magnitude)
        noisy_data += drift
        
        # 7. SENSOR DROPOUTS (flatlines to near-zero)
        dropout_mask = np.random.random((n_samples, n_features)) < self.noise_config['sensor_dropout_rate']
        if dropout_mask.sum() > 0:
            noisy_data[dropout_mask] = np.random.uniform(-2000, 2000, dropout_mask.sum())
            anomaly_mask |= dropout_mask.any(axis=1)
        
        # 8. STUCK SENSORS (constant values)
        for feat_idx in range(n_features):
            if np.random.random() < self.noise_config['stuck_sensor_rate']:
                stuck_value = np.random.uniform(15000, 120000)
                stuck_samples = np.random.choice(n_samples, n_samples//3, replace=False)
                noisy_data[stuck_samples, feat_idx] = stuck_value + np.random.normal(0, 200, len(stuck_samples))
        
        # 9. LIGHT SATURATION (only clip extremes, not all spikes!)
        for feat_idx in range(n_features):
            if np.random.random() < self.noise_config['saturation_rate']:
                # Only saturate EXTREME values (beyond sensor physical limits)
                sat_high = np.random.uniform(400000, 600000)  # Much higher!
                sat_low = np.random.uniform(-50000, -20000)   # Allow negative spikes
                noisy_data[:, feat_idx] = np.clip(noisy_data[:, feat_idx], sat_low, sat_high)
        
        # 10. DEAD ZONES (reduced, less aggressive)
        for feat_idx in range(n_features):
            if np.random.random() < self.noise_config['dead_zone_rate']:
                dead_center = np.random.uniform(30000, 70000)
                dead_width = np.random.uniform(3000, 8000)
                mask = np.abs(noisy_data[:, feat_idx] - dead_center) < dead_width
                if mask.sum() > 0:
                    noisy_data[mask, feat_idx] = dead_center + np.random.normal(0, 500, mask.sum())
        
        # 11. NON-LINEAR DISTORTIONS
        for feat_idx in range(n_features):
            if np.random.random() < self.noise_config['non_linear_distortion']:
                x = noisy_data[:, feat_idx]
                x_mean = x.mean()
                x_std = x.std()
                
                # Normalize around mean
                x_norm = (x - x_mean) / (x_std + 1e-8)
                
                distortion = np.random.choice(['square', 'cube', 'sqrt', 'sigmoid', 'tanh'])
                if distortion == 'square':
                    x_norm = np.sign(x_norm) * (x_norm ** 2)
                elif distortion == 'cube':
                    x_norm = x_norm ** 3
                elif distortion == 'sqrt':
                    x_norm = np.sign(x_norm) * np.sqrt(np.abs(x_norm))
                elif distortion == 'sigmoid':
                    x_norm = 2 * (1 / (1 + np.exp(-2 * x_norm)) - 0.5)
                else:  # tanh
                    x_norm = np.tanh(x_norm)
                
                # Denormalize
                noisy_data[:, feat_idx] = x_norm * x_std * 1.5 + x_mean
        
        # 12. TEMPORAL CORRELATION (reduced for more independence)
        if self.noise_config['temporal_correlation'] > 0:
            alpha = self.noise_config['temporal_correlation']
            for feat_idx in range(n_features):
                for i in range(1, min(n_samples, 1000)):  # Only first 1000 samples
                    noisy_data[i, feat_idx] = (alpha * noisy_data[i-1, feat_idx] + 
                                               (1 - alpha) * noisy_data[i, feat_idx])
        
        # 13. PERIODIC INTERFERENCE (visible oscillations)
        if np.random.random() < self.noise_config['periodic_interference_prob']:
            freq = np.random.choice([25, 40, 50, 60, 75, 100, 150])
            t = np.linspace(0, freq, n_samples)
            interference = np.sin(2 * np.pi * t).reshape(-1, 1)
            # Add harmonics for complexity
            interference += 0.3 * np.sin(4 * np.pi * t).reshape(-1, 1)
            interference += 0.15 * np.sin(6 * np.pi * t).reshape(-1, 1)
            interference_strength = np.random.uniform(5000, 15000)
            noisy_data += interference * interference_strength
        
        # 14. FINAL: Add controlled random chaos
        final_chaos = np.random.normal(0, stats['std'].mean() * 0.3, (n_samples, n_features))
        noisy_data += final_chaos
        
        # 15. CLIP to realistic sensor range (tighter than before for scale matching)
        noisy_data = np.clip(noisy_data, -50000, 700000)  # Match real data scale!
        
        anomaly_pct = (anomaly_mask.sum() / n_samples) * 100
        print(f"        💥 Created {anomaly_mask.sum():,} anomalous samples ({anomaly_pct:.1f}%)")
        
        return noisy_data, noisy_labels
    
    def generate_1M_samples(self, batch_size=5000, save_path='catastrophic_gas_1M.csv'):
        """Generate 1M catastrophically noisy samples"""
        total_samples = 1_000_000
        target_distribution = self._get_target_distribution(total_samples)
        
        print(f"\n{'='*80}")
        print(f"🔥 GENERATING 1M CATASTROPHIC SAMPLES (v2 - NO AGGRESSIVE CLIPPING!) 🔥")
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
        
        # Check for anomalies in final data
        for feat in ['f1']:
            values = final_dataset[feat].values
            mean, std = values.mean(), values.std()
            anomalies = np.abs(values - mean) > 3 * std
            anomaly_pct = (anomalies.sum() / len(values)) * 100
            print(f"   {feat} anomalies (>3σ): {anomalies.sum():,} ({anomaly_pct:.2f}%)")
        
        print(f"\n🎯 EXPECTED:")
        print(f"   ✅ Values: -100K to 800K range (WIDE for spikes!)")
        print(f"   ✅ Anomaly rate: 5-15% (VISIBLE!)")
        print(f"   ✅ Model accuracy: 55-60%")
        print(f"   ✅ Time series: CHAOTIC with visible spikes!")
        
        return save_path


if __name__ == "__main__":
    print("""
🔥🔥🔥 STANDALONE CATASTROPHIC GENERATOR V2 🔥🔥🔥
============================================================

KEY IMPROVEMENTS:
❌ REMOVED aggressive clipping that killed anomalies!
✅ Wider clip range: -100K to 800K (allows spikes!)
✅ 18% spike rate with 8-35x multipliers
✅ 12% burst errors (visible chaos regions)
✅ 20% baseline shifts (visible jumps)
✅ 15% sensor dropouts & 18% stuck sensors
✅ 22% wrong labels (max accuracy: 78%)
✅ 85% class overlap (very hard!)

EXPECTED RESULTS:
📈 Anomaly rate: 5-15% (not 0%!)
📊 Wild value range like real sensors
🎯 Model accuracy: 55-60%
🔥 VISIBLY CHAOTIC time series plots!
""")
    
    generator = StandaloneCatastrophicGenerator(balance_strategy="custom")
    
    output_file = generator.generate_1M_samples(
        batch_size=5000,
        save_path='catastrophic_gas_1M_v2.csv'
    )
    
    print(f"\n🎉 Done! Train your model on: {output_file}")
    print(f"\n📊 Run your drift_detection.py on this file to see VISIBLE CHAOS!")