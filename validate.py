import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import acf, pacf, adfuller
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# For deep learning models
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class SyntheticDataValidator:
    """
    Comprehensive validation suite for synthetic sensor data quality
    """
    
    def __init__(self, real_data, synthetic_data, gas_names=None):
        """
        Parameters:
        -----------
        real_data : np.array or pd.DataFrame
            Real sensor data (samples x features x timesteps) or (samples x features)
        synthetic_data : np.array or pd.DataFrame
            Synthetic sensor data with same shape as real_data
        gas_names : list
            Names of gas sensors (default: ['Gas1', 'Gas2', ...])
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.gas_names = gas_names or [f'Gas_{i+1}' for i in range(real_data.shape[1])]
        self.results = {}
        
    def statistical_comparison(self):
        """Compare basic statistical properties"""
        print("=" * 80)
        print("STATISTICAL COMPARISON")
        print("=" * 80)
        
        stats_comparison = []
        
        for i, gas in enumerate(self.gas_names):
            real_col = self.real_data[:, i].flatten()
            synth_col = self.synthetic_data[:, i].flatten()
            
            stats_dict = {
                'Gas': gas,
                'Real_Mean': np.mean(real_col),
                'Synth_Mean': np.mean(synth_col),
                'Real_Std': np.std(real_col),
                'Synth_Std': np.std(synth_col),
                'Real_Median': np.median(real_col),
                'Synth_Median': np.median(synth_col),
                'Real_Min': np.min(real_col),
                'Synth_Min': np.min(synth_col),
                'Real_Max': np.max(real_col),
                'Synth_Max': np.max(synth_col),
                'Real_Skew': stats.skew(real_col),
                'Synth_Skew': stats.skew(synth_col),
                'Real_Kurtosis': stats.kurtosis(real_col),
                'Synth_Kurtosis': stats.kurtosis(synth_col)
            }
            stats_comparison.append(stats_dict)
        
        df_stats = pd.DataFrame(stats_comparison)
        print(df_stats.to_string(index=False))
        self.results['statistical_comparison'] = df_stats
        
        return df_stats
    
    def distribution_tests(self):
        """Perform statistical tests to compare distributions"""
        print("\n" + "=" * 80)
        print("DISTRIBUTION TESTS (KS Test & Wasserstein Distance)")
        print("=" * 80)
        
        test_results = []
        
        for i, gas in enumerate(self.gas_names):
            real_col = self.real_data[:, i].flatten()
            synth_col = self.synthetic_data[:, i].flatten()
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(real_col, synth_col)
            
            # Wasserstein distance (Earth Mover's Distance)
            wasserstein = stats.wasserstein_distance(real_col, synth_col)
            
            # Jensen-Shannon divergence
            # Create histograms with same bins
            bins = np.linspace(
                min(real_col.min(), synth_col.min()),
                max(real_col.max(), synth_col.max()),
                50
            )
            real_hist, _ = np.histogram(real_col, bins=bins, density=True)
            synth_hist, _ = np.histogram(synth_col, bins=bins, density=True)
            
            # Normalize to probabilities
            real_hist = real_hist / real_hist.sum()
            synth_hist = synth_hist / synth_hist.sum()
            
            # JS divergence
            m = 0.5 * (real_hist + synth_hist)
            js_div = 0.5 * stats.entropy(real_hist, m) + 0.5 * stats.entropy(synth_hist, m)
            
            test_results.append({
                'Gas': gas,
                'KS_Statistic': ks_stat,
                'KS_PValue': ks_pvalue,
                'KS_Similar': 'Yes' if ks_pvalue > 0.05 else 'No',
                'Wasserstein_Distance': wasserstein,
                'JS_Divergence': js_div
            })
        
        df_tests = pd.DataFrame(test_results)
        print(df_tests.to_string(index=False))
        print("\nInterpretation:")
        print("- KS p-value > 0.05: Distributions are similar (good)")
        print("- Wasserstein Distance: Lower is better (0 = identical)")
        print("- JS Divergence: Lower is better (0 = identical, max ≈ 0.693)")
        
        self.results['distribution_tests'] = df_tests
        return df_tests
    
    def autocorrelation_analysis(self, max_lags=40):
        """Compare autocorrelation patterns"""
        print("\n" + "=" * 80)
        print("AUTOCORRELATION ANALYSIS")
        print("=" * 80)
        
        fig, axes = plt.subplots(len(self.gas_names), 2, figsize=(15, 3*len(self.gas_names)))
        if len(self.gas_names) == 1:
            axes = axes.reshape(1, -1)
        
        acf_scores = []
        
        for i, gas in enumerate(self.gas_names):
            real_col = self.real_data[:, i].flatten()
            synth_col = self.synthetic_data[:, i].flatten()
            
            # Calculate ACF
            real_acf = acf(real_col, nlags=max_lags, fft=True)
            synth_acf = acf(synth_col, nlags=max_lags, fft=True)
            
            # Plot
            axes[i, 0].plot(real_acf, label='Real', linewidth=2)
            axes[i, 0].plot(synth_acf, label='Synthetic', linewidth=2, alpha=0.7)
            axes[i, 0].set_title(f'{gas} - Autocorrelation')
            axes[i, 0].set_xlabel('Lag')
            axes[i, 0].set_ylabel('ACF')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Calculate similarity (correlation between ACF curves)
            acf_similarity = np.corrcoef(real_acf, synth_acf)[0, 1]
            
            # Difference plot
            axes[i, 1].plot(real_acf - synth_acf, color='red', linewidth=2)
            axes[i, 1].set_title(f'{gas} - ACF Difference (Similarity: {acf_similarity:.3f})')
            axes[i, 1].set_xlabel('Lag')
            axes[i, 1].set_ylabel('Real ACF - Synth ACF')
            axes[i, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[i, 1].grid(True, alpha=0.3)
            
            acf_scores.append({
                'Gas': gas,
                'ACF_Similarity': acf_similarity,
                'Max_ACF_Difference': np.max(np.abs(real_acf - synth_acf))
            })
        
        plt.tight_layout()
        plt.savefig('autocorrelation_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        df_acf = pd.DataFrame(acf_scores)
        print(df_acf.to_string(index=False))
        print("\nInterpretation: ACF Similarity close to 1.0 is good")
        
        self.results['autocorrelation'] = df_acf
        return df_acf
    
    def spectral_analysis(self):
        """Compare frequency domain characteristics"""
        print("\n" + "=" * 80)
        print("SPECTRAL ANALYSIS (Frequency Domain)")
        print("=" * 80)
        
        fig, axes = plt.subplots(len(self.gas_names), 1, figsize=(15, 3*len(self.gas_names)))
        if len(self.gas_names) == 1:
            axes = [axes]
        
        spectral_scores = []
        
        for i, gas in enumerate(self.gas_names):
            real_col = self.real_data[:, i].flatten()
            synth_col = self.synthetic_data[:, i].flatten()
            
            # FFT
            n = len(real_col)
            real_fft = np.abs(fft(real_col))[:n//2]
            synth_fft = np.abs(fft(synth_col))[:n//2]
            freqs = fftfreq(n, d=1.0)[:n//2]
            
            # Normalize
            real_fft = real_fft / np.max(real_fft)
            synth_fft = synth_fft / np.max(synth_fft)
            
            # Plot
            axes[i].semilogy(freqs[:1000], real_fft[:1000], label='Real', linewidth=2)
            axes[i].semilogy(freqs[:1000], synth_fft[:1000], label='Synthetic', linewidth=2, alpha=0.7)
            axes[i].set_title(f'{gas} - Power Spectral Density')
            axes[i].set_xlabel('Frequency')
            axes[i].set_ylabel('Power (log scale)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Calculate spectral similarity
            spectral_corr = np.corrcoef(real_fft[:1000], synth_fft[:1000])[0, 1]
            
            spectral_scores.append({
                'Gas': gas,
                'Spectral_Correlation': spectral_corr
            })
        
        plt.tight_layout()
        plt.savefig('spectral_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        df_spectral = pd.DataFrame(spectral_scores)
        print(df_spectral.to_string(index=False))
        print("\nInterpretation: Spectral Correlation close to 1.0 indicates similar frequency characteristics")
        
        self.results['spectral'] = df_spectral
        return df_spectral
    
    def stationarity_test(self):
        """Test for stationarity using Augmented Dickey-Fuller test"""
        print("\n" + "=" * 80)
        print("STATIONARITY TEST (Augmented Dickey-Fuller)")
        print("=" * 80)
        
        stationarity_results = []
        
        for i, gas in enumerate(self.gas_names):
            real_col = self.real_data[:, i].flatten()
            synth_col = self.synthetic_data[:, i].flatten()
            
            # ADF test
            real_adf = adfuller(real_col)
            synth_adf = adfuller(synth_col)
            
            stationarity_results.append({
                'Gas': gas,
                'Real_ADF_Stat': real_adf[0],
                'Real_PValue': real_adf[1],
                'Real_Stationary': 'Yes' if real_adf[1] < 0.05 else 'No',
                'Synth_ADF_Stat': synth_adf[0],
                'Synth_PValue': synth_adf[1],
                'Synth_Stationary': 'Yes' if synth_adf[1] < 0.05 else 'No',
                'Match': 'Yes' if (real_adf[1] < 0.05) == (synth_adf[1] < 0.05) else 'No'
            })
        
        df_station = pd.DataFrame(stationarity_results)
        print(df_station.to_string(index=False))
        print("\nInterpretation: p-value < 0.05 means stationary (rejects unit root)")
        
        self.results['stationarity'] = df_station
        return df_station
    
    def visualize_distributions(self):
        """Visualize distribution comparisons"""
        print("\n" + "=" * 80)
        print("DISTRIBUTION VISUALIZATION")
        print("=" * 80)
        
        fig, axes = plt.subplots(len(self.gas_names), 2, figsize=(15, 3*len(self.gas_names)))
        if len(self.gas_names) == 1:
            axes = axes.reshape(1, -1)
        
        for i, gas in enumerate(self.gas_names):
            real_col = self.real_data[:, i].flatten()
            synth_col = self.synthetic_data[:, i].flatten()
            
            # Histogram
            axes[i, 0].hist(real_col, bins=50, alpha=0.6, label='Real', density=True, color='blue')
            axes[i, 0].hist(synth_col, bins=50, alpha=0.6, label='Synthetic', density=True, color='red')
            axes[i, 0].set_title(f'{gas} - Distribution Comparison')
            axes[i, 0].set_xlabel('Value')
            axes[i, 0].set_ylabel('Density')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Q-Q plot
            percentiles = np.linspace(0, 100, 100)
            real_quantiles = np.percentile(real_col, percentiles)
            synth_quantiles = np.percentile(synth_col, percentiles)
            
            axes[i, 1].scatter(real_quantiles, synth_quantiles, alpha=0.5, s=20)
            axes[i, 1].plot([real_quantiles.min(), real_quantiles.max()], 
                           [real_quantiles.min(), real_quantiles.max()], 
                           'r--', linewidth=2, label='Perfect match')
            axes[i, 1].set_title(f'{gas} - Q-Q Plot')
            axes[i, 1].set_xlabel('Real Data Quantiles')
            axes[i, 1].set_ylabel('Synthetic Data Quantiles')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('distribution_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Plots saved: distribution_visualization.png")
    
    def full_validation_report(self):
        """Run all validation tests and generate comprehensive report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE SYNTHETIC DATA VALIDATION REPORT")
        print("=" * 80 + "\n")
        
        self.statistical_comparison()
        self.distribution_tests()
        self.autocorrelation_analysis()
        self.spectral_analysis()
        self.stationarity_test()
        self.visualize_distributions()
        
        print("\n" + "=" * 80)
        print("OVERALL QUALITY ASSESSMENT")
        print("=" * 80)
        
        # Calculate overall quality score
        quality_scores = []
        
        if 'distribution_tests' in self.results:
            dist_df = self.results['distribution_tests']
            avg_ks_pval = dist_df['KS_PValue'].mean()
            avg_wasserstein = dist_df['Wasserstein_Distance'].mean()
            avg_js = dist_df['JS_Divergence'].mean()
            
            quality_scores.append(f"Distribution Similarity: {avg_ks_pval:.3f} (p-value)")
            quality_scores.append(f"Average Wasserstein Distance: {avg_wasserstein:.4f}")
            quality_scores.append(f"Average JS Divergence: {avg_js:.4f}")
        
        if 'autocorrelation' in self.results:
            acf_df = self.results['autocorrelation']
            avg_acf_sim = acf_df['ACF_Similarity'].mean()
            quality_scores.append(f"Average ACF Similarity: {avg_acf_sim:.3f}")
        
        if 'spectral' in self.results:
            spec_df = self.results['spectral']
            avg_spec_corr = spec_df['Spectral_Correlation'].mean()
            quality_scores.append(f"Average Spectral Correlation: {avg_spec_corr:.3f}")
        
        for score in quality_scores:
            print(f"✓ {score}")
        
        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)
        
        return self.results


class SyntheticRealTransferLearning:
    """
    Test synthetic-to-real transfer learning capability
    """
    
    def __init__(self, real_X, real_y, synth_X, synth_y, input_shape, num_classes):
        """
        Parameters:
        -----------
        real_X : np.array
            Real sensor data features
        real_y : np.array
            Real labels
        synth_X : np.array
            Synthetic sensor data features
        synth_y : np.array
            Synthetic labels
        input_shape : tuple
            Shape of input (timesteps, features) or (features,)
        num_classes : int
            Number of classes for classification
        """
        self.real_X = real_X
        self.real_y = real_y
        self.synth_X = synth_X
        self.synth_y = synth_y
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.results = {}
        
    def create_model(self, model_type='lstm'):
        """Create deep learning model"""
        if model_type == 'lstm':
            model = models.Sequential([
                layers.LSTM(128, input_shape=self.input_shape, return_sequences=True),
                layers.Dropout(0.3),
                layers.LSTM(64, return_sequences=False),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        elif model_type == 'cnn':
            model = models.Sequential([
                layers.Conv1D(64, 3, activation='relu', input_shape=self.input_shape),
                layers.MaxPooling1D(2),
                layers.Conv1D(128, 3, activation='relu'),
                layers.MaxPooling1D(2),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        else:  # simple dense
            model = models.Sequential([
                layers.Flatten(input_shape=self.input_shape),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def test_transfer(self, model_type='lstm', epochs=50, batch_size=32):
        """
        Test 1: Train on synthetic, test on real (domain gap)
        """
        print("\n" + "=" * 80)
        print("TEST 1: SYNTHETIC → REAL TRANSFER (Domain Gap Measurement)")
        print("=" * 80)
        
        # Split real data for testing
        real_X_train, real_X_test, real_y_train, real_y_test = train_test_split(
            self.real_X, self.real_y, test_size=0.2, random_state=42
        )
        
        # Train on synthetic
        model = self.create_model(model_type)
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        print("\nTraining on SYNTHETIC data...")
        history = model.fit(
            self.synth_X, self.synth_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Test on real
        print("Testing on REAL data...")
        real_pred = model.predict(real_X_test)
        real_pred_classes = np.argmax(real_pred, axis=1)
        
        accuracy = accuracy_score(real_y_test, real_pred_classes)
        
        print(f"\n{'='*80}")
        print(f"Synthetic → Real Transfer Accuracy: {accuracy:.4f}")
        print(f"{'='*80}")
        print("\nClassification Report:")
        print(classification_report(real_y_test, real_pred_classes))
        
        self.results['synth_to_real'] = {
            'accuracy': accuracy,
            'predictions': real_pred_classes,
            'true_labels': real_y_test,
            'history': history.history
        }
        
        return accuracy
    
    def test_baseline_real_only(self, model_type='lstm', epochs=50, batch_size=32):
        """
        Test 2: Train on real, test on real (baseline)
        """
        print("\n" + "=" * 80)
        print("TEST 2: REAL → REAL BASELINE")
        print("=" * 80)
        
        # Split real data
        real_X_train, real_X_test, real_y_train, real_y_test = train_test_split(
            self.real_X, self.real_y, test_size=0.2, random_state=42
        )
        
        model = self.create_model(model_type)
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        print("\nTraining on REAL data...")
        history = model.fit(
            real_X_train, real_y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        print("Testing on REAL data...")
        real_pred = model.predict(real_X_test)
        real_pred_classes = np.argmax(real_pred, axis=1)
        
        accuracy = accuracy_score(real_y_test, real_pred_classes)
        
        print(f"\n{'='*80}")
        print(f"Real Only Baseline Accuracy: {accuracy:.4f}")
        print(f"{'='*80}")
        print("\nClassification Report:")
        print(classification_report(real_y_test, real_pred_classes))
        
        self.results['real_baseline'] = {
            'accuracy': accuracy,
            'predictions': real_pred_classes,
            'true_labels': real_y_test,
            'history': history.history
        }
        
        return accuracy
    
    def test_progressive_mixing(self, ratios=[1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0], 
                                model_type='lstm', epochs=50, batch_size=32):
        """
        Test 3: Progressive mixing of real and synthetic data
        """
        print("\n" + "=" * 80)
        print("TEST 3: PROGRESSIVE MIXING (Real + Synthetic)")
        print("=" * 80)
        
        # Split real data
        real_X_train, real_X_test, real_y_train, real_y_test = train_test_split(
            self.real_X, self.real_y, test_size=0.2, random_state=42
        )
        
        mixing_results = []
        
        for real_ratio in ratios:
            synth_ratio = 1.0 - real_ratio
            
            print(f"\n{'-'*80}")
            print(f"Testing Mix: {real_ratio*100:.0f}% Real + {synth_ratio*100:.0f}% Synthetic")
            print(f"{'-'*80}")
            
            if real_ratio == 1.0:
                # Use only real data
                X_train = real_X_train
                y_train = real_y_train
            elif real_ratio == 0.0:
                # Use only synthetic data
                X_train = self.synth_X
                y_train = self.synth_y
            else:
                # Mix real and synthetic
                n_real = int(len(real_X_train) * real_ratio)
                n_synth = int(len(self.synth_X) * synth_ratio)
                
                # Sample from each dataset
                real_idx = np.random.choice(len(real_X_train), n_real, replace=False)
                synth_idx = np.random.choice(len(self.synth_X), n_synth, replace=False)
                
                X_train = np.concatenate([real_X_train[real_idx], self.synth_X[synth_idx]])
                y_train = np.concatenate([real_y_train[real_idx], self.synth_y[synth_idx]])
                
                # Shuffle
                shuffle_idx = np.random.permutation(len(X_train))
                X_train = X_train[shuffle_idx]
                y_train = y_train[shuffle_idx]
            
            # Train model
            model = self.create_model(model_type)
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Test on real data
            real_pred = model.predict(real_X_test)
            real_pred_classes = np.argmax(real_pred, axis=1)
            accuracy = accuracy_score(real_y_test, real_pred_classes)
            
            print(f"Accuracy: {accuracy:.4f}")
            
            mixing_results.append({
                'Real_Ratio': real_ratio,
                'Synthetic_Ratio': synth_ratio,
                'Accuracy': accuracy,
                'Training_Samples': len(X_train)
            })
        
        # Summary
        df_mixing = pd.DataFrame(mixing_results)
        
        print(f"\n{'='*80}")
        print("PROGRESSIVE MIXING RESULTS")
        print(f"{'='*80}")
        print(df_mixing.to_string(index=False))
        
        # Find optimal ratio
        best_idx = df_mixing['Accuracy'].idxmax()
        best_ratio = df_mixing.loc[best_idx]
        
        print(f"\n{'='*80}")
        print(f"OPTIMAL MIX: {best_ratio['Real_Ratio']*100:.0f}% Real + {best_ratio['Synthetic_Ratio']*100:.0f}% Synthetic")
        print(f"Best Accuracy: {best_ratio['Accuracy']:.4f}")
        print(f"{'='*80}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(df_mixing['Real_Ratio'] * 100, df_mixing['Accuracy'], 
                marker='o', linewidth=2, markersize=8)
        plt.xlabel('Real Data Percentage (%)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Performance vs Real/Synthetic Data Ratio', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=best_ratio['Real_Ratio']*100, color='red', 
                   linestyle='--', label=f'Optimal: {best_ratio["Real_Ratio"]*100:.0f}%')
        plt.legend()
        plt.tight_layout()
        plt.savefig('progressive_mixing_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        self.results['progressive_mixing'] = df_mixing
        
        return df_mixing
    
    def full_transfer_analysis(self, model_type='lstm', epochs=50, batch_size=32):
        """Run all transfer learning tests"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TRANSFER LEARNING ANALYSIS")
        print("=" * 80)
        
        # Test 1: Synthetic to Real
        synth_to_real_acc = self.test_transfer(model_type, epochs, batch_size)
        
        # Test 2: Real Baseline
        real_baseline_acc = self.test_baseline_real_only(model_type, epochs, batch_size)
        
        # Test 3: Progressive Mixing
        mixing_results = self.test_progressive_mixing(
            ratios=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0],
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Domain Gap Analysis
        print(f"\n{'='*80}")
        print("DOMAIN GAP ANALYSIS")
        print(f"{'='*80}")
        
        domain_gap = real_baseline_acc - synth_to_real_acc
        gap_percentage = (domain_gap / real_baseline_acc) * 100
        
        print(f"Real Baseline Accuracy: {real_baseline_acc:.4f}")
        print(f"Synthetic→Real Accuracy: {synth_to_real_acc:.4f}")
        print(f"Domain Gap: {domain_gap:.4f} ({gap_percentage:.2f}%)")
        
        if domain_gap < 0.05:
            quality = "EXCELLENT"
            recommendation = "Synthetic data is high quality. Safe to use heavily."
        elif domain_gap < 0.10:
            quality = "GOOD"
            recommendation = "Synthetic data is good. Mix 50-70% real data."
        elif domain_gap < 0.15:
            quality = "MODERATE"
            recommendation = "Synthetic data has noticeable gap. Mix 70-80% real data."
        else:
            quality = "POOR"
            recommendation = "Large domain gap. Use primarily real data (>90%)."
        
        print(f"\nSynthetic Data Quality: {quality}")
        print(f"Recommendation: {recommendation}")
        
        # Best mixing ratio
        best_mix = mixing_results.loc[mixing_results['Accuracy'].idxmax()]
        print(f"\nOptimal Mix Found: {best_mix['Real_Ratio']*100:.0f}% Real + {best_mix['Synthetic_Ratio']*100:.0f}% Synthetic")
        print(f"Optimal Mix Accuracy: {best_mix['Accuracy']:.4f}")
        
        improvement = best_mix['Accuracy'] - synth_to_real_acc
        print(f"Improvement over Synthetic-Only: +{improvement:.4f} ({improvement/synth_to_real_acc*100:.2f}%)")
        
        return self.results


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage_validation():
    """
    Example: How to use the validation suite
    """
    print("EXAMPLE 1: DATA VALIDATION")
    print("="*80)
    
    # Assuming you have loaded your data
    # real_data shape: (n_samples, n_features, timesteps) or (n_samples, n_features)
    # synthetic_data: same shape
    
    # Example with dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 6  # 6 gas sensors
    timesteps = 100
    
    # Simulate real data
    real_data = np.random.randn(n_samples, n_features, timesteps)
    
    # Simulate synthetic data (slightly different distribution)
    synthetic_data = np.random.randn(n_samples, n_features, timesteps) * 1.1 + 0.05
    
    # Flatten for time series analysis (or use 3D for sequential models)
    real_flat = real_data.reshape(n_samples, -1)
    synth_flat = synthetic_data.reshape(n_samples, -1)
    
    gas_names = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
    
    # Initialize validator
    validator = SyntheticDataValidator(
        real_data=real_flat,
        synthetic_data=synth_flat,
        gas_names=gas_names
    )
    
    # Run full validation
    results = validator.full_validation_report()
    
    return results


def example_usage_transfer_learning():
    """
    Example: How to use transfer learning tests
    """
    print("\n\nEXAMPLE 2: TRANSFER LEARNING ANALYSIS")
    print("="*80)
    
    # Prepare your data
    np.random.seed(42)
    
    # Real data
    n_real = 500
    n_synth = 2000
    timesteps = 50
    n_features = 6
    num_classes = 6  # 6 different gases
    
    # Create dummy data (replace with your actual data)
    real_X = np.random.randn(n_real, timesteps, n_features)
    real_y = np.random.randint(0, num_classes, n_real)
    
    synth_X = np.random.randn(n_synth, timesteps, n_features) * 1.1
    synth_y = np.random.randint(0, num_classes, n_synth)
    
    # Initialize transfer learning tester
    transfer_tester = SyntheticRealTransferLearning(
        real_X=real_X,
        real_y=real_y,
        synth_X=synth_X,
        synth_y=synth_y,
        input_shape=(timesteps, n_features),
        num_classes=num_classes
    )
    
    # Run full analysis
    results = transfer_tester.full_transfer_analysis(
        model_type='lstm',  # Options: 'lstm', 'cnn', 'dense'
        epochs=30,
        batch_size=32
    )
    
    return results


def load_and_prepare_real_data(filepath):
    """
    Helper function to load your actual sensor data
    
    Modify this based on your data format
    """
    # Example for CSV
    df = pd.read_csv(filepath)
    
    # Assuming columns: timestamp, ethanol, ethylene, ammonia, acetaldehyde, acetone, toluene, label
    gas_columns = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
    
    X = df[gas_columns].values
    y = df['label'].values if 'label' in df.columns else None
    
    return X, y, gas_columns


def complete_pipeline_example(real_data_path, synthetic_data_path):
    """
    Complete pipeline: Load data → Validate → Test Transfer → Find Optimal Mix
    """
    print("="*80)
    print("COMPLETE VALIDATION & TRANSFER LEARNING PIPELINE")
    print("="*80)
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    real_X, real_y, gas_names = load_and_prepare_real_data(real_data_path)
    synth_X, synth_y, _ = load_and_prepare_real_data(synthetic_data_path)
    
    print(f"Real data shape: {real_X.shape}")
    print(f"Synthetic data shape: {synth_X.shape}")
    
    # Step 2: Statistical Validation
    print("\nStep 2: Running statistical validation...")
    validator = SyntheticDataValidator(real_X, synth_X, gas_names)
    validation_results = validator.full_validation_report()
    
    # Step 3: Reshape for deep learning (if needed)
    print("\nStep 3: Preparing data for deep learning...")
    
    # If you have sequential data, reshape to (samples, timesteps, features)
    # Example: sliding window approach
    def create_sequences(data, seq_length=50):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences)
    
    if len(real_X.shape) == 2:  # If not already sequential
        real_X_seq = create_sequences(real_X)
        synth_X_seq = create_sequences(synth_X)
    else:
        real_X_seq = real_X
        synth_X_seq = synth_X
    
    # Align labels if needed
    if real_y is not None and synth_y is not None:
        if len(real_y) > len(real_X_seq):
            real_y = real_y[:len(real_X_seq)]
        if len(synth_y) > len(synth_X_seq):
            synth_y = synth_y[:len(synth_X_seq)]
    
    # Step 4: Transfer Learning Analysis
    print("\nStep 4: Running transfer learning analysis...")
    
    if real_y is not None and synth_y is not None:
        transfer_tester = SyntheticRealTransferLearning(
            real_X=real_X_seq,
            real_y=real_y,
            synth_X=synth_X_seq,
            synth_y=synth_y,
            input_shape=(real_X_seq.shape[1], real_X_seq.shape[2]),
            num_classes=len(np.unique(real_y))
        )
        
        transfer_results = transfer_tester.full_transfer_analysis(
            model_type='lstm',
            epochs=50,
            batch_size=32
        )
    else:
        print("No labels provided, skipping transfer learning analysis")
        transfer_results = None
    
    # Step 5: Generate recommendations
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    
    # From validation results
    if 'distribution_tests' in validation_results:
        dist_df = validation_results['distribution_tests']
        similar_count = (dist_df['KS_Similar'] == 'Yes').sum()
        total_count = len(dist_df)
        
        print(f"\n1. Distribution Similarity: {similar_count}/{total_count} gases passed KS test")
        
        if similar_count == total_count:
            print("   ✓ EXCELLENT: All gas distributions match well")
        elif similar_count >= total_count * 0.7:
            print("   ✓ GOOD: Most gas distributions match")
        else:
            print("   ⚠ WARNING: Several gases have distribution mismatches")
    
    # From autocorrelation
    if 'autocorrelation' in validation_results:
        acf_df = validation_results['autocorrelation']
        avg_acf = acf_df['ACF_Similarity'].mean()
        
        print(f"\n2. Temporal Pattern Similarity: {avg_acf:.3f}")
        
        if avg_acf > 0.9:
            print("   ✓ EXCELLENT: Temporal patterns match very well")
        elif avg_acf > 0.7:
            print("   ✓ GOOD: Temporal patterns reasonably similar")
        else:
            print("   ⚠ WARNING: Temporal patterns differ significantly")
    
    # From transfer learning
    if transfer_results and 'progressive_mixing' in transfer_results:
        mix_df = transfer_results['progressive_mixing']
        best_row = mix_df.loc[mix_df['Accuracy'].idxmax()]
        
        print(f"\n3. Optimal Data Mix: {best_row['Real_Ratio']*100:.0f}% Real + {best_row['Synthetic_Ratio']*100:.0f}% Synthetic")
        print(f"   Expected Accuracy: {best_row['Accuracy']:.4f}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - Check saved plots for visual analysis")
    print("="*80)
    
    return validation_results, transfer_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Synthetic Data Validation & Transfer Learning Suite")
    print("="*80)
    print("\nThis suite provides:")
    print("1. Statistical validation of synthetic data quality")
    print("2. Transfer learning tests (synthetic→real)")
    print("3. Progressive mixing experiments to find optimal ratio")
    print("\n" + "="*80)
    
    # Choose which example to run:
    
    # OPTION 1: Run with dummy data (for testing)
    print("\nRunning with example/dummy data...")
    validation_results = example_usage_validation()
    transfer_results = example_usage_transfer_learning()
    
    # OPTION 2: Run with your actual data (uncomment and modify paths)
    # real_data_path = 'path/to/your/real_data.csv'
    # synthetic_data_path = 'path/to/your/synthetic_data.csv'
    # validation_results, transfer_results = complete_pipeline_example(
    #     real_data_path, 
    #     synthetic_data_path
    # )
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print("- autocorrelation_comparison.png")
    print("- spectral_comparison.png")
    print("- distribution_visualization.png")
    print("- progressive_mixing_results.png")