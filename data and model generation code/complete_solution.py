import numpy as np
import pandas as pd
from improved_model import ImprovedSensorClassifier, SensorDataPreprocessor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# STEP 1: LOAD YOUR DATA
# ============================================================================

print("="*80)
print("LOADING YOUR SENSOR DATA")
print("="*80)

real_data_path = "/Users/kshitijnavale/Desktop/sensor data/data/combined_sensor_data_clean.csv"
synth_data_path = "/Users/kshitijnavale/Desktop/sensor data/better_synthetic_gas_300k.csv"

# Load data
real_df = pd.read_csv(real_data_path)
synth_df = pd.read_csv(synth_data_path)

print(f"✓ Real data loaded: {real_df.shape}")
print(f"✓ Synthetic data loaded: {synth_df.shape}")

# Extract features (f1 to f128)
feature_columns = [f'f{i}' for i in range(1, 129)]

real_X = real_df[feature_columns].values
real_y = real_df['label'].values - 1  # Convert to 0-5 (was 1-6)

synth_X = synth_df[feature_columns].values
synth_y = synth_df['label'].values - 1  # Convert to 0-5

print(f"\nReal data: {real_X.shape}, labels: {real_y.shape}")
print(f"Synthetic data: {synth_X.shape}, labels: {synth_y.shape}")
print(f"Label range: {real_y.min()}-{real_y.max()}")

# Gas names mapping (from your data)
gas_names = {
    0: "Ammonia",      # label 1 → 0
    1: "Acetaldehyde", # label 2 → 1  (wait, check this mapping from your data)
    2: "Acetone",      # label 3 → 2
    3: "Ethanol",      # label 4 → 3
    4: "Ethylene",     # label 5 → 4
    5: "Toluene"       # label 6 → 5
}

# Check class distribution
print("\n📊 Class Distribution:")
print("-"*80)
print("Real data:")
for label in range(6):
    count = (real_y == label).sum()
    pct = count / len(real_y) * 100
    print(f"  Class {label} ({gas_names[label]}): {count:5d} ({pct:5.2f}%)")

print("\nSynthetic data:")
for label in range(6):
    count = (synth_y == label).sum()
    pct = count / len(synth_y) * 100
    print(f"  Class {label} ({gas_names[label]}): {count:5d} ({pct:5.2f}%)")

# ============================================================================
# STEP 2: RESHAPE DATA FOR TIME SERIES
# ============================================================================

print(f"\n{'='*80}")
print("RESHAPING DATA FOR LSTM/CNN")
print("="*80)

# Your data has 128 features. We need to decide on time series structure.
# Option 1: Treat each sample as a sequence (128 timesteps, 1 feature)
# Option 2: Treat as (n timesteps, n features) - e.g., (32, 4) or (16, 8)
# Option 3: Keep flat and use Dense network

# Let's try Option 2: Reshape to (16 timesteps, 8 features)
# This assumes your 128 features represent temporal readings

timesteps = 16
features_per_step = 8

real_X_seq = real_X.reshape(-1, timesteps, features_per_step)
synth_X_seq = synth_X.reshape(-1, timesteps, features_per_step)

print(f"Reshaped real data: {real_X_seq.shape}")
print(f"Reshaped synthetic data: {synth_X_seq.shape}")
print(f"Format: (samples, timesteps={timesteps}, features={features_per_step})")

# ============================================================================
# STEP 3: BASELINE TEST - REAL DATA ONLY
# ============================================================================

print(f"\n{'='*80}")
print("BASELINE: REAL DATA ONLY WITH IMPROVED MODEL")
print("="*80)

# Split real data
X_train, X_test, y_train, y_test = train_test_split(
    real_X_seq, real_y, test_size=0.2, stratify=real_y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# Normalize
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, features_per_step)
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)

X_val_flat = X_val.reshape(-1, features_per_step)
X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)

X_test_flat = X_test.reshape(-1, features_per_step)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

# Build and train model
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight

def build_cnn_lstm_model(input_shape, num_classes):
    """CNN-LSTM hybrid model"""
    model = models.Sequential([
        # CNN layers
        layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # LSTM layers
        layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3)),
        layers.Bidirectional(layers.LSTM(32, dropout=0.3)),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Build model
input_shape = (timesteps, features_per_step)
num_classes = 6

model = build_cnn_lstm_model(input_shape, num_classes)

print("\n📋 Model Architecture:")
model.summary()

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

print(f"\n⚖️  Class weights: {class_weight_dict}")

# Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
checkpoint = callbacks.ModelCheckpoint('best_baseline_model.keras', monitor='val_accuracy', 
                                       save_best_only=True, verbose=1)

# Train
print("\n🚀 Training baseline model (Real data only)...")
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=25,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# Evaluate
y_pred = model.predict(X_test_scaled, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
baseline_accuracy = accuracy_score(y_test, y_pred_classes)

print(f"\n{'='*80}")
print(f"🎯 BASELINE ACCURACY (Real Only): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print("="*80)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_classes, 
                          target_names=[gas_names[i] for i in range(6)]))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=[gas_names[i] for i in range(6)],
           yticklabels=[gas_names[i] for i in range(6)])
plt.title('Baseline Model - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('baseline_confusion_matrix.png', dpi=150)
plt.show()

# ============================================================================
# STEP 4: TEST DIFFERENT MIXING RATIOS
# ============================================================================

print(f"\n{'='*80}")
print("TESTING DIFFERENT REAL/SYNTHETIC MIXING RATIOS")
print("="*80)

def train_with_mix(real_X, real_y, synth_X, synth_y, real_ratio, scaler):
    """Train model with specific mixing ratio"""
    
    # Sample data
    n_real = int(len(real_X) * real_ratio)
    n_synth = int(len(synth_X) * (1 - real_ratio)) if real_ratio < 1.0 else 0
    
    # Get indices
    real_idx = np.random.choice(len(real_X), n_real, replace=False)
    
    if n_synth > 0:
        synth_idx = np.random.choice(len(synth_X), n_synth, replace=False)
        X_mixed = np.concatenate([real_X[real_idx], synth_X[synth_idx]])
        y_mixed = np.concatenate([real_y[real_idx], synth_y[synth_idx]])
    else:
        X_mixed = real_X[real_idx]
        y_mixed = real_y[real_idx]
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_mixed))
    X_mixed = X_mixed[shuffle_idx]
    y_mixed = y_mixed[shuffle_idx]
    
    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_mixed, y_mixed, test_size=0.2, stratify=y_mixed, random_state=42
    )
    
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42
    )
    
    # Normalize
    X_tr_scaled = scaler.transform(X_tr.reshape(-1, features_per_step)).reshape(X_tr.shape)
    X_v_scaled = scaler.transform(X_v.reshape(-1, features_per_step)).reshape(X_v.shape)
    X_te_scaled = scaler.transform(X_te.reshape(-1, features_per_step)).reshape(X_te.shape)
    
    # Build model
    model = build_cnn_lstm_model(input_shape, num_classes)
    
    # Class weights
    cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw_dict = dict(enumerate(cw))
    
    # Train
    early = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=0)
    
    model.fit(
        X_tr_scaled, y_tr,
        validation_data=(X_v_scaled, y_v),
        epochs=80,
        batch_size=32,
        class_weight=cw_dict,
        callbacks=[early, reduce],
        verbose=0
    )
    
    # Evaluate
    y_pred = model.predict(X_te_scaled, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_te, y_pred_classes)
    
    return accuracy

# Test different ratios
ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
results = []

print("\nTesting mixing ratios (this will take 10-15 minutes)...")
for ratio in ratios:
    print(f"\n{'#'*80}")
    print(f"Testing: {ratio*100:.0f}% Real + {(1-ratio)*100:.0f}% Synthetic")
    print(f"{'#'*80}")
    
    # Run 3 times and average (for stability)
    accuracies = []
    for run in range(3):
        print(f"  Run {run+1}/3...", end=' ')
        acc = train_with_mix(real_X_seq, real_y, synth_X_seq, synth_y, ratio, scaler)
        accuracies.append(acc)
        print(f"Accuracy: {acc:.4f}")
    
    avg_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    results.append({
        'Real_Ratio': ratio,
        'Synthetic_Ratio': 1 - ratio,
        'Avg_Accuracy': avg_acc,
        'Std_Accuracy': std_acc,
        'Min_Accuracy': np.min(accuracies),
        'Max_Accuracy': np.max(accuracies)
    })
    
    print(f"  Average: {avg_acc:.4f} ± {std_acc:.4f}")

# ============================================================================
# STEP 5: ANALYZE RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("MIXING RATIO RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Find best
best_idx = results_df['Avg_Accuracy'].idxmax()
best = results_df.iloc[best_idx]

print(f"\n🏆 BEST CONFIGURATION:")
print(f"   Real Ratio: {best['Real_Ratio']*100:.0f}%")
print(f"   Synthetic Ratio: {best['Synthetic_Ratio']*100:.0f}%")
print(f"   Accuracy: {best['Avg_Accuracy']:.4f} ± {best['Std_Accuracy']:.4f}")
print(f"   Range: [{best['Min_Accuracy']:.4f}, {best['Max_Accuracy']:.4f}]")

# Plot results
plt.figure(figsize=(12, 6))
plt.errorbar(results_df['Real_Ratio']*100, results_df['Avg_Accuracy'],
            yerr=results_df['Std_Accuracy'], marker='o', linewidth=2, 
            markersize=8, capsize=5, capthick=2)
plt.axhline(y=0.167, color='red', linestyle='--', label='Random Baseline (16.7%)', alpha=0.5)
plt.axhline(y=baseline_accuracy, color='green', linestyle='--', 
           label=f'Real Only Baseline ({baseline_accuracy:.1%})', alpha=0.5)
plt.xlabel('Real Data Percentage (%)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Performance vs Real/Synthetic Data Ratio', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('mixing_ratio_results.png', dpi=150)
plt.show()

# ============================================================================
# STEP 6: FINAL SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("FINAL SUMMARY")
print("="*80)

improvement = best['Avg_Accuracy'] - 0.23  # vs old simple LSTM
improvement_pct = (improvement / 0.23) * 100

print(f"\n📈 Performance Comparison:")
print(f"   Old Simple LSTM (validation test): 23%")
print(f"   Improved CNN-LSTM (real only): {baseline_accuracy*100:.2f}%")
print(f"   Best Mix ({best['Real_Ratio']*100:.0f}% real): {best['Avg_Accuracy']*100:.2f}%")
print(f"   Improvement: +{improvement*100:.2f}% ({improvement_pct:.1f}% relative gain)")

if best['Avg_Accuracy'] > 0.75:
    print(f"\n✅ EXCELLENT! Model is production-ready")
    recommendation = "Proceed to dashboard development"
elif best['Avg_Accuracy'] > 0.65:
    print(f"\n✓ GOOD! Model is usable")
    recommendation = "Can proceed to dashboard, or improve with ensemble"
elif best['Avg_Accuracy'] > 0.55:
    print(f"\n⚠ MODERATE performance")
    recommendation = "Try ensemble methods or collect more data"
else:
    print(f"\n⚠ Needs improvement")
    recommendation = "Try different architectures or feature engineering"

print(f"\n💡 Recommendation: {recommendation}")

# Save best model configuration
import json
config = {
    'optimal_real_ratio': float(best['Real_Ratio']),
    'optimal_synthetic_ratio': float(best['Synthetic_Ratio']),
    'best_accuracy': float(best['Avg_Accuracy']),
    'baseline_accuracy': float(baseline_accuracy),
    'improvement': float(improvement),
    'timesteps': timesteps,
    'features_per_step': features_per_step,
    'num_classes': num_classes,
    'gas_names': gas_names,
    'recommendation': recommendation
}

with open('best_configuration.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n✓ Configuration saved: best_configuration.json")
print(f"✓ Baseline model saved: best_baseline_model.keras")
print(f"✓ Plots saved: baseline_confusion_matrix.png, mixing_ratio_results.png")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print("="*80)