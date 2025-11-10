"""
FINAL MODEL TRAINING SCRIPT
Trains THREE models:
1. Real-only model (99.21% accuracy expected)
2. Synthetic-only model (for comparison)
3. Optimal mix model (50% Real + 50% Synthetic) - 99.91% accuracy expected

All models saved to: /Users/kshitijnavale/Desktop/sensor data/model/

Run time: ~45 minutes total
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import json
import joblib
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data paths
    REAL_DATA_PATH = "/Users/kshitijnavale/Desktop/sensor data/data/combined_sensor_data_clean.csv"
    SYNTH_DATA_PATH = "/Users/kshitijnavale/Desktop/sensor data/better_synthetic_gas_300k.csv"
    
    # Model save directory
    MODEL_DIR = "/Users/kshitijnavale/Desktop/sensor data/model"
    
    # Model parameters
    TIMESTEPS = 16
    FEATURES_PER_STEP = 8
    NUM_CLASSES = 6
    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    
    # Optimal mix ratio (from your results)
    OPTIMAL_REAL_RATIO = 0.5
    OPTIMAL_SYNTH_RATIO = 0.5
    
    # Output files (will be saved in MODEL_DIR)
    REAL_MODEL_NAME = "model_real_only.keras"
    SYNTH_MODEL_NAME = "model_synthetic_only.keras"
    OPTIMAL_MODEL_NAME = "model_optimal_50_50_mix.keras"
    SCALER_NAME = "feature_scaler.pkl"
    CONFIG_NAME = "training_config.json"
    RESULTS_NAME = "training_results.json"

config = Config()

# Create model directory if it doesn't exist
os.makedirs(config.MODEL_DIR, exist_ok=True)
print(f"✓ Model directory: {config.MODEL_DIR}")

# Gas information
GAS_INFO = {
    0: "Ammonia",
    1: "Acetaldehyde", 
    2: "Acetone",
    3: "Ethanol",
    4: "Ethylene",
    5: "Toluene"
}

GAS_DETAILS = {
    0: {"name": "Ammonia", "danger": "Toxic", "threshold": 25, "color": "#3b82f6"},
    1: {"name": "Acetaldehyde", "danger": "Irritant", "threshold": 50, "color": "#8b5cf6"},
    2: {"name": "Acetone", "danger": "Flammable", "threshold": 750, "color": "#ec4899"},
    3: {"name": "Ethanol", "danger": "Flammable", "threshold": 1000, "color": "#10b981"},
    4: {"name": "Ethylene", "danger": "Asphyxiant", "threshold": 100, "color": "#f59e0b"},
    5: {"name": "Toluene", "danger": "Toxic", "threshold": 50, "color": "#ef4444"}
}

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data():
    """Load and preprocess data"""
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    # Load CSVs
    print(f"\nLoading real data from: {config.REAL_DATA_PATH}")
    real_df = pd.read_csv(config.REAL_DATA_PATH)
    print(f"✓ Real data: {real_df.shape}")
    
    print(f"\nLoading synthetic data from: {config.SYNTH_DATA_PATH}")
    synth_df = pd.read_csv(config.SYNTH_DATA_PATH)
    print(f"✓ Synthetic data: {synth_df.shape}")
    
    # Extract features
    feature_cols = [f'f{i}' for i in range(1, 129)]
    
    real_X = real_df[feature_cols].values
    real_y = real_df['label'].values - 1  # Convert 1-6 to 0-5
    
    synth_X = synth_df[feature_cols].values
    synth_y = synth_df['label'].values - 1
    
    print(f"\n✓ Features extracted: {len(feature_cols)} features (f1-f128)")
    print(f"✓ Real labels range: {real_y.min()}-{real_y.max()}")
    print(f"✓ Synthetic labels range: {synth_y.min()}-{synth_y.max()}")
    
    # Class distribution
    print(f"\n📊 Real data class distribution:")
    for cls in range(6):
        count = (real_y == cls).sum()
        pct = count / len(real_y) * 100
        print(f"   {GAS_INFO[cls]}: {count:,} ({pct:.1f}%)")
    
    print(f"\n📊 Synthetic data class distribution:")
    for cls in range(6):
        count = (synth_y == cls).sum()
        pct = count / len(synth_y) * 100
        print(f"   {GAS_INFO[cls]}: {count:,} ({pct:.1f}%)")
    
    # Reshape to 3D time series (samples, timesteps, features)
    print(f"\nReshaping to time series format...")
    real_X_seq = real_X.reshape(-1, config.TIMESTEPS, config.FEATURES_PER_STEP)
    synth_X_seq = synth_X.reshape(-1, config.TIMESTEPS, config.FEATURES_PER_STEP)
    
    print(f"✓ Real data shape: {real_X_seq.shape} → (samples, timesteps={config.TIMESTEPS}, features={config.FEATURES_PER_STEP})")
    print(f"✓ Synthetic data shape: {synth_X_seq.shape}")
    
    return real_X_seq, real_y, synth_X_seq, synth_y

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_cnn_lstm_model():
    """Build CNN-LSTM hybrid model (proven architecture from your results)"""
    model = models.Sequential([
        # CNN layers for feature extraction
        layers.Conv1D(64, 3, activation='relu', padding='same', 
                     input_shape=(config.TIMESTEPS, config.FEATURES_PER_STEP)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # LSTM layers for temporal dependencies
        layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3)),
        layers.Bidirectional(layers.LSTM(32, dropout=0.3)),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(config.NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(X_data, y_data, model_name, save_name, scaler=None):
    """Train and evaluate a single model"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, stratify=y_data, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    print(f"\n📊 Data splits:")
    print(f"   Training:   {len(X_train):,} samples")
    print(f"   Validation: {len(X_val):,} samples")
    print(f"   Test:       {len(X_test):,} samples")
    
    # Normalize data
    print(f"\nNormalizing data...")
    if scaler is None:
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, config.FEATURES_PER_STEP)
        scaler.fit(X_train_flat)
    
    X_train_scaled = scaler.transform(X_train.reshape(-1, config.FEATURES_PER_STEP)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, config.FEATURES_PER_STEP)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, config.FEATURES_PER_STEP)).reshape(X_test.shape)
    
    print("✓ Data normalized")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', 
                                        classes=np.unique(y_train), 
                                        y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"\n⚖️  Class weights:")
    for cls, weight in class_weight_dict.items():
        print(f"   {GAS_INFO[cls]}: {weight:.3f}")
    
    # Build model
    print(f"\n🏗️  Building model...")
    model = build_cnn_lstm_model()
    
    print(f"\n📋 Model Architecture:")
    model.summary()
    
    # Callbacks
    model_path = os.path.join(config.MODEL_DIR, save_name)
    
    model_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print(f"\n🚀 Training started at {datetime.now().strftime('%H:%M:%S')}")
    print(f"   Max epochs: {config.EPOCHS}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Using class weights: Yes")
    print(f"\nThis will take approximately 10-15 minutes...\n")
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=model_callbacks,
        verbose=1
    )
    
    print(f"\n✓ Training completed at {datetime.now().strftime('%H:%M:%S')}")
    
    # Evaluate on test set
    print(f"\n{'='*80}")
    print("EVALUATION ON TEST SET")
    print("="*80)
    
    y_pred = model.predict(X_test_scaled, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\n📊 Classification Report:")
    print("-"*80)
    report = classification_report(y_test, y_pred_classes, 
                                   target_names=[GAS_INFO[i] for i in range(6)])
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[GAS_INFO[i] for i in range(6)],
               yticklabels=[GAS_INFO[i] for i in range(6)])
    plt.title(f'{model_name} - Confusion Matrix\nAccuracy: {accuracy:.2%}', 
             fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(config.MODEL_DIR, f'confusion_matrix_{save_name.replace(".keras", ".png")}')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {cm_path}")
    
    # Training history plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_path = os.path.join(config.MODEL_DIR, f'training_history_{save_name.replace(".keras", ".png")}')
    plt.savefig(history_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved: {history_path}")
    
    print(f"\n✓ Model saved: {model_path}")
    
    # Return results
    results = {
        'model_name': model_name,
        'save_path': model_path,
        'test_accuracy': float(accuracy),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'epochs_trained': len(history.history['accuracy']),
        'final_train_acc': float(history.history['accuracy'][-1]),
        'final_val_acc': float(history.history['val_accuracy'][-1]),
        'best_val_acc': float(max(history.history['val_accuracy'])),
        'timestamp': datetime.now().isoformat()
    }
    
    return model, scaler, results, history

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "🚀"*40)
    print("GAS SENSOR CLASSIFICATION - FINAL MODEL TRAINING")
    print("🚀"*40)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models will be saved to: {config.MODEL_DIR}")
    
    # Load data
    real_X, real_y, synth_X, synth_y = load_data()
    
    all_results = {}
    
    # ========================================================================
    # MODEL 1: REAL DATA ONLY
    # ========================================================================
    
    print(f"\n{'#'*80}")
    print("MODEL 1/3: TRAINING ON REAL DATA ONLY")
    print(f"{'#'*80}")
    
    model_real, scaler_real, results_real, history_real = train_model(
        real_X, real_y,
        model_name="Real Data Only Model",
        save_name=config.REAL_MODEL_NAME
    )
    
    all_results['real_only'] = results_real
    
    # ========================================================================
    # MODEL 2: SYNTHETIC DATA ONLY
    # ========================================================================
    
    print(f"\n{'#'*80}")
    print("MODEL 2/3: TRAINING ON SYNTHETIC DATA ONLY")
    print(f"{'#'*80}")
    
    # Use same number of samples as real data for fair comparison
    n_synth_samples = len(real_X)
    synth_idx = np.random.choice(len(synth_X), n_synth_samples, replace=False)
    synth_X_sampled = synth_X[synth_idx]
    synth_y_sampled = synth_y[synth_idx]
    
    print(f"\n📊 Using {n_synth_samples:,} synthetic samples (same as real data)")
    
    model_synth, scaler_synth, results_synth, history_synth = train_model(
        synth_X_sampled, synth_y_sampled,
        model_name="Synthetic Data Only Model",
        save_name=config.SYNTH_MODEL_NAME,
        scaler=scaler_real  # Use same scaler as real data
    )
    
    all_results['synthetic_only'] = results_synth
    
    # ========================================================================
    # MODEL 3: OPTIMAL MIX (50% Real + 50% Synthetic)
    # ========================================================================
    
    print(f"\n{'#'*80}")
    print("MODEL 3/3: TRAINING ON OPTIMAL MIX (50% REAL + 50% SYNTHETIC)")
    print(f"{'#'*80}")
    
    # Mix data
    n_real = len(real_X) // 2
    n_synth = 50000  # Take 50k from synthetic (plenty available)
    
    print(f"\n📊 Mixing data:")
    print(f"   Real samples: {n_real:,}")
    print(f"   Synthetic samples: {n_synth:,}")
    print(f"   Total: {n_real + n_synth:,}")
    
    real_idx = np.random.choice(len(real_X), n_real, replace=False)
    synth_idx = np.random.choice(len(synth_X), n_synth, replace=False)
    
    X_mixed = np.concatenate([real_X[real_idx], synth_X[synth_idx]])
    y_mixed = np.concatenate([real_y[real_idx], synth_y[synth_idx]])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_mixed))
    X_mixed = X_mixed[shuffle_idx]
    y_mixed = y_mixed[shuffle_idx]
    
    model_optimal, scaler_optimal, results_optimal, history_optimal = train_model(
        X_mixed, y_mixed,
        model_name="Optimal Mix Model (50% Real + 50% Synthetic)",
        save_name=config.OPTIMAL_MODEL_NAME,
        scaler=scaler_real  # Use same scaler
    )
    
    all_results['optimal_mix'] = results_optimal
    
    # ========================================================================
    # SAVE ARTIFACTS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("SAVING ARTIFACTS")
    print("="*80)
    
    # Save scaler
    scaler_path = os.path.join(config.MODEL_DIR, config.SCALER_NAME)
    joblib.dump(scaler_real, scaler_path)
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Save configuration
    config_dict = {
        'timesteps': config.TIMESTEPS,
        'features_per_step': config.FEATURES_PER_STEP,
        'num_classes': config.NUM_CLASSES,
        'gas_info': GAS_INFO,
        'gas_details': GAS_DETAILS,
        'real_data_path': config.REAL_DATA_PATH,
        'synth_data_path': config.SYNTH_DATA_PATH,
        'model_dir': config.MODEL_DIR,
        'training_date': datetime.now().isoformat()
    }
    
    config_path = os.path.join(config.MODEL_DIR, config.CONFIG_NAME)
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ Configuration saved: {config_path}")
    
    # Save results
    results_path = os.path.join(config.MODEL_DIR, config.RESULTS_NAME)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Results saved: {results_path}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("="*80)
    
    print(f"\n📊 Model Performance Comparison:")
    print("-"*80)
    print(f"{'Model':<40} {'Accuracy':<15} {'Samples':<15}")
    print("-"*80)
    print(f"{'1. Real Data Only':<40} {all_results['real_only']['test_accuracy']:.4f} ({all_results['real_only']['test_accuracy']*100:.2f}%)"
          f"   {all_results['real_only']['training_samples']:>10,}")
    print(f"{'2. Synthetic Data Only':<40} {all_results['synthetic_only']['test_accuracy']:.4f} ({all_results['synthetic_only']['test_accuracy']*100:.2f}%)"
          f"   {all_results['synthetic_only']['training_samples']:>10,}")
    print(f"{'3. Optimal Mix (50/50)':<40} {all_results['optimal_mix']['test_accuracy']:.4f} ({all_results['optimal_mix']['test_accuracy']*100:.2f}%)"
          f"   {all_results['optimal_mix']['training_samples']:>10,}")
    print("-"*80)
    
    best_model = max(all_results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"\n🏆 Best Model: {best_model[0].replace('_', ' ').title()}")
    print(f"   Accuracy: {best_model[1]['test_accuracy']:.4f} ({best_model[1]['test_accuracy']*100:.2f}%)")
    
    print(f"\n📁 All files saved to: {config.MODEL_DIR}")
    print(f"\n✅ Models:")
    print(f"   • {config.REAL_MODEL_NAME}")
    print(f"   • {config.SYNTH_MODEL_NAME}")
    print(f"   • {config.OPTIMAL_MODEL_NAME}")
    
    print(f"\n✅ Supporting files:")
    print(f"   • {config.SCALER_NAME}")
    print(f"   • {config.CONFIG_NAME}")
    print(f"   • {config.RESULTS_NAME}")
    print(f"   • confusion_matrix_*.png (3 files)")
    print(f"   • training_history_*.png (3 files)")
    
    print(f"\n{'='*80}")
    print("🎉 SUCCESS! All models trained and ready for dashboard deployment")
    print("="*80)
    print(f"\n💡 Next step: Run the dashboard application to use these models")
    print(f"   Dashboard will load models from: {config.MODEL_DIR}")
    
    return all_results

# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    try:
        results = main()
        print(f"\n✅ Training pipeline completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()