#!/usr/bin/env python3
"""
Create compatible models for SensorDrift dashboard
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os

def load_data():
    """Load and combine datasets"""
    print("📊 Loading datasets...")
    
    # Load real data
    real_df = pd.read_csv('/app/data/combined_sensor_data_clean.csv')
    print(f"✅ Real data: {real_df.shape}")
    
    # Load synthetic data
    synth_df = pd.read_csv('/app/data/realistic_synthetic_gas_300k.csv')
    print(f"✅ Synthetic data: {synth_df.shape}")
    
    return real_df, synth_df

def preprocess_data(real_df, synth_df):
    """Preprocess and combine data"""
    print("🔧 Preprocessing data...")
    
    # Get feature columns (first 8 features for sensors)
    feature_cols = [f'f{i}' for i in range(1, 9)]
    
    # Get common columns
    common_cols = list(set(real_df.columns) & set(synth_df.columns))
    feature_cols = [col for col in feature_cols if col in common_cols]
    
    # Extract features and labels
    real_X = real_df[feature_cols].values
    synth_X = synth_df[feature_cols].values
    
    # Get labels
    if 'gas' in real_df.columns and 'gas' in synth_df.columns:
        real_y = real_df['gas'].values
        synth_y = synth_df['gas'].values
    elif 'label' in real_df.columns and 'label' in synth_df.columns:
        real_y = real_df['label'].values
        synth_y = synth_df['label'].values
    else:
        # Create dummy labels based on data source
        real_y = np.zeros(len(real_df))
        synth_y = np.ones(len(synth_df))
    
    # Combine
    X = np.vstack([real_X, synth_X])
    y = np.hstack([real_y, synth_y])
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Combined data: {X_scaled.shape}, Classes: {len(le.classes_)}")
    
    return X_scaled, y_encoded, scaler, le

def create_model(input_shape, num_classes):
    """Create a simple, compatible model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_save_models():
    """Train and save compatible models"""
    # Load data
    real_df, synth_df = load_data()
    X, y, scaler, le = preprocess_data(real_df, synth_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create models
    input_shape = X.shape[1]
    num_classes = len(np.unique(y))
    
    print(f"🤖 Training models with {input_shape} features, {num_classes} classes...")
    
    # Model 1: Real + Synthetic (Optimal)
    model_optimal = create_model(input_shape, num_classes)
    model_optimal.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
    
    # Model 2: Real only
    real_indices = np.arange(len(real_df))
    X_real = X[real_indices]
    y_real = y[real_indices]
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
    
    model_real = create_model(input_shape, num_classes)
    model_real.fit(X_real_train, y_real_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
    
    # Model 3: Synthetic only
    synth_indices = np.arange(len(real_df), len(X))
    X_synth = X[synth_indices]
    y_synth = y[synth_indices]
    X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(X_synth, y_synth, test_size=0.2, random_state=42)
    
    model_synth = create_model(input_shape, num_classes)
    model_synth.fit(X_synth_train, y_synth_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
    
    # Save models
    model_dir = '/app/model'
    
    # Remove old models
    old_models = ['model_optimal_50_50_mix.keras', 'model_real_only.keras', 'model_synthetic_only.keras']
    for old_model in old_models:
        old_path = os.path.join(model_dir, old_model)
        if os.path.exists(old_path):
            os.remove(old_path)
            print(f"🗑️  Removed old model: {old_model}")
    
    # Save new models
    model_optimal.save(os.path.join(model_dir, 'model_optimal_50_50_mix.keras'))
    model_real.save(os.path.join(model_dir, 'model_real_only.keras'))
    model_synth.save(os.path.join(model_dir, 'model_synthetic_only.keras'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(model_dir, 'feature_scaler.pkl'))
    
    # Test accuracies
    acc_optimal = model_optimal.evaluate(X_test, y_test, verbose=0)[1]
    acc_real = model_real.evaluate(X_real_test, y_real_test, verbose=0)[1]
    acc_synth = model_synth.evaluate(X_synth_test, y_synth_test, verbose=0)[1]
    
    print(f"✅ Optimal model accuracy: {acc_optimal:.3f}")
    print(f"✅ Real model accuracy: {acc_real:.3f}")
    print(f"✅ Synthetic model accuracy: {acc_synth:.3f}")
    print(f"✅ Models saved to {model_dir}")

if __name__ == "__main__":
    train_and_save_models()
