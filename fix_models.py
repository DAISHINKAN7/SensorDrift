#!/usr/bin/env python3
"""
Fix model compatibility issues by converting to TensorFlow 2.15 format
"""
import os
import tensorflow as tf
import numpy as np

def fix_model(model_path):
    """Fix a single model by recreating it"""
    try:
        print(f"Fixing {model_path}...")
        
        # Try to load the model
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Save it again to fix compatibility
        fixed_path = model_path.replace('.keras', '_fixed.keras')
        model.save(fixed_path)
        
        # Replace original
        os.rename(fixed_path, model_path)
        print(f"✅ Fixed {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Could not fix {model_path}: {e}")
        return False

def main():
    model_dir = "/app/model"
    models = [
        "model_optimal_50_50_mix.keras",
        "model_real_only.keras", 
        "model_synthetic_only.keras"
    ]
    
    for model_name in models:
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            fix_model(model_path)
        else:
            print(f"⚠️  Model not found: {model_path}")

if __name__ == "__main__":
    main()
