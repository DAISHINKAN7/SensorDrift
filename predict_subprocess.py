"""
Subprocess script to make predictions without mutex issues
This runs in a separate process each time
"""

import sys
import os
import numpy as np
import json

# Set environment before TF import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tensorflow as tf

# Configure TF
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

MODEL_DIR = "/Users/kshitijnavale/Desktop/sensor data/model"

def predict(model_type, input_path):
    """Load model and predict"""
    
    # Load model
    model_files = {
        'real': 'model_real_only.keras',
        'synthetic': 'model_synthetic_only.keras',
        'optimal': 'model_optimal_50_50_mix.keras'
    }
    
    model_path = os.path.join(MODEL_DIR, model_files[model_type])
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Load input
    X = np.load(input_path)
    
    # Predict
    predictions = model.predict(X, verbose=0)[0]
    
    predicted_class = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class])
    
    # Return as JSON
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': predictions.tolist()
    }
    
    print(json.dumps(result))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python predict_subprocess.py <model_type> <input_path>")
        sys.exit(1)
    
    model_type = sys.argv[1]
    input_path = sys.argv[2]
    
    predict(model_type, input_path)