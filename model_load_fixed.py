import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'

import tensorflow as tf
import numpy as np

model_path = '/Users/kshitijnavale/Desktop/sensor data/model/model_real_only.keras'
model = tf.keras.models.load_model(model_path)
print(model.summary())
