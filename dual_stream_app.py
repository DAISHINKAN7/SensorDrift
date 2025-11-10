from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import threading
import time

app = Flask(__name__)

class DualStreamer:
    def __init__(self):
        self.real_data = pd.read_csv('/Users/kshitijnavale/Desktop/sensor data/data/combined_sensor_data_clean.csv')
        self.synthetic_data = pd.read_csv('/Users/kshitijnavale/Desktop/sensor data/data/realistic_synthetic_gas_300k.csv')
        self.real_index = 0
        self.synthetic_index = 0
        self.current_real = None
        self.current_synthetic = None
        self.streaming = False
        
        # Gas mapping
        self.gas_names = {0: 'Ammonia', 1: 'Acetaldehyde', 2: 'Acetone', 3: 'Ethanol', 4: 'Ethylene', 5: 'Toluene'}
        
    def start_streaming(self):
        self.streaming = True
        threading.Thread(target=self._stream_data, daemon=True).start()
        
    def _stream_data(self):
        while self.streaming:
            # Get real data sample
            if self.real_index < len(self.real_data):
                real_row = self.real_data.iloc[self.real_index]
                gas_label = real_row['label']
                gas_name = self.gas_names.get(gas_label, f'Gas_{gas_label}')
                
                # Extract concentration from first few features (simplified)
                concentration = abs(real_row['f1']) / 100  # Normalize
                
                self.current_real = {
                    'gas_type': gas_name,
                    'concentration': concentration,
                    'label': gas_label,
                    'features': [real_row[f'f{i}'] for i in range(1, 9)]
                }
                self.real_index += 1
            
            # Get synthetic data sample
            if self.synthetic_index < len(self.synthetic_data):
                synth_row = self.synthetic_data.iloc[self.synthetic_index]
                synth_label = synth_row['label']
                synth_gas_name = self.gas_names.get(synth_label, f'Gas_{synth_label}')
                
                # Extract concentration from first few features (simplified)
                synth_concentration = abs(synth_row['f1']) / 1000  # Different normalization
                
                self.current_synthetic = {
                    'gas_type': synth_gas_name,
                    'concentration': synth_concentration,
                    'label': synth_label,
                    'features': [synth_row[f'f{i}'] for i in range(1, 9)]
                }
                self.synthetic_index += 1
                
            time.sleep(0.5)  # 0.5 second intervals for faster streaming

streamer = DualStreamer()

@app.route('/')
def index():
    return render_template('dual_comparison.html')

@app.route('/start_stream')
def start_stream():
    streamer.start_streaming()
    return jsonify({'status': 'started'})

@app.route('/get_data')
def get_data():
    return jsonify({
        'real': streamer.current_real,
        'synthetic': streamer.current_synthetic,
        'timestamp': time.time()
    })

@app.route('/stop_stream')
def stop_stream():
    streamer.streaming = False
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
