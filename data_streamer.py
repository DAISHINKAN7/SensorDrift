import pandas as pd
import numpy as np

class DataStreamer:
    def __init__(self):
        self.datasets = {
            'synthetic': '/Users/kshitijnavale/Desktop/sensor data/data/realistic_synthetic_gas_300k.csv',
            'combined': '/Users/kshitijnavale/Desktop/sensor data/data/combined_sensor_data_clean.csv'
        }
        self.current_data = None
        self.current_index = 0
    
    def load_dataset(self, dataset_name):
        """Load selected dataset"""
        if dataset_name in self.datasets:
            self.current_data = pd.read_csv(self.datasets[dataset_name])
            self.current_index = 0
            return True
        return False
    
    def get_next_sample(self):
        """Get next sample from loaded dataset"""
        if self.current_data is None or self.current_index >= len(self.current_data):
            return None
        
        sample = self.current_data.iloc[self.current_index]
        self.current_index += 1
        return sample
    
    def get_batch(self, size=100):
        """Get batch of samples"""
        if self.current_data is None:
            return None
        
        end_idx = min(self.current_index + size, len(self.current_data))
        batch = self.current_data.iloc[self.current_index:end_idx]
        self.current_index = end_idx
        return batch
