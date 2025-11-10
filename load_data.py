import pandas as pd

def load_sensor_data():
    # Load both datasets
    df1 = pd.read_csv('/Users/kshitijnavale/Desktop/sensor data/data/realistic_synthetic_gas_300k.csv')
    df2 = pd.read_csv('/Users/kshitijnavale/Desktop/sensor data/data/combined_sensor_data_clean.csv')
    
    # Combine datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    return combined_df

# Usage
data = load_sensor_data()
print(f"Total samples: {len(data)}")
print(f"Features: {data.columns.tolist()}")
