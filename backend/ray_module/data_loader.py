import os
import json
import pandas as pd

def load_data(dataset_file):
    data_path = os.path.join("uploads", "datasets", dataset_file)

    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
        texts = [item['text'] for item in data]
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        texts = df['text'].tolist()
    else:
        raise ValueError("Unsupported file format. Please upload a JSON or CSV file.")
    
    return texts
