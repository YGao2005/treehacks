import os
import sys
import numpy as np
import pandas as pd
import argparse
import heartpy as hp
from tqdm import tqdm
import matplotlib.pyplot as plt

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', default="./heartbeat_data/")
    args = parser.parse_args()
    return args

def get_heart_metrics(file_path):
    """
    Analyze PPG and accelerometer data to extract:
    - Heart Rate (HR)
    - Heart Rate Variability (HRV)
    - Movement Metrics
    """
    # Load data
    df = pd.read_csv(file_path)

    # Preprocessing
    ppg_signal = list(df['red'].values)
    accel_x = df['acc_x'].values
    accel_y = df['acc_y'].values
    accel_z = df['acc_z'].values
    sample_rate = len(df) / 60
    
    # 1. Calculate Movement Metrics
    df['acc_magnitude'] = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    movement_metrics = {
        'avg_movement': float(np.mean(df['acc_magnitude'])),
        'max_movement': np.max(df['acc_magnitude']),
        'movement_variability': np.std(df['acc_magnitude'])
    }
    sign = True
    # 2. Process PPG Signal for HR and HRV
    try:
        # Filter PPG signal
        filtered_ppg = hp.filter_signal(ppg_signal, 
                                        cutoff=[0.67, 3.0],
                                      sample_rate=sample_rate, 
                                      filtertype='bandpass')
        
        # Analyze with HeartPy
        wd, m = hp.process(filtered_ppg, sample_rate=sample_rate)

        # Extract metrics
        hr_metrics = {
            'heart_rate': float(m['bpm']),
            'hrv_rmssd': float(m['rmssd']),
            'hrv_sdnn': float(m['sdnn']),
        }
    
    except Exception as e:
        print(f"Error in PPG analysis: {str(e)}")
        hr_metrics = {
            'heart_rate': None,
            'hrv_rmssd': None,
            'hrv_sdnn': None,
            'num_beats': None
        }
        sign = False
    res = {**hr_metrics, **movement_metrics}
    res = {key: [value] for key, value in res.items()}
    return pd.DataFrame(data=res), sign


def get_ppg(in_dir):
    metadata = pd.read_csv(os.path.join(in_dir, "metadata.csv"), on_bad_lines='skip')
    print(len(metadata))

    data = {
        'regular': pd.DataFrame(),
        'irregular': pd.DataFrame(),
        'afib': pd.DataFrame(),
        'unclassified': pd.DataFrame()
    }
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        file = row['filename']
        file_path = file[:file.find('_')]
        try:
            metrics, sign = get_heart_metrics(os.path.join(in_dir, file_path, file))
        except:
            continue
        if sign:
            data[file_path] = pd.concat([data[file_path], metrics], ignore_index=True)

    data['regular'].to_csv('regular_ppg.csv')
    data['irregular'].to_csv('irregular_ppg.csv')
    data['afib'].to_csv('afib_ppg.csv')
    data['unclassified'].to_csv('unclassified_ppg.csv')

    print('regular:', len(data['regular']))
    print('irregular:', len(data['irregular']))
    print('afib:', len(data['afib']))
    print('unclassified:', len(data['unclassified']))


if __name__ == "__main__":
    args = getargs()
    get_ppg(args.in_dir)
