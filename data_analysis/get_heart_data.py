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
    print(len(df))

    # Preprocessing
    ppg_signal = list(df['IR'].values)
    print(type(ppg_signal))
    accel_x = df['acc_x'].values
    accel_y = df['acc_y'].values
    accel_z = df['acc_z'].values
    sample_rate = int(len(df) / 60)
    print(sample_rate)
    
    # 1. Calculate Movement Metrics
    df['acc_magnitude'] = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    movement_metrics = {
        'avg_movement': np.mean(df['acc_magnitude']),
        'max_movement': np.max(df['acc_magnitude']),
        'movement_variability': np.std(df['acc_magnitude'])
    }
    
    # 2. Process PPG Signal for HR and HRV
    try:
        # Filter PPG signal
        filtered_ppg = hp.filter_signal(ppg_signal, 
                                      cutoff=[0.5, 5], 
                                      sample_rate=sample_rate, 
                                      filtertype='bandpass')
        
        # Analyze with HeartPy
        wd, m = hp.process(filtered_ppg, sample_rate=sample_rate)
        print(m.keys())
        
        # Extract metrics
        hr_metrics = {
            'heart_rate': m['bpm'],
            'hrv_rmssd': m['rmssd'],
            'hrv_sdnn': m['sdnn'],
        }
        
    except Exception as e:
        print(f"Error in PPG analysis: {str(e)}")
        hr_metrics = {
            'heart_rate': None,
            'hrv_rmssd': None,
            'hrv_sdnn': None,
            'num_beats': None
        }

    return {**hr_metrics, **movement_metrics}, df


def get_ppg(in_dir):
    metadata = pd.read_csv(os.path.join(in_dir, "metadata.csv"), on_bad_lines='skip')

    for idx, row in metadata.iterrows():
        file = row['filename']
        file_path = file[:file.find('_')]
        metrics, df = get_heart_metrics(os.path.join(in_dir, file_path, file))
        print(metrics)
        exit()



if __name__ == "__main__":
    args = getargs()
    get_ppg(args.in_dir)
