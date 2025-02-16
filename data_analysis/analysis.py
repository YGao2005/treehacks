import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Load data
def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_path', default="./heartbeat_data/")
    parser.add_argument('--all', action="store_true")
    args = parser.parse_args()
    return args

def visualize_instance(in_dir):
    df = pd.read_csv(os.path.join(in_dir, "regular", "regular_10.csv"))

# Calculate accelerometer magnitude
    df['acc_magnitude'] = (df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)**0.5

# Plot
    plt.figure(figsize=(15, 6))
    plt.plot(df['sample_index'], df['green'], label='PPG (Green)', alpha=0.7)
#    plt.plot(df['sample_index'], df['red'], label='PPG (Red)', alpha=0.7)
#    plt.plot(df['sample_index'], df['IR'], label='PPG (IR)', alpha=0.7)
    plt.plot(df['sample_index'], df['acc_magnitude'], label='Acceleration', alpha=0.7)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude / Acceleration (g)')
    plt.legend()
    plt.title('Heart Rate vs. Physical Activity')
    plt.show()

def visualize_data(in_file):
    plt.figure()
    df = pd.read_csv(in_file)
    label_colors = ['red', 'green', 'blue']
    legend_labels = [False, False, False]
    label_names = ['regular', 'irregular', 'afib']

    for idx, row in df.iterrows():
        bpm = row['heart_rate']
        hrv_rmssd = row['hrv_rmssd']
        hrv_sdnn = row['hrv_sdnn']
        label = int(row['label'])
        if not legend_labels[label]:
            legend_labels[label] = True
            cur_label = label_names[label]
        else:
            cur_label = None
        plt.plot(hrv_rmssd, bpm, marker='o', linestyle=None,
                 color=label_colors[label], label=cur_label)
    plt.xlabel('HRV rmssd')
    plt.ylabel('bpm')
    plt.legend(title="Legend", loc='upper right')
    plt.title('hrv vs bpm')
    plt.show()


if __name__ == "__main__":
    args = getargs()
    if args.all:
        args.in_path = './dataset_ppg.csv'
        visualize_data(args.in_path)
    visualize_data(args.in_path)
