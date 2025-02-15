import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Load data
def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', default="./heartbeat_data/")
    args = parser.parse_args()
    return args

def visualize(in_dir):
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


if __name__ == "__main__":
    args = getargs()
    visualize(args.in_dir)
    
