import os
import sys
import json
import numpy as np
import argparse

def get_stress_level(in_dir='.'):
    data = json.load(open(os.path.join(in_dir,'message2.txt'), 'r'))

    heart_rate_data = data['data'][0]['heart_data']['heart_rate_data']['summary']
    avg_hrv_rmssd = heart_rate_data['avg_hrv_rmssd']
    avg_hrv_sdnn = heart_rate_data['avg_hrv_sdnn']
    avg_bpm = heart_rate_data['avg_hr_bpm']

    # NOTE: hr metrics
    hrv_sdnn_metric = 10 * (avg_hrv_sdnn)/150
    hr_fluctuations_metric = 10 * (heart_rate_data['max_hr_bpm'] - heart_rate_data['min_hr_bpm'])/100

    # NOTE: blood pressure data
    blood_pressure_samples = data['data'][0]['blood_pressure_data']['blood_pressure_samples']
    avg_diastolic_bp = sum(sample["diastolic_bp"] for sample in blood_pressure_samples) / len(blood_pressure_samples)
    avg_systolic_bp = sum(sample["systolic_bp"] for sample in blood_pressure_samples) / len(blood_pressure_samples)
    blood_pressure_metric = 10 * (1-(avg_systolic_bp/180 + avg_diastolic_bp/120)/2)

    # NOTE: glucose data
    glucose_values = [sample["blood_glucose_mg_per_dL"] for sample in data['data'][0]["glucose_data"]["blood_glucose_samples"]]
    mean_glucose = sum(glucose_values) / len(glucose_values)
    var_glucose = sum((x - mean_glucose) ** 2 for x in glucose_values) / len(glucose_values)
    std_glucose = float(np.sqrt(var_glucose))
    glucose_metric = 10 * (std_glucose/60)

    # NOTE: temperature

    temp_values = [sample["temperature_celsius"] for sample in data['data'][0]["temperature_data"]["body_temperature_samples"]]
    mean_temp  = sum(temp_values) / len(temp_values)
#    var_temp = sum((x - mean_temp) ** 2 for x in temp_values) / len(temp_values)
#    std_temp = float(np.sqrt(var_temp))
    temp_metric = 10 * (abs(37-mean_temp))/3

    # NOTE: oxygen
    oxygen_values = [sample["percentage"] for sample in data['data'][0]["oxygen_data"]["saturation_samples"]]
    low_oxy_count = sum(i < 90 for i in set(oxygen_values))
    oxygen_metric = 10 * (1 - (low_oxy_count/len(oxygen_values)))

    # NOTE: hydration value
    stress_val = hr_fluctuations_metric* 0.25 + blood_pressure_metric * 0.2 + glucose_metric * 0.15 + \
        temp_metric * 0.15 + hrv_sdnn_metric * 0.15 + oxygen_metric * 0.1
    return stress_val / 10


if __name__ == "__main__":
    get_stress_level()
