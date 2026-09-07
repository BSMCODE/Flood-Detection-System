# predict flood file: main deliverable file which the end user can run to predict a flood!

# ========= HOW TO USE =========
# Run this script in the terminal: python predict_flood.py
# Enter values when prompted

import joblib
import numpy as np
import csv
from datetime import datetime
import os


def valid(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value < 0: raise ValueError
            return value
        except:
            print("Please enter a valid non-negative number.")

model = joblib.load("flood_model.pk1")
weights = model['weights'] #imports our b1, b2 etc
bias = model['bias'] # imports b0

print("*******************  WELCOME TO THE FLOOD PREDICTOR!  *******************")

# inputs
rain_today = valid("Today's rainfall (mm): ")
rain_lag1 = valid("Yesterday's rainfall (mm): ")
rain_lag2 = valid("Rainfall 2 days ago (mm): ")
temp_lag1 = valid("Yesterday's temperature (C): ")

rain_3day_sum = rain_today + rain_lag1 + rain_lag2

features = np.array([[rain_lag1, rain_lag2, temp_lag1, rain_3day_sum]])
linear_model = np.dot(features, weights) + bias
p = 1/(1+np.exp(-linear_model)) #calculate p using logistic regression
prediction = int(p >= 0.5)

print(f'\nFlood Prediction for Tomorrow: {'Yes' if prediction else 'No'}')
print(f'Probability of a flood tomorrow: {p[0]:.2f}')
write_header = not os.path.exists('log.csv')  
with open("log.csv", "a", newline='') as file:
    writer = csv.writer(file)

    if write_header:
        writer.writerow([
            "Time", "rain_today_mm", "rain_lag1_mm", "rain_lag2_mm", "temp_lag1_C", "rain_3day_sum_mm", "Flood Probability","Prediction"])
    
    writer.writerow([datetime.now().isoformat(), rain_today, rain_lag1, rain_lag2, temp_lag1, rain_3day_sum, float(p[0]), prediction])


