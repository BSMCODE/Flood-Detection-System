import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import joblib
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score
from plot import plot_catchment
from flood_model import sigmoid, logistic_regression_train, logistic_regression_predict
from sklearn.metrics import confusion_matrix

# loading data into a dataframe
sf = pd.read_csv('data/streamflow_TOP5.csv', parse_dates=['date']) # sf = streamflow!!
rain = pd.read_csv('data/precipitation_TOP5.csv', parse_dates=['date'])
temp = pd.read_csv('data/temperature_TOP5.csv', parse_dates=['date'])

# Melt each to long format
sf_l = sf.melt(id_vars='date', var_name='catchment_id', value_name='streamflow')
rain_l = rain.melt(id_vars='date', var_name='catchment_id', value_name='precipitation')
temp_l = temp.melt(id_vars='date', var_name='catchment_id', value_name='temperature')

# create a mega-dataframe to hold our data
df = pd.merge(sf_l, rain_l, on=['date', 'catchment_id'])
df = pd.merge(df, temp_l, on=['date', 'catchment_id'])

# adding lagged features

# Sort data to ensure correct lagging
df = df.sort_values(by=['catchment_id', 'date'])

# Create lagged features within each catchment group
df['rain_lag1'] = df.groupby('catchment_id')['precipitation'].shift(1) # rain on previous day
df['rain_lag2'] = df.groupby('catchment_id')['precipitation'].shift(2) # rain 2 days before
df['temp_lag1'] = df.groupby('catchment_id')['temperature'].shift(1) # temp on previous day

# rain 3 day sum:

df['rain_3day_sum'] = (
    df.groupby('catchment_id')['precipitation']
    .shift(1)
    .rolling(window=3, min_periods=1)
    .sum()
    .reset_index(level=0, drop=True)
)

# Definition of 'flood day': where streamflow on that day > 95th percentile of all streamflow in that catchment

def find_95(streamflow_vals):
    return np.percentile(streamflow_vals, 95)


thresholds = {
    "912101A": find_95(df[df["catchment_id"] == "912101A"]['streamflow'].tolist()),
    "912105A": find_95(df[df["catchment_id"] == "912105A"]['streamflow'].tolist()),
    "915011A": find_95(df[df["catchment_id"] == "915011A"]['streamflow'].tolist()),
    "915206A": find_95(df[df["catchment_id"] == "915206A"]['streamflow'].tolist()),
    "917107A": find_95(df[df["catchment_id"] == "917107A"]['streamflow'].tolist())
}

df['threshold'] = df['catchment_id'].map(thresholds) # store the threshold value for each catchment in the dataframe
df['flood'] = df['streamflow'] > df['threshold'] # set flood value in dataframe to be True if streamflow on that day exceeds flood threshold

# Plotting: we plot rainfall and temperature on the same plot as streamflow to show how these variables correlate with flooding
for c in df['catchment_id'].unique():
    plot_catchment(df, c)

# run the model:

df_c = df.dropna(subset=['rain_lag1', 'rain_lag2', 'temp_lag1', 'rain_3day_sum', 'flood']).copy()
# Define input and target
features = df_c[['rain_lag1', 'rain_lag2', 'temp_lag1', 'rain_3day_sum']].values
target = df_c['flood'].astype(int).values

# Split: 80% train, 20% test (no shuffle, keep temporal order)
split_idx = int(0.8 * len(features))
X_train, X_test = features[:split_idx], features[split_idx:]
y_train, y_test = target[:split_idx], target[split_idx:]

# Train model
weights, bias = logistic_regression_train(X_train, y_train)

# Predict
y_pred = logistic_regression_predict(X_test, weights, bias) #y_pred is 

# Evaluate accuracy, precision and recall values, and F-1 recall_score
accuracy = accuracy_score(y_test, y_pred) 
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

f1_score = (2* precision * recall)/(precision + recall)

# plot predicted vs actual flood days for all 5 catchments' testing data on the same confusion matrix
# add plot code here from scratch in this file main.py

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Flood', 'Flood'], 
            yticklabels=['No Flood', 'Flood'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Actual vs Flood Prediction For Testing Data From 5 Catchments')
plt.tight_layout()
plt.show()
plt.savefig('Predicted vs Actual Flood Days.png') # save confusion matrix


# true positive, true negative, false positives, false negatives
TP = np.sum((y_test == 1) & (y_pred == 1)) 
TN = np.sum((y_test == 0) & (y_pred == 0)) 
FP = np.sum((y_test == 0) & (y_pred == 1))
FN = np.sum((y_test == 1) & (y_pred == 0))
print(f"Confusion Matrix:\nTP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

# saving the model
joblib.dump({'weights': weights, 'bias': bias}, 'flood_model.pk1')
print('Model updated!')

#display weight+bias+variables
print(f"Bias (b0): {bias:.4f}")
for i, w in enumerate(weights, start=1):
    print(f"Weight b{i}: {w:.4f}")

# print of  evalutaion 
print('')
print(f'Accuracy Value: {accuracy:.3f}')
print(f'Precision Value: {precision:.3f}')
print(f'Recall Value: {recall:.3f}')
print(f'F1 Score : {f1_score:.3f}')

# generate predicted vs actual flood for random catchment using our model

df_224 = pd.read_csv('data/224213_data.csv', parse_dates=['date'])
df_224 = df_224.sort_values(by='date')

df_224['actual_flood'] = df_224['streamflow'] > np.percentile(df_224['streamflow'], 95)
df_224['rain_lag1'] = df_224['rainfall'].shift(1)
df_224['rain_lag2'] = df_224['rainfall'].shift(2)
df_224['rain_3day_sum'] = df_224['rainfall'].shift(1).rolling(window=3, min_periods=1).sum()
df_224['temp_lag1'] = df_224['max_temp'].shift(1)

df_224_clean = df_224.dropna(subset=['rain_lag1', 'rain_lag2', 'rain_3day_sum', 'temp_lag1'])

X = df_224_clean[['rain_lag1', 'rain_lag2', 'temp_lag1', 'rain_3day_sum']].values

df_224_clean['predicted_flood'] = logistic_regression_predict(X, weights, bias)

# plotting
cm = confusion_matrix(df_224_clean['actual_flood'], df_224_clean['predicted_flood'])

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['No Flood', 'Flood'], 
            yticklabels=['No Flood', 'Flood'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Catchment 224213')
plt.tight_layout()
plt.savefig('test_confusion_matrix.png')
plt.show()

print('------------ EVALUATION of MODEL PERFORMANCE on NEW CATCHMENT - DARGO RIVER, VICTORIA ------------')
print('')
print(f'Accuracy: {accuracy_score(df_224_clean['actual_flood'], df_224_clean['predicted_flood'])*100:.3f}%')
print(f'Precision: % of predicted floods that were actual floods: {precision_score(df_224_clean['actual_flood'], df_224_clean['predicted_flood'])*100:.3f}%')
print(f'Recall: % of actual floods the model correctly found: {recall_score(df_224_clean['actual_flood'], df_224_clean['predicted_flood'])*100:.3f}%')

# plot as a timeline for a small range of recent dates
df_zoom = df_224_clean[df_224_clean['date'].between('2010-01-01', '2022-01-01')]
plt.figure(figsize=(14, 2))

actual_y = df_zoom['actual_flood']
predicted_y = df_zoom['predicted_flood'] + 0.15 # add an offset val

plt.scatter(df_zoom['date'], actual_y, label='Actual Flood', s=10, color='blue', alpha=0.4)
plt.scatter(df_zoom['date'], predicted_y, label='Predicted Flood', s=10, color='orange', alpha=0.4)

plt.yticks([0, 0.15, 1, 1.15], ['No (Actual)', 'No (Pred)', 'Yes (Actual)', 'Yes (Pred)'])
plt.ylim(-0.2, 1.3)
plt.xlim(df_zoom['date'].min(), df_zoom['date'].max())

plt.title("Floods Predicted By Model vs Actual Floods (2010–2022) — Dargo River, Victoria")
plt.xlabel("Date")
plt.ylabel("Flood Status")
plt.legend()
plt.tight_layout()
plt.savefig("224213_timeseries.png")
plt.show()
