# Plot.py: python file which holds out plot functions!
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def find_95(streamflow_vals):
    return np.percentile(streamflow_vals, 95)

# Plotting function
def plot_catchment(df, catchment):

    df_catch = df[df['catchment_id'] == catchment]
    df_sf = df_catch[df_catch['streamflow'] > -99.99].copy()
    df_sf['streamflow'] += 0.01  # Avoid log(0) errors by adding a small amount (0.01)

    threshold = find_95(df_sf['streamflow'].values)
    df_sf['flood'] = df_sf['streamflow'] >= threshold # streamflow only dataframe
    df_rain = df_catch[df_catch['precipitation'] > -99.99].copy() # rainfall only dataframe
    df_temp = df_catch[df_catch['temperature'] > -99.99].copy() # temperature only dataframe

    # plot streamflow with left side y axis
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(df_sf['date'], df_sf['streamflow'], color='blue', label='Streamflow')
    ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Flood Threshold ({threshold:.0f})')
    ax1.set_yscale('log')
    ax1.set_ylabel('Streamflow (ML/day)')
    ax1.set_xlabel('Date')
    ax1.set_xlim(df_sf['date'].min(), df_sf['date'].max())

    # plot the rainfall with the right side y axis
    ax2 = ax1.twinx()
    ax2.bar(df_rain['date'], df_rain['precipitation'], color='red', label='Precipitation')
    ax2.set_ylabel('Rainfall (mm)')
    maxrain = df_rain['precipitation'].max()
    ax2.set_ylim(0, maxrain)

    # add legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # plot and save figure
    plt.title(f'Streamflow and Rainfall in Catchment {catchment}')
    plt.tight_layout()
    plt.savefig(f'plots/{catchment}_sf_rainfall_combined.png')
    plt.close()

    # do the same thing but now plot streamflow with max tempatures
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(df_sf['date'], df_sf['streamflow'], color='blue', label='Streamflow')
    ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Flood Threshold ({threshold:.0f})')
    ax1.set_yscale('log')
    ax1.set_ylabel('Streamflow (ML/day)')
    ax1.set_xlabel('Date')
    ax1.set_xlim(df_sf['date'].min(), df_sf['date'].max())

    ax2 = ax1.twinx()
    ax2.plot(df_temp['date'], df_temp['temperature'], color='green', alpha=0.4, linewidth=0.8, label='Max Temp')
    ax2.set_ylabel('Temperature (C)')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(f'Streamflow and Temperature in Catchment {catchment}')
    plt.tight_layout()
    plt.savefig(f'plots/{catchment}_sf_temp_combined.png')
    plt.close()


