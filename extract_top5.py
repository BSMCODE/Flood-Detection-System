# This is code I used to extract data from 5 catchments from each of the relevant files in the CAMELS AUS dataset
# Note: the filepaths are specific to my computer
# This produced the files under the data folder in this Ed workspace

import pandas as pd
import os

#top 5 catchments (match column names exactly)
top_ids = ['912101A', '912105A', '915011A', '915206A', '917107A']

# streamflow
sf = pd.read_csv(r"C:\Users\hawts\OneDrive\Documents\Personal\USYD\Sem 1 2025\ENGG1810\Assessment\Grand_challenge\data\03_streamflow\03_streamflow\streamflow_MLd_inclInfilled.csv")
sf['date'] = pd.to_datetime(sf[['year', 'month', 'day']])
sf_top5 = sf[['date'] + top_ids]

sf_top5.to_csv('data_small/streamflow_TOP5.csv', index=False)

# ppt
precip = pd.read_csv(r"C:\Users\hawts\OneDrive\Documents\Personal\USYD\Sem 1 2025\ENGG1810\Assessment\Grand_challenge\data\05_hydrometeorology\05_hydrometeorology\01_precipitation_timeseries\precipitation_AGCD.csv")
precip['date'] = pd.to_datetime(precip[['year', 'month', 'day']])
precip_top5 = precip[['date'] + top_ids]
precip_top5.to_csv('data_small/precipitation_TOP5.csv', index=False)

# temp
temp = pd.read_csv(r"C:\Users\hawts\OneDrive\Documents\Personal\USYD\Sem 1 2025\ENGG1810\Assessment\Grand_challenge\data\05_hydrometeorology\05_hydrometeorology\03_Other\AGCD\tmax_AGCD.csv")
temp['date'] = pd.to_datetime(temp[['year', 'month', 'day']])
temp_top5 = temp[['date'] + top_ids]
temp_top5.to_csv('data_small/temperature_TOP5.csv', index=False)

# metadata for each catchment!
meta = pd.read_csv(r"C:\Users\hawts\OneDrive\Documents\Personal\USYD\Sem 1 2025\ENGG1810\Assessment\Grand_challenge\data\01_id_name_metadata\01_id_name_metadata\id_name_metadata.csv")
meta['station_id'] = meta['station_id'].astype(str).str.strip()
top_ids_cleaned = [cid.strip() for cid in top_ids]
meta_top5 = meta[meta['station_id'].isin(top_ids_cleaned)]
meta_top5.to_csv('data_small/metadata_TOP5.csv', index=False)

# sigs
signatures_path = (r"C:\Users\hawts\OneDrive\Documents\Personal\USYD\Sem 1 2025\ENGG1810\Assessment\Grand_challenge\data\03_streamflow\03_streamflow\streamflow_signatures.csv")
signatures = pd.read_csv(signatures_path)
signatures['CatchID'] = signatures['CatchID'].astype(str).str.strip()
sig_top5 = signatures[signatures['CatchID'].isin(top_ids)]
sig_top5.to_csv('data_small/streamflow_signatures_TOP5.csv', index=False)

# Additonal catchment: load a new, random test catchment to plot the accuracy of our prediction model
# catchment 224213: Dargo River, South east coast (Victoria), part of the Mitchell-Thompson Rivers
sf_224213 = sf[['date', '224213']].rename(columns={'224213': 'streamflow'})
rain_224213 = precip[['date', '224213']].rename(columns={'224213': 'rainfall'})
temp_224213 = temp[['date', '224213']].rename(columns={'224213': 'max_temp'})
df_merged = sf_224213.merge(rain_224213, on='date').merge(temp_224213, on='date')
df_merged.to_csv('data_small/224213_data.csv', index=False)

print('done') # simple flag
