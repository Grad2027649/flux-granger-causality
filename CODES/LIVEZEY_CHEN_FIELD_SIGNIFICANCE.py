#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 17:04:47 2025

@author: cassandracalderella
"""

import pandas as pd
import random
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time

start_time = time.time()

plt.rcParams['font.size'] = 12
predictands = ['NDVI','SIF','VOD']
today_date = datetime.today().strftime('%Y%m%d')
'''
path = sorted(glob('*Granger_Causality_20250504.xlsx'))
path_nondvi = sorted(glob('*Granger_Causality_20250504.xlsx'))
path_nondvi.pop(8)
path2 = sorted(glob('*Granger_Causality_VOD_20250504.xlsx'))
'''

path = sorted(glob('*Granger_Causality_ERA5_20250505.xlsx'))
path_nondvi = sorted(glob('*Granger_Causality_ERA5_20250505.xlsx'))
path_nondvi.pop(8)
path2 = sorted(glob('*Granger_Causality_ERA5_WITH_VOD_20250505.xlsx'))


sif_total = []
ndvi_total = []
vod_total = []

def generate_num(threshold):
    int1 = random.uniform(0,1)
    if int1 <= threshold:
       less_than = True
    else:
       less_than = False
    return less_than

def monte_carlo_simulation(N,sample_size,threshold):
    results = []
    for _ in range(N):
        simulation_results = []
        for _ in range(sample_size):
            less = generate_num(threshold)
            simulation_results.append(less)
        results.append(simulation_results)
    return results

def livezey_chen_test(N,sample_size,total_true,threshold,variable):
    results = monte_carlo_simulation(N,sample_size,threshold) 
    results_df = (pd.DataFrame(results)).astype(str)

    counts = results_df.apply(results_df.value_counts,axis=1)
    counts = counts.fillna(0)
    
    max_count = int(np.max(counts['True']))
    min_count = int(np.min(counts['True']))
    
    n_bins = np.arange(min_count,max_count,1)
    p = np.percentile(counts['True'],95)
    
    fig,ax = plt.subplots(figsize=(14,8))
    n, bins, patches = ax.hist(counts['True'],bins=n_bins,color='green',edgecolor='black')
    ax.axvline(x = total_true, color = 'r')
    ax.axvline(x = p, color = 'r',ls='--')
    ax.set_ylabel('Count')
    ax.set_xlabel('Number of Points with Significant Correlations (p<%s)' %(threshold))
    ax.set_title('Number of Points with Significant Correlations for %s (10,000 Trials)' %(variable))
    
    less_count = counts['True'][counts['True'] < total_true].count().sum()
    significance_level = (less_count/N)*100
    return significance_level

for i,f in enumerate(path):
    sif_df = pd.read_excel(f,sheet_name='SIF')
    sif_df = sif_df.set_index('Date')
    sif_count = sif_df[sif_df <= 0.05].count()
    sif_total.append(sif_count)
    
for j,g in enumerate(path_nondvi):
    ndvi_df = pd.read_excel(g,sheet_name='NDVI')
    ndvi_df = ndvi_df.set_index('Date')
    ndvi_count = ndvi_df[ndvi_df <= 0.05].count()
    ndvi_total.append(ndvi_count)
    
for k,h in enumerate(path2):
    vod_df = pd.read_excel(h,sheet_name='VOD')
    vod_df = vod_df.set_index('Date')
    vod_count = vod_df[vod_df <= 0.05].count()
    vod_total.append(vod_count)

sif_total_df = pd.concat(sif_total,axis=1)
sif_total_df = sif_total_df.T
sum_sif = sif_total_df.sum().sum()

ndvi_total_df = pd.concat(ndvi_total,axis=1)
ndvi_total_df = ndvi_total_df.T
sum_ndvi = ndvi_total_df.sum().sum()

vod_total_df = pd.concat(vod_total,axis=1)
vod_total_df = vod_total_df.T
sum_vod = vod_total_df.sum().sum()

#num_stations_sif = 12
#num_stations_ndvi = 11
#num_stations_vod = 6
num_stations_sif = 12
num_stations_ndvi = 11
num_stations_vod = 4
num_sif_timesteps = 12
num_ndvi_timesteps = 24
num_vod_timesteps = 18
sif_sample_size = num_stations_sif*num_sif_timesteps
ndvi_sample_size = num_stations_ndvi*num_ndvi_timesteps
vod_sample_size = num_stations_vod*num_vod_timesteps
one_percent_threshold = 0.01
five_percent_threshold = 0.05
ten_percent_threshold = 0.1

print(f"NDVI Sample Size: {ndvi_sample_size}",f"NDVI Total Number of True Points: {sum_ndvi}")
print(f"SIF Sample Size: {sif_sample_size}",f"SIF Total Number of True Points: {sum_sif}")
print(f"VOD Sample Size: {vod_sample_size}",f"VOD Total Number of True Points: {sum_vod}")

trials = 10000
significance_level_ndvi = livezey_chen_test(trials,ndvi_sample_size,sum_ndvi,five_percent_threshold,predictands[0])
significance_level_sif = livezey_chen_test(trials,sif_sample_size,sum_sif,five_percent_threshold,predictands[1])
significance_level_vod = livezey_chen_test(trials,vod_sample_size,sum_vod,five_percent_threshold,predictands[2])
print(f"NDVI Significance Level: {significance_level_ndvi}")
print(f"SIF Significance Level: {significance_level_sif}")
print(f"VOD Significance Level: {significance_level_vod}")

end_time = time.time()
elapsed_time = end_time-start_time
elapsed_time_min = elapsed_time/60.0
if elapsed_time < 60.0:
   print(f"Script execution time: {elapsed_time:.2f} second(s)")
else:
   print(f"Script execution time: {elapsed_time_min:.2f} minute(s)")