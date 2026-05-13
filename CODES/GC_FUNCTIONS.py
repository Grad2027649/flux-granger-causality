#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:03:17 2024

@author: cassandracalderella
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from itertools import cycle, islice
from glob import glob

warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 16
plt.rcParams["figure.autolayout"] = True

sites = pd.read_excel("Fluxnet_Sites.xlsx",sheet_name='Metadata',names=["Station", "Latitude", "Longitude", "Modis_Lat", "Modis_Lon", "Long_Name"])
stations = sites['Station'].values
latitudes = sites['Latitude'].values
longitudes = sites['Longitude'].values
modislats = sites['Modis_Lat'].values
modislons = sites['Modis_Lon'].values
longnames = sites['Long_Name'].values

varnames = ['TA','VPD','SWR','TS','SWC']
predictands = ['NDVI','SIF']
varlabels = ['Air Temperature Anomaly','Vapor Pressure Deficit Anomaly','Incoming Shortwave Radiation Anomaly',
             'Soil Temperature Anomaly','Soil Water Content Anomaly']

units_dict = {'TA':'degC','SW':'W/m**2','LW':'W/m**2','VPD':'hPa','PA':'kPa','P':'kg/m**2/s',
          'WS':'m/s','WD':'degree','RH':'Pa/hPa','USTAR':'m/s','NETRAD':'W/m**2',
          'PPFD':'micromole/s/m**2','CO2':'micromole/mole','TS':'degC','SWC':'mm**2/cm**2',
          'G':'W/m**2','LE':'W/m**2','H':'W/m**2','NEE':'micromole/s/m**2',
          'RECO':'micromole/s/m**2','GPP':'micromole/s/m**2'}

TA_units = units_dict.get('TA')
VPD_units = units_dict.get('VPD')
SWR_units = units_dict.get('SW')
TS_units = units_dict.get('TS')
SWC_units = units_dict.get('SWC')

'''
Stations
~~~~~~~~~~~~~
AMF_US-EML [0]
AMF_US-ICh [1]
AMF_US-Fcr [2]
FLX_FI-Hyy [3]
FLX_FI-Sod [4]
FLX_GL-NuF [5]
FLX_GL-ZaH [6]
FLX_RU-Cok [7]
FLX_SJ-Adv [8]
FLX_US-Prr [9]
AMF_US-BZF [10]
FLX_SE_Deg [11]
FLX_CA-Oas [12]
AMF_CL-SDF [13]
FLX_CN-Ha2 [14]
FLX_CN-HaM [15]
'''

def read_fluxnet_station(stationstring):
    path = sorted(glob("STATION_FILES_QC/"+stationstring+"*.xlsx"))
    #statid = path[0][21:24]
    statid = path[0][24:27]
    if statid == 'EML':
       latitude = latitudes[0]
       longitude = longitudes[0]
       long_name = longnames[0]
       substring = stations[0]
    elif statid == 'ICh':
       latitude = latitudes[1]
       longitude = longitudes[1]
       long_name = longnames[1]
       substring = stations[1]
    elif statid == 'Fcr':
       latitude = latitudes[2]
       longitude = longitudes[2]
       long_name = longnames[2]
       substring = stations[2]
    elif statid == 'Hyy':
       latitude = latitudes[3]
       longitude = longitudes[3]
       long_name = longnames[3]
       substring = stations[3]
    elif statid == 'Sod':
       latitude = latitudes[4]
       longitude = longitudes[4]
       long_name = longnames[4]
       substring = stations[4]
    elif statid == 'NuF':
       latitude = latitudes[5]
       longitude = longitudes[5]
       long_name = longnames[5]
       substring = stations[5]
    elif statid == 'ZaH':
       latitude = latitudes[6]
       longitude = longitudes[6]
       long_name = longnames[6]
       substring = stations[6]
    elif statid == 'Cok':
       latitude = latitudes[7]
       longitude = longitudes[7]
       long_name = longnames[7]
       substring = stations[7]
    elif statid == 'Adv':
       latitude = latitudes[8]
       longitude = longitudes[8]
       long_name = longnames[8]
       substring = stations[8]
    elif statid == 'Prr':
       latitude = latitudes[9]
       longitude = longitudes[9]
       long_name = longnames[9]
       substring = stations[9]
    elif statid == 'BZF':
       latitude = latitudes[10]
       longitude = longitudes[10]
       long_name = longnames[10]
       substring = stations[10]
    elif statid == 'Deg':
       latitude = latitudes[11]
       longitude = longitudes[11]
       long_name = longnames[11]
       substring = stations[11] 
    elif statid == 'Oas':
       latitude = latitudes[12]
       longitude = longitudes[12]
       long_name = longnames[12]
       substring = stations[12]
    elif statid == 'SDF':
       latitude = latitudes[13]
       longitude = longitudes[13]
       long_name = longnames[13]
       substring = stations[13]
    elif statid == 'Ha2':
       latitude = latitudes[14]
       longitude = longitudes[14]
       long_name = longnames[14]
       substring = stations[14]
    elif statid == 'HaM':
       latitude = latitudes[15]
       longitude = longitudes[15]
       long_name = longnames[15]
       substring = stations[15]
    elif statid == 'Fyo':
       latitude = latitudes[16]
       longitude = longitudes[16]
       long_name = longnames[16]
       substring = stations[16]
    elif statid == 'Sor':
       latitude = latitudes[17]
       longitude = longitudes[17]
       long_name = longnames[17]
       substring = stations[17]
    elif statid == 'LP1':
       latitude = latitudes[18]
       longitude = longitudes[18]
       long_name = longnames[18]
       substring = stations[18]
    return path[0],latitude,longitude,long_name,substring

def openfile_sif(filename,sheetname='SIF',begin_str='01-15'):
    df = pd.read_excel(filename,sheet_name=sheetname)
    col = df.pop('Dates')
    df.insert(0,'Dates',col)
    col2 = df.pop(sheetname)
    df.insert(1,sheetname,col2)
    df['Dates'] = pd.to_datetime(df["Dates"]).dt.date.to_numpy(dtype='str')
    df = df[~df.isnull().any(axis=1)]
    df = df.reset_index()
    begin_idx = df.index[df['Dates'].str.find(begin_str) > -1][0]
    df = df.iloc[begin_idx:,:]
    df.drop('index',axis=1,inplace=True)
    df.set_index('Dates',inplace=True)
    return df

def openfile_ndvi(filename,sheetname='NDVI',begin_str='01-01'):
    df = pd.read_excel(filename,sheet_name=sheetname)
    col = df.pop('Dates')
    df.insert(0,'Dates',col)
    col2 = df.pop(sheetname)
    df.insert(1,sheetname,col2)
    df['Dates'] = pd.to_datetime(df["Dates"]).dt.date.to_numpy(dtype='str')
    df = df[~df.isnull().any(axis=1)]
    df = df.reset_index()
    begin_idx = df.index[df['Dates'].str.find(begin_str) > -1][0]
    df = df.iloc[begin_idx:,:]
    df.drop('index',axis=1,inplace=True)
    df.set_index('Dates',inplace=True)
    return df

def openfile_vod(filename,sheetname='VOD',begin_str='02-01'):
    df = pd.read_excel(filename,sheet_name=sheetname)
    col = df.pop('Dates')
    df.insert(0,'Dates',col)
    col2 = df.pop(sheetname)
    df.insert(1,sheetname,col2)
    df['Dates'] = pd.to_datetime(df["Dates"]).dt.date.to_numpy(dtype='str')
    df = df[~df.isnull().any(axis=1)]
    df = df.reset_index()
    begin_idx = df.index[df['Dates'].str.find(begin_str) > -1][0]
    df = df.iloc[begin_idx:,:]
    df.drop('index',axis=1,inplace=True)
    df.set_index('Dates',inplace=True)
    return df

def getdata_predictand(df,causedvar):
    decompose_predictand = seasonal_decompose(df[causedvar],model='additive',period=24,extrapolate_trend='freq')
    predictand_trend = decompose_predictand.trend
    predictand_residual = decompose_predictand.resid
    predictand_mean = np.mean(df[causedvar].values)
    new_predictand_trend = predictand_trend - predictand_mean
    predictand_anomaly = new_predictand_trend + predictand_residual
    return predictand_anomaly

def getdata_predictor(df,causingvar):
    decompose_predictor = seasonal_decompose(df[causingvar],model='additive',period=24,extrapolate_trend='freq')
    predictor_trend = decompose_predictor.trend
    predictor_residual = decompose_predictor.resid
    predictor_mean = np.mean(df[causingvar].values)
    new_predictor_trend = predictor_trend - predictor_mean
    predictor_anomaly = new_predictor_trend + predictor_residual 
    return predictor_anomaly
'''
def get_growing_season_sif(df,var,year,begin_str='04-30',end_str='09-30'):
    orig_df = df.reset_index().rename(columns={0:var})
    orig_df['Dates'] = pd.to_datetime(orig_df["Dates"]).dt.date.to_numpy(dtype='str')
    year_subset = orig_df[orig_df.Dates.str.startswith(year)]
    begin_idx = year_subset.index[year_subset['Dates'].str.find(begin_str) > -1][0]
    end_idx = year_subset.index[year_subset['Dates'].str.find(end_str) > -1][0]
    growing_df = year_subset.loc[begin_idx:end_idx]
    return growing_df

def get_growing_season_sif(df,var,yearlist,yearlength,begin_str='04-30',end_str='09-30'):
    growth = []
    for i in range(yearlength):
        orig_df = df.reset_index().rename(columns={0:var})
        orig_df['Dates'] = pd.to_datetime(orig_df["Dates"]).dt.date.to_numpy(dtype='str')
        year_subset = orig_df[orig_df.Dates.str.startswith(yearlist[i])]
        if year_subset.empty:
           continue
        else:
           begin_idx = year_subset.index[year_subset['Dates'].str.find(begin_str) > -1][0]
           end_idx = year_subset.index[year_subset['Dates'].str.find(end_str) > -1][0]
           growing_df = year_subset.loc[begin_idx:end_idx]
           growth.append(growing_df)
    final_growth = pd.concat(growth)
    return final_growth
'''
def get_growing_season_sif(df,var,yearlist,yearlength,begin_str='04-30',end_str='09-30'):
    growth = []
    for i in range(yearlength):
        orig_df = df.reset_index().rename(columns={0:var})
        orig_df['Dates'] = pd.to_datetime(orig_df["Dates"]).dt.date.to_numpy(dtype='str')
        year_subset = orig_df[orig_df.Dates.str.startswith(yearlist[i])]
        if year_subset.empty:
           continue
        else:
           begin_date = yearlist[i]+'-'+begin_str
           end_date = yearlist[i]+'-'+end_str
           res_begin = year_subset.isin([begin_date]).any().any()
           res_end = year_subset.isin([end_date]).any().any()
           #print(res_begin,res_end)
           if ((res_begin) and (res_end)):
              begin_idx = year_subset.index[year_subset['Dates'].str.find(begin_str) > -1][0]
              end_idx = year_subset.index[year_subset['Dates'].str.find(end_str) > -1][0]
              growing_df = year_subset.loc[begin_idx:end_idx]
              growth.append(growing_df)
           elif ((res_begin) and (not res_end)):
              begin_idx = year_subset.index[year_subset['Dates'].str.find(begin_str) > -1][0]
              growing_df = year_subset.loc[begin_idx:]
              growth.append(growing_df)
           else:
              continue
    final_growth = pd.concat(growth)
    return final_growth

def get_growing_season_ndvi(df,var,year,begin_str='04-15',end_str='09-30'):
    orig_df = df.reset_index().rename(columns={0:var})
    orig_df['Dates'] = pd.to_datetime(orig_df["Dates"]).dt.date.to_numpy(dtype='str')
    year_subset = orig_df[orig_df.Dates.str.startswith(year)]
    mask = (year_subset['Dates'] > year+'-'+begin_str) & (year_subset['Dates'] <= year+'-'+end_str)
    growing_df = year_subset.loc[mask]
    return growing_df

def get_growing_season_vod(df,var,year,begin_str='04-21',end_str='09-21'):
    orig_df = df.reset_index().rename(columns={0:var})
    orig_df['Dates'] = pd.to_datetime(orig_df["Dates"]).dt.date.to_numpy(dtype='str')
    year_subset = orig_df[orig_df.Dates.str.startswith(year)]
    mask = (year_subset['Dates'] > year+'-'+begin_str) & (year_subset['Dates'] <= year+'-'+end_str)
    growing_df = year_subset.loc[mask]
    return growing_df

def check_stationarity(ts):
    result = adfuller(ts)
    print('p-value: {:.3f}'.format(result[1]))
    if result[1] <= 0.05:
        print("Reject Null Hypothesis - Data is Stationary")
        return True
    else:
        print("Fail to Reject Null Hypothesis - Data is Non-Stationary")
        return False
    
def make_data_stationary(data):
    stationary = check_stationarity(data)
    diff_count = 0
    while not stationary:
        diff_count += 1
        data = data.diff().dropna()
        print(f'\nAfter {diff_count} differences:')
        stationary = check_stationarity(data)
    return data

def get_best_lag(df):
    best_aic = np.inf
    best_lag = 0
    for lag in range(1, 20):  # Adjust the range as needed
        model = VAR(df)
        results = model.fit(lag)
        aic = results.aic
        if aic < -100:
            best_lag = lag
            break        
        if aic < best_aic:
            best_aic = aic
            best_lag = lag
            break
    return best_lag

def run_granger_causality_years(df,yearlength,rangeofyears,predictand,varstring):
    columns = [1]
    p_value = []
    yr = []
    final_df = []
    for m in range(yearlength):
        gd = df[df.Dates.str.startswith(rangeofyears[m])]
        dif_predictand = make_data_stationary(gd[predictand])
        dif_predictor = make_data_stationary(gd[varstring])
        dif_df = pd.DataFrame([dif_predictand,dif_predictor]).T
        dif_df_nona = dif_df.dropna()
        best_lag = get_best_lag(dif_df_nona)
        gc = grangercausalitytests(dif_df_nona[[predictand,varstring]],maxlag=best_lag,verbose=False)
        res = gc[1][0]['ssr_ftest'][1]
        p_value.append(res)
        yr.append(rangeofyears[m])
        pvaluedf = pd.DataFrame(p_value,columns=['p-value'])
        pvaluedf['Year'] = yr
        pvaluedf['Lag'] = list(islice(cycle(columns),len(pvaluedf)))
        pvaluedf = pvaluedf.iloc[:,[2,1,0]]
        pvaluedf['Lag'] = pvaluedf['Lag'].astype(str)
        pvaluedf = pvaluedf.set_index('Year')
        lag1 = pvaluedf[pvaluedf.Lag.str.startswith('1')]
        final_df.append(lag1)
    
    final_data = final_df[yearlength-1]
    return final_data
'''
def run_granger_causality_dates(df,datelength,rangeofdates,predictand,varstring):
    columns = [1]
    p_value = []
    yr = []
    final_df = []
    for m in range(datelength):
        try:
          gd = df[df.Dates.str.contains(rangeofdates[m])]
          dif_predictand = make_data_stationary(gd[predictand])
          dif_predictor = make_data_stationary(gd[varstring])
          dif_df = pd.DataFrame([dif_predictand,dif_predictor]).T
          dif_df_nona = dif_df.dropna()
          best_lag = get_best_lag(dif_df_nona)
          gc = grangercausalitytests(dif_df_nona[[predictand,varstring]],maxlag=best_lag,verbose=False)
          res = gc[1][0]['ssr_ftest'][1]
          p_value.append(res)
          yr.append(rangeofdates[m])
          pvaluedf = pd.DataFrame(p_value,columns=['p-value'])
          pvaluedf['Date'] = yr
          pvaluedf['Lag'] = list(islice(cycle(columns),len(pvaluedf)))
          pvaluedf = pvaluedf.iloc[:,[2,1,0]]
          pvaluedf['Lag'] = pvaluedf['Lag'].astype(str)
          pvaluedf = pvaluedf.set_index('Date')
          lag1 = pvaluedf[pvaluedf.Lag.str.startswith('1')]
          final_df.append(lag1)
        except:
          continue
    #final_data = final_df[datelength-1]
    final_data = final_df[datelength-2]
    return final_data
'''

def run_granger_causality_dates(df,datelength,rangeofdates,predictand,varstring):
    columns = [1]
    p_value = []
    yr = []
    final_df = []
    for m in range(datelength):
        print(m,rangeofdates[m])
        gd = df[df.Dates.str.contains(rangeofdates[m])]
        gc = grangercausalitytests(gd[[predictand,varstring]],maxlag=1,verbose=False)
        res = gc[1][0]['ssr_ftest'][1]
        p_value.append(res)
        yr.append(rangeofdates[m])
        pvaluedf = pd.DataFrame(p_value,columns=[varstring+'_'+'p-value'])
        pvaluedf['Date'] = yr
        pvaluedf['Lag'] = list(islice(cycle(columns),len(pvaluedf)))
        pvaluedf = pvaluedf.iloc[:,[2,1,0]]
        pvaluedf['Lag'] = pvaluedf['Lag'].astype(str)
        pvaluedf = pvaluedf.set_index('Date')
        lag1 = pvaluedf[pvaluedf.Lag.str.startswith('1')]
        final_df.append(lag1)
    
    final_data = final_df[datelength-1]
    return final_data

def run_granger_causality_dates_ndvi(df,datelength,rangeofdates,predictand,varstring):
    columns = [1]
    p_value = []
    yr = []
    final_df = []
    for m in range(datelength):
          gd = df[df.Dates.str.contains(rangeofdates[m])]
          print(rangeofdates[m])
          print(predictand)
          dif_predictand = make_data_stationary(gd[predictand])
          print(varstring)
          dif_predictor = make_data_stationary(gd[varstring])
          dif_df = pd.DataFrame([dif_predictand,dif_predictor]).T
          dif_df_nona = dif_df.dropna()
          best_lag = get_best_lag(dif_df_nona)
          gc = grangercausalitytests(dif_df_nona[[predictand,varstring]],maxlag=best_lag,verbose=False)
          res = gc[1][0]['ssr_ftest'][1]
          p_value.append(res)
          yr.append(rangeofdates[m])
          pvaluedf = pd.DataFrame(p_value,columns=['p-value'])
          pvaluedf['Date'] = yr
          pvaluedf['Lag'] = list(islice(cycle(columns),len(pvaluedf)))
          pvaluedf = pvaluedf.iloc[:,[2,1,0]]
          pvaluedf['Lag'] = pvaluedf['Lag'].astype(str)
          pvaluedf = pvaluedf.set_index('Date')
          lag1 = pvaluedf[pvaluedf.Lag.str.startswith('1')]
          final_df.append(lag1)
          
    final_data = final_df[datelength-1]
    return final_data
    
def plot_growing_dates(testlag1,testlag2,testlag3,testlag4,testlag5,siflag1,siflag2,siflag3,siflag4,siflag5,
                        vodlag1,vodlag2,vodlag3,vodlag4,vodlag5, 
                        varlabel,name,lat,lon,statid,predictand,threshold=0.05):
    fig,ax = plt.subplots(nrows=5,ncols=1,figsize=(25,20))
    line1, = ax[0].plot(siflag1['p-value'],'--go',label=predictand[1]+' Anomaly')
    ax_01 = ax[0].twiny()
    line2, = ax_01.plot(testlag1['p-value'],'--bo',label=predictand[0]+' Anomaly')
    line3, = ax_01.plot(vodlag1['p-value'],'--mo',label=predictand[2]+' Anomaly')
    ax_01.set_axis_off()
    ax[1].plot(siflag2['p-value'],'--go')
    ax_11 = ax[1].twiny()
    ax_11.plot(testlag2['p-value'],'--bo')
    ax_11.plot(vodlag2['p-value'],'--mo')
    ax_11.set_axis_off()
    ax[2].plot(siflag3['p-value'],'--go')
    ax_21 = ax[2].twiny()
    ax_21.plot(testlag3['p-value'],'--bo')
    ax_21.plot(vodlag3['p-value'],'--mo')
    ax_21.set_axis_off()
    ax[3].plot(siflag4['p-value'],'--go')
    ax_31 = ax[3].twiny()
    ax_31.plot(testlag4['p-value'],'--bo')
    ax_31.plot(vodlag4['p-value'],'--mo')
    ax_31.set_axis_off()
    ax[4].plot(siflag5['p-value'],'--go')
    ax_41 = ax[4].twiny()
    ax_41.plot(testlag5['p-value'],'--bo')
    ax_41.plot(vodlag5['p-value'],'--mo')
    ax_41.set_axis_off()
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    ax[3].set_yscale('log')
    ax[4].set_yscale('log')
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()
    ax[4].grid()
    #ax[0].legend([predictand[0]+' Anomaly',predictand[1]+' Anomaly'],bbox_to_anchor=(1.05, 1.0),fontsize=20,loc='upper left')
    ax[0].axhline(y=0.05, color='r', linestyle='-')
    ax[1].axhline(y=0.05, color='r', linestyle='-')
    ax[2].axhline(y=0.05, color='r', linestyle='-')
    ax[3].axhline(y=0.05, color='r', linestyle='-')
    ax[4].axhline(y=0.05, color='r', linestyle='-')
    trans0 = transforms.blended_transform_factory(ax[0].get_yticklabels()[0].get_transform(), ax[0].transData)
    trans1 = transforms.blended_transform_factory(ax[1].get_yticklabels()[0].get_transform(), ax[1].transData)
    trans2 = transforms.blended_transform_factory(ax[2].get_yticklabels()[0].get_transform(), ax[2].transData)
    trans3 = transforms.blended_transform_factory(ax[3].get_yticklabels()[0].get_transform(), ax[3].transData)
    trans4 = transforms.blended_transform_factory(ax[4].get_yticklabels()[0].get_transform(), ax[4].transData)
    ax[0].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans0, 
        ha="right", va="center")
    ax[1].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans1, 
        ha="right", va="center")
    ax[2].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans2, 
        ha="right", va="center")
    ax[3].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans3, 
        ha="right", va="center")
    ax[4].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans4, 
        ha="right", va="center")
    ax[0].set_title(varlabel[0]+' (%s)' %(TA_units))
    ax[1].set_title(varlabel[1]+' (%s)' %(VPD_units))
    ax[2].set_title(varlabel[2]+' (%s)' %(SWR_units))
    ax[3].set_title(varlabel[3]+' (%s)' %(TS_units))
    ax[4].set_title(varlabel[4]+' (%s)' %(SWC_units))
    fig.add_subplot(111, frameon=False)
    plt.figlegend(handles=[line1,line2,line3])
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Date")
    plt.ylabel("p-value",labelpad=10)
    plt.tight_layout()
    fig.suptitle('Significance of Fluxnet/Ameriflux Climate Drivers at %s\n$%s^\circ$N, $%s^\circ$E' %(name,lat,lon))
    fig.savefig('%s_GC_Growing_Dates_V2.png' %(statid),dpi=200,bbox_inches='tight')
    #fig.savefig('%s_GC_Growing_Dates_Flux.png' %(statid),dpi=200,bbox_inches='tight')
    
def plot_growing_dates_V2(testlag1,testlag2,testlag3,testlag4,siflag1,siflag2,siflag3,siflag4,
                        vodlag1,vodlag2,vodlag3,vodlag4, 
                        varlabel,name,lat,lon,statid,predictand,threshold=0.05):
    fig,ax = plt.subplots(nrows=4,ncols=1,figsize=(25,20))
    line1, = ax[0].plot(siflag1['TA_p-value'],'--go',label=predictand[1]+' Anomaly')
    ax_01 = ax[0].twiny()
    line2, = ax_01.plot(testlag1['TA_p-value'],'--bo',label=predictand[0]+' Anomaly')
    line3, = ax_01.plot(vodlag1['TA_p-value'],'--mo',label=predictand[2]+' Anomaly')
    ax_01.set_axis_off()
    ax[1].plot(siflag2['VPD_p-value'],'--go')
    ax_11 = ax[1].twiny()
    ax_11.plot(testlag2['VPD_p-value'],'--bo')
    ax_11.plot(vodlag2['VPD_p-value'],'--mo')
    ax_11.set_axis_off()
    ax[2].plot(siflag3['SWR_p-value'],'--go')
    ax_21 = ax[2].twiny()
    ax_21.plot(testlag3['SWR_p-value'],'--bo')
    ax_21.plot(vodlag3['SWR_p-value'],'--mo')
    ax_21.set_axis_off()
    ax[3].plot(siflag4['TS_p-value'],'--go')
    ax_31 = ax[3].twiny()
    ax_31.plot(testlag4['TS_p-value'],'--bo')
    ax_31.plot(vodlag4['TS_p-value'],'--mo')
    ax_31.set_axis_off()
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    ax[3].set_yscale('log')
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()
    #ax[0].legend([predictand[0]+' Anomaly',predictand[1]+' Anomaly'],bbox_to_anchor=(1.05, 1.0),fontsize=20,loc='upper left')
    ax[0].axhline(y=0.05, color='r', linestyle='-')
    ax[1].axhline(y=0.05, color='r', linestyle='-')
    ax[2].axhline(y=0.05, color='r', linestyle='-')
    ax[3].axhline(y=0.05, color='r', linestyle='-')
    trans0 = transforms.blended_transform_factory(ax[0].get_yticklabels()[0].get_transform(), ax[0].transData)
    trans1 = transforms.blended_transform_factory(ax[1].get_yticklabels()[0].get_transform(), ax[1].transData)
    trans2 = transforms.blended_transform_factory(ax[2].get_yticklabels()[0].get_transform(), ax[2].transData)
    trans3 = transforms.blended_transform_factory(ax[3].get_yticklabels()[0].get_transform(), ax[3].transData)
    ax[0].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans0, 
        ha="right", va="center")
    ax[1].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans1, 
        ha="right", va="center")
    ax[2].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans2, 
        ha="right", va="center")
    ax[3].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans3, 
        ha="right", va="center")
    ax[0].set_title(varlabel[0]+' (%s)' %(TA_units))
    ax[1].set_title(varlabel[1]+' (%s)' %(VPD_units))
    ax[2].set_title(varlabel[2]+' (%s)' %(SWR_units))
    ax[3].set_title(varlabel[3]+' (%s)' %(TS_units))
    fig.add_subplot(111, frameon=False)
    plt.figlegend(handles=[line1,line2,line3])
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Date")
    plt.ylabel("p-value",labelpad=10)
    plt.tight_layout()
    fig.suptitle('Significance of Fluxnet/Ameriflux Climate Drivers at %s\n$%s^\circ$N, $%s^\circ$E' %(name,lat,lon))
    fig.savefig('%s_GC_Growing_Dates_V2.png' %(statid),dpi=200,bbox_inches='tight')
    #fig.savefig('%s_GC_Growing_Dates_Flux.png' %(statid),dpi=200,bbox_inches='tight')
    
'''
def plot_growing_dates(testlag1,testlag2,testlag3,testlag4,testlag5,siflag1,siflag2,siflag3,siflag4,siflag5,
                     varlabel,name,lat,lon,statid,predictand,threshold=0.05):
    fig,ax = plt.subplots(nrows=5,ncols=2,figsize=(25,20))
    ax[0,0].plot(testlag1['p-value'],'--bo')
    ax[0,1].plot(siflag1['p-value'],'--go')
    ax[1,0].plot(testlag2['p-value'],'--bo')
    ax[1,1].plot(siflag2['p-value'],'--go')
    ax[2,0].plot(testlag3['p-value'],'--bo')
    ax[2,1].plot(siflag3['p-value'],'--go')
    ax[3,0].plot(testlag4['p-value'],'--bo')
    ax[3,1].plot(siflag4['p-value'],'--go')
    ax[4,0].plot(testlag5['p-value'],'--bo')
    ax[4,1].plot(siflag5['p-value'],'--go')
    ax[0,0].set_yscale('log')
    ax[0,1].set_yscale('log')
    ax[1,0].set_yscale('log')
    ax[1,1].set_yscale('log')
    ax[2,0].set_yscale('log')
    ax[2,1].set_yscale('log')
    ax[3,0].set_yscale('log')
    ax[3,1].set_yscale('log')
    ax[4,0].set_yscale('log')
    ax[4,1].set_yscale('log')
    ax[0,0].grid()
    ax[0,1].grid()
    ax[1,0].grid()
    ax[1,1].grid()
    ax[2,0].grid()
    ax[2,1].grid()
    ax[3,0].grid()
    ax[3,1].grid()
    ax[4,0].grid()
    ax[4,1].grid()
    fig.legend([predictand[0]+' Anomaly',predictand[1]+' Anomaly'],bbox_to_anchor=(1.05, 1.0),fontsize=20,loc='upper left')
    ax[0,0].axhline(y=0.05, color='r', linestyle='-')
    ax[0,1].axhline(y=0.05, color='r', linestyle='-')
    ax[1,0].axhline(y=0.05, color='r', linestyle='-')
    ax[1,1].axhline(y=0.05, color='r', linestyle='-')
    ax[2,0].axhline(y=0.05, color='r', linestyle='-')
    ax[2,1].axhline(y=0.05, color='r', linestyle='-')
    ax[3,0].axhline(y=0.05, color='r', linestyle='-')
    ax[3,1].axhline(y=0.05, color='r', linestyle='-')
    ax[4,0].axhline(y=0.05, color='r', linestyle='-')
    ax[4,1].axhline(y=0.05, color='r', linestyle='-')
    trans0 = transforms.blended_transform_factory(ax[0,0].get_yticklabels()[0].get_transform(), ax[0,0].transData)
    trans1 = transforms.blended_transform_factory(ax[0,1].get_yticklabels()[0].get_transform(), ax[0,1].transData)
    trans2 = transforms.blended_transform_factory(ax[1,0].get_yticklabels()[0].get_transform(), ax[1,0].transData)
    trans3 = transforms.blended_transform_factory(ax[1,1].get_yticklabels()[0].get_transform(), ax[1,1].transData)
    trans4 = transforms.blended_transform_factory(ax[2,0].get_yticklabels()[0].get_transform(), ax[2,0].transData)
    trans5 = transforms.blended_transform_factory(ax[2,1].get_yticklabels()[0].get_transform(), ax[2,1].transData)
    trans6 = transforms.blended_transform_factory(ax[3,0].get_yticklabels()[0].get_transform(), ax[3,0].transData)
    trans7 = transforms.blended_transform_factory(ax[3,1].get_yticklabels()[0].get_transform(), ax[3,1].transData)
    trans8 = transforms.blended_transform_factory(ax[4,0].get_yticklabels()[0].get_transform(), ax[4,0].transData)
    trans9 = transforms.blended_transform_factory(ax[4,1].get_yticklabels()[0].get_transform(), ax[4,1].transData)
    ax[0,0].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans0, 
        ha="right", va="center")
    ax[0,1].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans1, 
        ha="right", va="center")
    ax[1,0].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans2, 
        ha="right", va="center")
    ax[1,1].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans3, 
        ha="right", va="center")
    ax[2,0].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans4, 
        ha="right", va="center")
    ax[2,1].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans5, 
        ha="right", va="center")
    ax[3,0].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans6, 
        ha="right", va="center")
    ax[3,1].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans7, 
        ha="right", va="center")
    ax[4,0].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans8, 
        ha="right", va="center")
    ax[4,1].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans9, 
        ha="right", va="center")
    ax[0,0].set_title(varlabel[0])
    ax[0,1].set_title(varlabel[0])
    ax[1,0].set_title(varlabel[1])
    ax[1,1].set_title(varlabel[1])
    ax[2,0].set_title(varlabel[2])
    ax[2,1].set_title(varlabel[2])
    ax[3,0].set_title(varlabel[3])
    ax[3,1].set_title(varlabel[3])
    ax[4,0].set_title(varlabel[4])
    ax[4,1].set_title(varlabel[4])
    ax[4,0].set_xlabel("Month")
    ax[4,1].set_xlabel("Date")
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("p-value",labelpad=10)
    plt.tight_layout()
    fig.suptitle('Significance of Climate Drivers at %s\n$%s^\circ$N, $%s^\circ$E' %(name,lat,lon))
    fig.savefig('%s_GC_Growing_Dates.png' %(statid),dpi=200,bbox_inches='tight')
    
def plot_growing_dates_V2(testlag1,testlag2,testlag3,testlag4,siflag1,siflag2,siflag3,siflag4,
                     varlabel,name,lat,lon,statid,predictand,threshold=0.05):
    fig,ax = plt.subplots(nrows=4,ncols=2,figsize=(25,20))
    ax[0,0].plot(testlag1['p-value'],'--bo')
    ax[0,1].plot(siflag1['p-value'],'--go')
    ax[1,0].plot(testlag2['p-value'],'--bo')
    ax[1,1].plot(siflag2['p-value'],'--go')
    ax[2,0].plot(testlag3['p-value'],'--bo')
    ax[2,1].plot(siflag3['p-value'],'--go')
    ax[3,0].plot(testlag4['p-value'],'--bo')
    ax[3,1].plot(siflag4['p-value'],'--go')
    ax[0,0].set_yscale('log')
    ax[0,1].set_yscale('log')
    ax[1,0].set_yscale('log')
    ax[1,1].set_yscale('log')
    ax[2,0].set_yscale('log')
    ax[2,1].set_yscale('log')
    ax[3,0].set_yscale('log')
    ax[3,1].set_yscale('log')
    ax[0,0].grid()
    ax[0,1].grid()
    ax[1,0].grid()
    ax[1,1].grid()
    ax[2,0].grid()
    ax[2,1].grid()
    ax[3,0].grid()
    ax[3,1].grid()
    fig.legend([predictand[0]+' Anomaly',predictand[1]+' Anomaly'],bbox_to_anchor=(1.05, 1.0),fontsize=20,loc='upper left')
    ax[0,0].axhline(y=0.05, color='r', linestyle='-')
    ax[0,1].axhline(y=0.05, color='r', linestyle='-')
    ax[1,0].axhline(y=0.05, color='r', linestyle='-')
    ax[1,1].axhline(y=0.05, color='r', linestyle='-')
    ax[2,0].axhline(y=0.05, color='r', linestyle='-')
    ax[2,1].axhline(y=0.05, color='r', linestyle='-')
    ax[3,0].axhline(y=0.05, color='r', linestyle='-')
    ax[3,1].axhline(y=0.05, color='r', linestyle='-')
    trans0 = transforms.blended_transform_factory(ax[0,0].get_yticklabels()[0].get_transform(), ax[0,0].transData)
    trans1 = transforms.blended_transform_factory(ax[0,1].get_yticklabels()[0].get_transform(), ax[0,1].transData)
    trans2 = transforms.blended_transform_factory(ax[1,0].get_yticklabels()[0].get_transform(), ax[1,0].transData)
    trans3 = transforms.blended_transform_factory(ax[1,1].get_yticklabels()[0].get_transform(), ax[1,1].transData)
    trans4 = transforms.blended_transform_factory(ax[2,0].get_yticklabels()[0].get_transform(), ax[2,0].transData)
    trans5 = transforms.blended_transform_factory(ax[2,1].get_yticklabels()[0].get_transform(), ax[2,1].transData)
    trans6 = transforms.blended_transform_factory(ax[3,0].get_yticklabels()[0].get_transform(), ax[3,0].transData)
    trans7 = transforms.blended_transform_factory(ax[3,1].get_yticklabels()[0].get_transform(), ax[3,1].transData)
    ax[0,0].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans0, 
        ha="right", va="center")
    ax[0,1].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans1, 
        ha="right", va="center")
    ax[1,0].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans2, 
        ha="right", va="center")
    ax[1,1].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans3, 
        ha="right", va="center")
    ax[2,0].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans4, 
        ha="right", va="center")
    ax[2,1].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans5, 
        ha="right", va="center")
    ax[3,0].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans6, 
        ha="right", va="center")
    ax[3,1].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans7, 
        ha="right", va="center")
    ax[0,0].set_title(varlabel[0])
    ax[0,1].set_title(varlabel[0])
    ax[1,0].set_title(varlabel[1])
    ax[1,1].set_title(varlabel[1])
    ax[2,0].set_title(varlabel[2])
    ax[2,1].set_title(varlabel[2])
    ax[3,0].set_title(varlabel[3])
    ax[3,1].set_title(varlabel[3])
    ax[3,0].set_xlabel("Month")
    ax[3,1].set_xlabel("Date")
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("p-value",labelpad=10)
    plt.tight_layout()
    fig.suptitle('Significance of Climate Drivers at %s\n$%s^\circ$N, $%s^\circ$E' %(name,lat,lon))
    fig.savefig('%s_GC_Growing_Dates.png' %(statid),dpi=200,bbox_inches='tight')
''' 