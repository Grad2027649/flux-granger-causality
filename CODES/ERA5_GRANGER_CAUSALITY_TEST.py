#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 17:00:46 2025

@author: cassandracalderella
"""

import pandas as pd
import GC_FUNCTIONS as GC
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import cycle, islice
from datetime import datetime

pd.options.mode.chained_assignment = None
sites = pd.read_excel("Fluxnet_Sites.xlsx",sheet_name='Metadata',names=["Station", "Latitude", "Longitude", "Modis_Lat", "Modis_Lon", "Long_Name"])
stations = sites['Station'].values

today_date = datetime.today().strftime('%Y%m%d')

years = ['1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006',
         '2007','2008','2009','2010','2011','2012','2013','2014','2015',
         '2016','2017','2018','2019','2020','2021']

varnames = ['TA','VPD','ISR','TS','SWC']
predictands = ['NDVI','SIF','VOD']
varlabels = ['Air Temperature Anomaly','Vapor Pressure Deficit Anomaly','Incoming Shortwave Radiation Anomaly',
             'Soil Temperature Anomaly','Soil Water Content Anomaly']

datestrings = ['04-15','04-30','05-15','05-31','06-15','06-30',
              '07-15','07-31','08-15','08-31','09-15','09-30']
daterange = len(datestrings)

#ndvimonthstrings = ['04','05','06','07','08','09']
ndvimonthstrings = ['04-06','04-07','04-22','04-23','05-08','05-09',
                    '05-24','05-25','06-09','06-10','06-25','06-26',
                    '07-11','07-12','07-27','07-28','08-12','08-13',
                    '08-28','08-29','09-13','09-14','09-29','09-30']
ndvimonthlength = len(ndvimonthstrings)

#vodmonthstrings = ['05','06','07','08','09']
vodmonthstrings = ['04-01','04-11','04-21','05-01','05-11','05-21',
                   '06-01','06-11','06-21','07-01','07-11','07-21',
                   '08-01','08-11','08-21','09-01','09-11','09-21']
vodmonthlength = len(vodmonthstrings)

search_strings = ['-01-','-02-','-03-','-10-','-11-','-12-']
pattern = '|'.join(search_strings)

def custom_grouper(x):
    month_start = x.replace(day=1)
    return pd.Timestamp(month_start) + pd.Timedelta(days=int((x.day - 2) / 10) * 10)

def get_growing_season(df,pattern,var):
    df = df.reset_index()
    df['Dates'] = pd.to_datetime(df["Dates"],format='%Y-%m-%d').dt.date.to_numpy(dtype='str')
    grow_df = df[df.Dates.str.contains(pattern) == False]
    grow_df = grow_df.rename(columns={0:var})
    return grow_df

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

def run_granger_causality_dates(df,datelength,rangeofdates,predictand,varstring):
    columns = [1]
    p_value = []
    yr = []
    final_df = []
    for m in range(datelength):
        gd = df[df.Dates.str.contains(rangeofdates[m])]
        count = df.Dates.str.contains(rangeofdates[m]).sum()
        if count <= 4:
           continue
        else:
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
    
    final_data = final_df[len(final_df)-1]
    return final_data

stat = stations[18]

filename,lat,lon,longname,substring = GC.read_fluxnet_station(stat)

sif_df = GC.openfile_sif(filename)
sif_df_years = sif_df.reset_index()
sif_df_years['Dates'] = pd.to_datetime(sif_df_years["Dates"])
years = list({i for i in sif_df_years['Dates'].dt.year})
years.sort()
years = list(map(str,years))
yearlen = len(years)

ndvi_df = GC.openfile_ndvi(filename)

vod_df = GC.openfile_vod(filename)
vod_df = vod_df.replace('--',np.nan)
vod_df = vod_df.dropna()


era5_df = pd.read_excel('%s_ERA5.xlsx' %(substring))
col = era5_df.pop('Dates')
era5_df.insert(0,'Dates',col)

era5_sif = era5_df.groupby(pd.Grouper(key='Dates',freq='SM')).mean()
era5_sif = era5_sif[1:]

era5_ndvi = era5_df.groupby(pd.Grouper(key='Dates',freq='16D')).mean()

era5_vod = era5_df
era5_vod['Dates'] = era5_vod['Dates'].apply(custom_grouper)
era5_vod = era5_vod.groupby('Dates').mean()

sif_anomaly = getdata_predictand(sif_df,'SIF')
ta_anomaly = getdata_predictor(era5_sif,'TA')
vpd_anomaly = getdata_predictor(era5_sif,'VPD')
swr_anomaly = getdata_predictor(era5_sif,'ISR')
ts_anomaly = getdata_predictor(era5_sif,'TS')
swc_anomaly = getdata_predictor(era5_sif,'SWC')

sif_growing = get_growing_season(sif_anomaly,pattern,predictands[1])
sif_ta_growing = get_growing_season(ta_anomaly,pattern,varnames[0])
sif_vpd_growing = get_growing_season(vpd_anomaly,pattern,varnames[1])
sif_swr_growing = get_growing_season(swr_anomaly,pattern,varnames[2])
sif_ts_growing = get_growing_season(ts_anomaly,pattern,varnames[3])
sif_swc_growing = get_growing_season(swc_anomaly,pattern,varnames[4])

sif_growing_df = pd.DataFrame(sif_growing)
sif_growing_df[varnames[0]] = sif_ta_growing[varnames[0]]
sif_growing_df[varnames[1]] = sif_vpd_growing[varnames[1]]
sif_growing_df[varnames[2]] = sif_swr_growing[varnames[2]]
sif_growing_df[varnames[3]] = sif_ts_growing[varnames[3]]
sif_growing_df[varnames[4]] = sif_swc_growing[varnames[4]]

ndvi_anomaly = getdata_predictand(ndvi_df,'NDVI')
ta_anomaly_ndvi = getdata_predictor(era5_ndvi,'TA')
vpd_anomaly_ndvi = getdata_predictor(era5_ndvi,'VPD')
swr_anomaly_ndvi = getdata_predictor(era5_ndvi,'ISR')
ts_anomaly_ndvi = getdata_predictor(era5_ndvi,'TS')
swc_anomaly_ndvi = getdata_predictor(era5_ndvi,'SWC')

ndvi_growing = get_growing_season(ndvi_anomaly,pattern,predictands[0])
ndvi_ta_growing = get_growing_season(ta_anomaly_ndvi,pattern,varnames[0])
ndvi_vpd_growing = get_growing_season(vpd_anomaly_ndvi,pattern,varnames[1])
ndvi_swr_growing = get_growing_season(swr_anomaly_ndvi,pattern,varnames[2])
ndvi_ts_growing = get_growing_season(ts_anomaly_ndvi,pattern,varnames[3])
ndvi_swc_growing = get_growing_season(swc_anomaly_ndvi,pattern,varnames[4])

ndvi_growing_df = pd.DataFrame(ndvi_growing)
ndvi_growing_df[varnames[0]] = ndvi_ta_growing[varnames[0]]
ndvi_growing_df[varnames[1]] = ndvi_vpd_growing[varnames[1]]
ndvi_growing_df[varnames[2]] = ndvi_swr_growing[varnames[2]]
ndvi_growing_df[varnames[3]] = ndvi_ts_growing[varnames[3]]
ndvi_growing_df[varnames[4]] = ndvi_swc_growing[varnames[4]]

vod_anomaly = getdata_predictand(vod_df,'VOD')
ta_anomaly_vod = getdata_predictor(era5_vod,'TA')
vpd_anomaly_vod = getdata_predictor(era5_vod,'VPD')
swr_anomaly_vod = getdata_predictor(era5_vod,'ISR')
ts_anomaly_vod = getdata_predictor(era5_vod,'TS')
swc_anomaly_vod = getdata_predictor(era5_vod,'SWC')

vod_growing = get_growing_season(vod_anomaly,pattern,predictands[2])
vod_ta_growing = get_growing_season(ta_anomaly_vod,pattern,varnames[0])
vod_vpd_growing = get_growing_season(vpd_anomaly_vod,pattern,varnames[1])
vod_swr_growing = get_growing_season(swr_anomaly_vod,pattern,varnames[2])
vod_ts_growing = get_growing_season(ts_anomaly_vod,pattern,varnames[3])
vod_swc_growing = get_growing_season(swc_anomaly_vod,pattern,varnames[4])

vod_growing_df = pd.DataFrame(vod_growing)
vod_growing_df[varnames[0]] = vod_ta_growing[varnames[0]]
vod_growing_df[varnames[1]] = vod_vpd_growing[varnames[1]]
vod_growing_df[varnames[2]] = vod_swr_growing[varnames[2]]
vod_growing_df[varnames[3]] = vod_ts_growing[varnames[3]]
vod_growing_df[varnames[4]] = vod_swc_growing[varnames[4]]


check_nan_ndvi = ndvi_growing_df.isnull().values.any()
if check_nan_ndvi:
   ndvi_growing_df = ndvi_growing_df.dropna()
else:
   ndvi_growing_df = ndvi_growing_df  
   
check_nan_sif = sif_growing_df.isnull().values.any()
if check_nan_sif:
   sif_growing_df = sif_growing_df.dropna()
else:
   sif_growing_df = sif_growing_df

check_nan_vod = vod_growing_df.isnull().values.any()
if check_nan_vod:
   vod_growing_df = vod_growing_df.dropna()
else:
   vod_growing_df = vod_growing_df

sif_ta_granger = run_granger_causality_dates(sif_growing_df,daterange,datestrings,predictands[1],varnames[0])
sif_vpd_granger = run_granger_causality_dates(sif_growing_df,daterange,datestrings,predictands[1],varnames[1])
sif_swr_granger = run_granger_causality_dates(sif_growing_df,daterange,datestrings,predictands[1],varnames[2])
sif_ts_granger = run_granger_causality_dates(sif_growing_df,daterange,datestrings,predictands[1],varnames[3])
sif_swc_granger = run_granger_causality_dates(sif_growing_df,daterange,datestrings,predictands[1],varnames[4])

ndvi_ta_granger = run_granger_causality_dates(ndvi_growing_df,ndvimonthlength,ndvimonthstrings,predictands[0],varnames[0])
ndvi_vpd_granger = run_granger_causality_dates(ndvi_growing_df,ndvimonthlength,ndvimonthstrings,predictands[0],varnames[1])
ndvi_swr_granger = run_granger_causality_dates(ndvi_growing_df,ndvimonthlength,ndvimonthstrings,predictands[0],varnames[2])
ndvi_ts_granger = run_granger_causality_dates(ndvi_growing_df,ndvimonthlength,ndvimonthstrings,predictands[0],varnames[3])
ndvi_swc_granger = run_granger_causality_dates(ndvi_growing_df,ndvimonthlength,ndvimonthstrings,predictands[0],varnames[4])

vod_ta_granger = run_granger_causality_dates(vod_growing_df,vodmonthlength,vodmonthstrings,predictands[2],varnames[0])
vod_vpd_granger = run_granger_causality_dates(vod_growing_df,vodmonthlength,vodmonthstrings,predictands[2],varnames[1])
vod_swr_granger = run_granger_causality_dates(vod_growing_df,vodmonthlength,vodmonthstrings,predictands[2],varnames[2])
vod_ts_granger = run_granger_causality_dates(vod_growing_df,vodmonthlength,vodmonthstrings,predictands[2],varnames[3])
vod_swc_granger = run_granger_causality_dates(vod_growing_df,vodmonthlength,vodmonthstrings,predictands[2],varnames[4])

sif_granger = pd.concat([sif_ta_granger['TA_p-value'],sif_vpd_granger['VPD_p-value'],sif_swr_granger['ISR_p-value'],
                         sif_ts_granger['TS_p-value'],sif_swc_granger['SWC_p-value']],axis=1)
ndvi_granger = pd.concat([ndvi_ta_granger['TA_p-value'],ndvi_vpd_granger['VPD_p-value'],ndvi_swr_granger['ISR_p-value'],
                         ndvi_ts_granger['TS_p-value'],ndvi_swc_granger['SWC_p-value']],axis=1)
vod_granger = pd.concat([vod_ta_granger['TA_p-value'],vod_vpd_granger['VPD_p-value'],vod_swr_granger['ISR_p-value'],
                         vod_ts_granger['TS_p-value'],vod_swc_granger['SWC_p-value']],axis=1)

with pd.ExcelWriter("%s_Granger_Causality_ERA5_WITH_VOD_%s.xlsx" %(substring,today_date)) as writer:
#with pd.ExcelWriter("%s_Granger_Causality_ERA5_20250505.xlsx" %(substring)) as writer:
     sif_granger.to_excel(writer,sheet_name='SIF')
     ndvi_granger.to_excel(writer,sheet_name='NDVI')
     vod_granger.to_excel(writer,sheet_name='VOD')