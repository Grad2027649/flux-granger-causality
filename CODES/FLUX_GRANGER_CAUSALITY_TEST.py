#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 10:11:25 2025

@author: cassandracalderella
"""

import pandas as pd
import GC_FUNCTIONS as GC
import numpy as np
import READ_DATA as RD
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import cycle, islice
from datetime import datetime

pd.options.mode.chained_assignment = None
sites = pd.read_excel("Fluxnet_Sites.xlsx",sheet_name='Metadata',names=["Station", "Latitude", "Longitude", "Modis_Lat", "Modis_Lon", "Long_Name"])
stations = sites['Station'].values

today_date = datetime.today().strftime('%Y%m%d')

varnames = ['TA','VPD','SWR','TS','SWC']
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
#ndvimonthstrings = ['04-07','04-23','05-09','05-25','06-10','06-26',
#                    '07-12','07-28','08-13','08-29','09-14','09-30']
ndvimonthlength = len(ndvimonthstrings)

#vodmonthstrings = ['05','06','07','08','09']
vodmonthstrings = ['04-01','04-11','04-21','05-01','05-11','05-21',
                   '06-01','06-11','06-21','07-01','07-11','07-21',
                   '08-01','08-11','08-21','09-01','09-11','09-21']
vodmonthlength = len(vodmonthstrings)

years = ['1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006',
         '2007','2008','2009','2010','2011','2012','2013','2014','2015',
         '2016','2017','2018','2019','2020','2021','2022']

today_date = datetime.today().strftime('%Y%m%d')

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
FLX_RU-Fyo [16]
FLX_DK-Sor [17]
AMF_CA-LP1 [18]
'''

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

filename,pixellat,pixellon,longname,substring,station_lat,station_lon = RD.read_fluxnet_station(stat)
GCfilename,GClat,GClon,GClongname,GCsubstring = GC.read_fluxnet_station(stat)
search_strings = ['-01-','-02-','-03-','-10-','-11-','-12-']
pattern = '|'.join(search_strings)

if (substring == 'FLX_SE_Deg'):
    start,end = RD.getyears2(filename)
    startindex = years.index(start)
    endindex = years.index(end)
else:
    start,end = RD.getyears(filename)
    startindex = years.index(start)
    if (substring == 'AMF_CA-LP1'):
       endindex = years.index(years[-1])
    else:
       endindex = years.index(end)
       
 
if ((substring == 'FLX_FI-Hyy') or (substring == 'FLX_FI-Sod') or (substring == 'FLX_GL-ZaH') or (substring == 'FLX_SE_Deg') or (substring == 'FLX_CA-Oas')):
    yearrange = years[7:endindex+1]
    yearlen = len(yearrange)
else:
   yearrange = years[startindex:endindex+1]
   yearlen = len(yearrange)
   
sif_df = pd.read_excel(GCfilename,sheet_name='SIF')
ndvi_df = pd.read_excel(GCfilename,sheet_name='NDVI')
vod_df = pd.read_excel(GCfilename,sheet_name='VOD')
vod_df = vod_df.replace('--',np.nan)

if ((substring == 'FLX_GL-ZaH') or (substring == 'FLX_SE_Deg')):
    sif_df = sif_df.drop(columns=['SWC_F_MDS_1','SWC_F_MDS_1_QC'])
    ndvi_df = ndvi_df.drop(columns=['SWC_F_MDS_1','SWC_F_MDS_1_QC'])
    vod_df = vod_df.drop(columns=['SWC_F_MDS_1','SWC_F_MDS_1_QC'])
    sif_df = sif_df[(sif_df.TA_F_MDS_QC >= 0.75) & (sif_df.VPD_F_MDS_QC >= 0.75) & (sif_df.SW_IN_F_MDS_QC >= 0.75)
                    & (sif_df.TS_F_MDS_1_QC >= 0.75)]

    ndvi_df = ndvi_df[(ndvi_df.TA_F_MDS_QC >= 0.75) & (ndvi_df.VPD_F_MDS_QC >= 0.75) & (ndvi_df.SW_IN_F_MDS_QC >= 0.75)
                    & (ndvi_df.TS_F_MDS_1_QC >= 0.75)]
    
    vod_df = vod_df[(vod_df.TA_F_MDS_QC >= 0.75) & (vod_df.VPD_F_MDS_QC >= 0.75) & (vod_df.SW_IN_F_MDS_QC >= 0.75)
                    & (vod_df.TS_F_MDS_1_QC >= 0.75)]
    vod_df = vod_df.dropna()

    sif_anomaly = getdata_predictand(sif_df,'SIF')
    ta_anomaly = getdata_predictor(sif_df,'TA_F_MDS')
    vpd_anomaly = getdata_predictor(sif_df,'VPD_F_MDS')
    swr_anomaly = getdata_predictor(sif_df,'SW_IN_F_MDS')
    ts_anomaly = getdata_predictor(sif_df,'TS_F_MDS_1')
    sif_anom_df = pd.DataFrame(sif_anomaly,columns=[predictands[1]])
    sif_anom_df[varnames[0]] = ta_anomaly
    sif_anom_df[varnames[1]] = vpd_anomaly
    sif_anom_df[varnames[2]] = swr_anomaly
    sif_anom_df[varnames[3]] = ts_anomaly
    sif_anom_df['Dates'] = sif_df['Dates']

    sif_growing_df = sif_anom_df[sif_anom_df.Dates.str.contains(pattern) == False]

    ndvi_anomaly = getdata_predictand(ndvi_df,'NDVI')
    ta_anomaly_ndvi = getdata_predictor(ndvi_df,'TA_F_MDS')
    vpd_anomaly_ndvi = getdata_predictor(ndvi_df,'VPD_F_MDS')
    swr_anomaly_ndvi = getdata_predictor(ndvi_df,'SW_IN_F_MDS')
    ts_anomaly_ndvi = getdata_predictor(ndvi_df,'TS_F_MDS_1')
    ndvi_anom_df = pd.DataFrame(ndvi_anomaly,columns=[predictands[0]])
    ndvi_anom_df[varnames[0]] = ta_anomaly_ndvi
    ndvi_anom_df[varnames[1]] = vpd_anomaly_ndvi
    ndvi_anom_df[varnames[2]] = swr_anomaly_ndvi
    ndvi_anom_df[varnames[3]] = ts_anomaly_ndvi
    ndvi_anom_df['Dates'] = ndvi_df['Dates']

    ndvi_growing_df = ndvi_anom_df[ndvi_anom_df.Dates.str.contains(pattern) == False]
    
    vod_anomaly = getdata_predictand(vod_df,'VOD')
    ta_anomaly_vod = getdata_predictor(vod_df,'TA_F_MDS')
    vpd_anomaly_vod = getdata_predictor(vod_df,'VPD_F_MDS')
    swr_anomaly_vod = getdata_predictor(vod_df,'SW_IN_F_MDS')
    ts_anomaly_vod = getdata_predictor(vod_df,'TS_F_MDS_1')
    vod_anom_df = pd.DataFrame(vod_anomaly,columns=[predictands[2]])
    vod_anom_df[varnames[0]] = ta_anomaly_vod
    vod_anom_df[varnames[1]] = vpd_anomaly_vod
    vod_anom_df[varnames[2]] = swr_anomaly_vod
    vod_anom_df[varnames[3]] = ts_anomaly_vod
    vod_anom_df['Dates'] = vod_df['Dates']
    
    vod_growing_df = vod_anom_df[vod_anom_df.Dates.str.contains(pattern) == False]

    sif_ta_granger = run_granger_causality_dates(sif_growing_df,daterange,datestrings,predictands[1],varnames[0])
    sif_vpd_granger = run_granger_causality_dates(sif_growing_df,daterange,datestrings,predictands[1],varnames[1])
    sif_swr_granger = run_granger_causality_dates(sif_growing_df,daterange,datestrings,predictands[1],varnames[2])
    sif_ts_granger = run_granger_causality_dates(sif_growing_df,daterange,datestrings,predictands[1],varnames[3])

    ndvi_ta_granger = run_granger_causality_dates(ndvi_growing_df,ndvimonthlength,ndvimonthstrings,predictands[0],varnames[0])
    ndvi_vpd_granger = run_granger_causality_dates(ndvi_growing_df,ndvimonthlength,ndvimonthstrings,predictands[0],varnames[1])
    ndvi_swr_granger = run_granger_causality_dates(ndvi_growing_df,ndvimonthlength,ndvimonthstrings,predictands[0],varnames[2])
    ndvi_ts_granger = run_granger_causality_dates(ndvi_growing_df,ndvimonthlength,ndvimonthstrings,predictands[0],varnames[3])
    
    vod_ta_granger = run_granger_causality_dates(vod_growing_df,vodmonthlength,vodmonthstrings,predictands[2],varnames[0])
    vod_vpd_granger = run_granger_causality_dates(vod_growing_df,vodmonthlength,vodmonthstrings,predictands[2],varnames[1])
    vod_swr_granger = run_granger_causality_dates(vod_growing_df,vodmonthlength,vodmonthstrings,predictands[2],varnames[2])
    vod_ts_granger = run_granger_causality_dates(vod_growing_df,vodmonthlength,vodmonthstrings,predictands[2],varnames[3])

    sif_granger = pd.concat([sif_ta_granger['TA_p-value'],sif_vpd_granger['VPD_p-value'],sif_swr_granger['SWR_p-value'],
                             sif_ts_granger['TS_p-value']],axis=1)
    ndvi_granger = pd.concat([ndvi_ta_granger['TA_p-value'],ndvi_vpd_granger['VPD_p-value'],ndvi_swr_granger['SWR_p-value'],
                             ndvi_ts_granger['TS_p-value']],axis=1)
    vod_granger = pd.concat([vod_ta_granger['TA_p-value'],vod_vpd_granger['VPD_p-value'],vod_swr_granger['SWR_p-value'],
                             vod_ts_granger['TS_p-value']],axis=1)
    
    #with pd.ExcelWriter("%s_Granger_Causality_%s.xlsx" %(substring,today_date)) as writer:
    #     sif_granger.to_excel(writer,sheet_name='SIF')
    #     ndvi_granger.to_excel(writer,sheet_name='NDVI')
    with pd.ExcelWriter("%s_Granger_Causality_VOD_%s.xlsx" %(substring,today_date)) as writer:
         sif_granger.to_excel(writer,sheet_name='SIF')
         ndvi_granger.to_excel(writer,sheet_name='NDVI')
         vod_granger.to_excel(writer,sheet_name='VOD')

elif (substring == 'FLX_GL-NuF'):
    sif_df = sif_df[(sif_df.TA_F_MDS_QC >= 0.75) & (sif_df.VPD_F_MDS_QC >= 0.75) & (sif_df.SW_IN_F_MDS_QC >= 0.75)
                    & (sif_df.TS_F_MDS_1_QC >= 0.75)]

    ndvi_df = ndvi_df[(ndvi_df.TA_F_MDS_QC >= 0.75) & (ndvi_df.VPD_F_MDS_QC >= 0.75) & (ndvi_df.SW_IN_F_MDS_QC >= 0.75)
                    & (ndvi_df.TS_F_MDS_1_QC >= 0.75)]

    sif_anomaly = getdata_predictand(sif_df,'SIF')
    ta_anomaly = getdata_predictor(sif_df,'TA_F_MDS')
    vpd_anomaly = getdata_predictor(sif_df,'VPD_F_MDS')
    swr_anomaly = getdata_predictor(sif_df,'SW_IN_F_MDS')
    ts_anomaly = getdata_predictor(sif_df,'TS_F_MDS_1')
    sif_anom_df = pd.DataFrame(sif_anomaly,columns=[predictands[1]])
    sif_anom_df[varnames[0]] = ta_anomaly
    sif_anom_df[varnames[1]] = vpd_anomaly
    sif_anom_df[varnames[2]] = swr_anomaly
    sif_anom_df[varnames[3]] = ts_anomaly
    sif_anom_df['Dates'] = sif_df['Dates']

    sif_growing_df = sif_anom_df[sif_anom_df.Dates.str.contains(pattern) == False]

    ndvi_anomaly = getdata_predictand(ndvi_df,'NDVI')
    ta_anomaly_ndvi = getdata_predictor(ndvi_df,'TA_F_MDS')
    vpd_anomaly_ndvi = getdata_predictor(ndvi_df,'VPD_F_MDS')
    swr_anomaly_ndvi = getdata_predictor(ndvi_df,'SW_IN_F_MDS')
    ts_anomaly_ndvi = getdata_predictor(ndvi_df,'TS_F_MDS_1')
    ndvi_anom_df = pd.DataFrame(ndvi_anomaly,columns=[predictands[0]])
    ndvi_anom_df[varnames[0]] = ta_anomaly_ndvi
    ndvi_anom_df[varnames[1]] = vpd_anomaly_ndvi
    ndvi_anom_df[varnames[2]] = swr_anomaly_ndvi
    ndvi_anom_df[varnames[3]] = ts_anomaly_ndvi
    ndvi_anom_df['Dates'] = ndvi_df['Dates']

    ndvi_growing_df = ndvi_anom_df[ndvi_anom_df.Dates.str.contains(pattern) == False]

    sif_ta_granger = run_granger_causality_dates(sif_growing_df,daterange,datestrings,predictands[1],varnames[0])
    sif_vpd_granger = run_granger_causality_dates(sif_growing_df,daterange,datestrings,predictands[1],varnames[1])
    sif_swr_granger = run_granger_causality_dates(sif_growing_df,daterange,datestrings,predictands[1],varnames[2])
    sif_ts_granger = run_granger_causality_dates(sif_growing_df,daterange,datestrings,predictands[1],varnames[3])

    ndvi_ta_granger = run_granger_causality_dates(ndvi_growing_df,ndvimonthlength,ndvimonthstrings,predictands[0],varnames[0])
    ndvi_vpd_granger = run_granger_causality_dates(ndvi_growing_df,ndvimonthlength,ndvimonthstrings,predictands[0],varnames[1])
    ndvi_swr_granger = run_granger_causality_dates(ndvi_growing_df,ndvimonthlength,ndvimonthstrings,predictands[0],varnames[2])
    ndvi_ts_granger = run_granger_causality_dates(ndvi_growing_df,ndvimonthlength,ndvimonthstrings,predictands[0],varnames[3])

    sif_granger = pd.concat([sif_ta_granger['TA_p-value'],sif_vpd_granger['VPD_p-value'],sif_swr_granger['SWR_p-value'],
                             sif_ts_granger['TS_p-value']],axis=1)
    ndvi_granger = pd.concat([ndvi_ta_granger['TA_p-value'],ndvi_vpd_granger['VPD_p-value'],ndvi_swr_granger['SWR_p-value'],
                             ndvi_ts_granger['TS_p-value']],axis=1)
    
    #with pd.ExcelWriter("%s_Granger_Causality_%s.xlsx" %(substring,today_date)) as writer:
    #     sif_granger.to_excel(writer,sheet_name='SIF')
    #     ndvi_granger.to_excel(writer,sheet_name='NDVI')
else:
    sif_df = sif_df[(sif_df.TA_F_MDS_QC >= 0.75) & (sif_df.VPD_F_MDS_QC >= 0.75) & (sif_df.SW_IN_F_MDS_QC >= 0.75)
                    & (sif_df.TS_F_MDS_1_QC >= 0.75) & (sif_df.SWC_F_MDS_1_QC >= 0.75)]
    
    ndvi_df = ndvi_df[(ndvi_df.TA_F_MDS_QC >= 0.75) & (ndvi_df.VPD_F_MDS_QC >= 0.75) & (ndvi_df.SW_IN_F_MDS_QC >= 0.75)
                    & (ndvi_df.TS_F_MDS_1_QC >= 0.75) & (ndvi_df.SWC_F_MDS_1_QC >= 0.75)]
    
    vod_df = vod_df[(vod_df.TA_F_MDS_QC >= 0.75) & (vod_df.VPD_F_MDS_QC >= 0.75) & (vod_df.SW_IN_F_MDS_QC >= 0.75)
                    & (vod_df.TS_F_MDS_1_QC >= 0.75) & (vod_df.SWC_F_MDS_1_QC >= 0.75)]
    vod_df = vod_df.dropna()
    
    sif_anomaly = getdata_predictand(sif_df,'SIF')
    ta_anomaly = getdata_predictor(sif_df,'TA_F_MDS')
    vpd_anomaly = getdata_predictor(sif_df,'VPD_F_MDS')
    swr_anomaly = getdata_predictor(sif_df,'SW_IN_F_MDS')
    ts_anomaly = getdata_predictor(sif_df,'TS_F_MDS_1')
    swc_anomaly = getdata_predictor(sif_df,'SWC_F_MDS_1')
    sif_anom_df = pd.DataFrame(sif_anomaly,columns=[predictands[1]])
    sif_anom_df[varnames[0]] = ta_anomaly
    sif_anom_df[varnames[1]] = vpd_anomaly
    sif_anom_df[varnames[2]] = swr_anomaly
    sif_anom_df[varnames[3]] = ts_anomaly
    sif_anom_df[varnames[4]] = swc_anomaly
    sif_anom_df['Dates'] = sif_df['Dates']
    
    sif_growing_df = sif_anom_df[sif_anom_df.Dates.str.contains(pattern) == False]
    
    ndvi_anomaly = getdata_predictand(ndvi_df,'NDVI')
    ta_anomaly_ndvi = getdata_predictor(ndvi_df,'TA_F_MDS')
    vpd_anomaly_ndvi = getdata_predictor(ndvi_df,'VPD_F_MDS')
    swr_anomaly_ndvi = getdata_predictor(ndvi_df,'SW_IN_F_MDS')
    ts_anomaly_ndvi = getdata_predictor(ndvi_df,'TS_F_MDS_1')
    swc_anomaly_ndvi = getdata_predictor(ndvi_df,'SWC_F_MDS_1')
    ndvi_anom_df = pd.DataFrame(ndvi_anomaly,columns=[predictands[0]])
    ndvi_anom_df[varnames[0]] = ta_anomaly_ndvi
    ndvi_anom_df[varnames[1]] = vpd_anomaly_ndvi
    ndvi_anom_df[varnames[2]] = swr_anomaly_ndvi
    ndvi_anom_df[varnames[3]] = ts_anomaly_ndvi
    ndvi_anom_df[varnames[4]] = swc_anomaly_ndvi
    ndvi_anom_df['Dates'] = ndvi_df['Dates']
    
    ndvi_growing_df = ndvi_anom_df[ndvi_anom_df.Dates.str.contains(pattern) == False]
    '''
    april_strings = ndvi_growing_df['Dates'][ndvi_growing_df.Dates.str.contains('-04-')].tolist()
    may_strings = ndvi_growing_df['Dates'][ndvi_growing_df.Dates.str.contains('-05-')].tolist()
    june_strings = ndvi_growing_df['Dates'][ndvi_growing_df.Dates.str.contains('-06-')].tolist()
    july_strings = ndvi_growing_df['Dates'][ndvi_growing_df.Dates.str.contains('-07-')].tolist()
    august_strings = ndvi_growing_df['Dates'][ndvi_growing_df.Dates.str.contains('-08-')].tolist()
    september_strings = ndvi_growing_df['Dates'][ndvi_growing_df.Dates.str.contains('-09-')].tolist()
    '''
    
    vod_anomaly = getdata_predictand(vod_df,'VOD')
    ta_anomaly_vod = getdata_predictor(vod_df,'TA_F_MDS')
    vpd_anomaly_vod = getdata_predictor(vod_df,'VPD_F_MDS')
    swr_anomaly_vod = getdata_predictor(vod_df,'SW_IN_F_MDS')
    ts_anomaly_vod = getdata_predictor(vod_df,'TS_F_MDS_1')
    swc_anomaly_vod = getdata_predictor(vod_df,'SWC_F_MDS_1')
    vod_anom_df = pd.DataFrame(vod_anomaly,columns=[predictands[2]])
    vod_anom_df[varnames[0]] = ta_anomaly_vod
    vod_anom_df[varnames[1]] = vpd_anomaly_vod
    vod_anom_df[varnames[2]] = swr_anomaly_vod
    vod_anom_df[varnames[3]] = ts_anomaly_vod
    vod_anom_df[varnames[4]] = swc_anomaly_vod
    vod_anom_df['Dates'] = vod_df['Dates']
    
    vod_growing_df = vod_anom_df[vod_anom_df.Dates.str.contains(pattern) == False]
    
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
    
    sif_granger = pd.concat([sif_ta_granger['TA_p-value'],sif_vpd_granger['VPD_p-value'],sif_swr_granger['SWR_p-value'],
                             sif_ts_granger['TS_p-value'],sif_swc_granger['SWC_p-value']],axis=1)
    ndvi_granger = pd.concat([ndvi_ta_granger['TA_p-value'],ndvi_vpd_granger['VPD_p-value'],ndvi_swr_granger['SWR_p-value'],
                             ndvi_ts_granger['TS_p-value'],ndvi_swc_granger['SWC_p-value']],axis=1)
    vod_granger = pd.concat([vod_ta_granger['TA_p-value'],vod_vpd_granger['VPD_p-value'],vod_swr_granger['SWR_p-value'],
                             vod_ts_granger['TS_p-value'],vod_swc_granger['SWC_p-value']],axis=1)
    #with pd.ExcelWriter("%s_Granger_Causality_%s.xlsx" %(substring,today_date)) as writer:
    #     sif_granger.to_excel(writer,sheet_name='SIF')
    #     ndvi_granger.to_excel(writer,sheet_name='NDVI')

    with pd.ExcelWriter("%s_Granger_Causality_VOD_%s.xlsx" %(substring,today_date)) as writer:
         sif_granger.to_excel(writer,sheet_name='SIF')
         ndvi_granger.to_excel(writer,sheet_name='NDVI')
         vod_granger.to_excel(writer,sheet_name='VOD')
