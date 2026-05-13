#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:30:20 2024

@author: cassandracalderella
"""

import pandas as pd
import numpy as np
from pyhdf.SD import SD, SDC
from glob import glob
from netCDF4 import Dataset
from datetime import datetime
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import cycle, islice
import warnings
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 16
plt.rcParams["figure.autolayout"] = True
pd.options.mode.chained_assignment = None
sites = pd.read_excel("Fluxnet_Sites.xlsx",sheet_name='Metadata',names=["Station", "Latitude", "Longitude", "Modis_Lat", "Modis_Lon", "Long_Name"])
stations = sites['Station'].values
modislats = sites['Modis_Lat'].values
modislons = sites['Modis_Lon'].values
longnames = sites['Long_Name'].values
latitudes = sites['Latitude'].values
longitudes = sites['Longitude'].values

geofile = '/Volumes/Expansion/MODIS_Geolocation.xlsx'
geo = pd.read_excel(geofile)
lat = geo['Lat'].values
lat = lat[0:3600]
lon = geo['Lon'].values

'''
Get File Name
'''
def read_fluxnet_station(stationstring):
    path = sorted(glob("/Volumes/Expansion/FLUXNET/"+stationstring+"_FLUXNET*/*FULLSET_DD*.csv"))
    statid = path[0][34:37]
    if statid == 'EML':
       pixellat = modislats[0]
       pixellon = modislons[0]
       long_name = longnames[0]
       substring = stations[0]
       station_lat = latitudes[0]
       station_lon = longitudes[0]
    elif statid == 'ICh':
       pixellat = modislats[1]
       pixellon = modislons[1]
       long_name = longnames[1]
       substring = stations[1]
       station_lat = latitudes[1]
       station_lon = longitudes[1]
    elif statid == 'Fcr':
       pixellat = modislats[2]
       pixellon = modislons[2]
       long_name = longnames[2]
       substring = stations[2]
       station_lat = latitudes[2]
       station_lon = longitudes[2]
    elif statid == 'Hyy':
       pixellat = modislats[3]
       pixellon = modislons[3]
       long_name = longnames[3]
       substring = stations[3]
       station_lat = latitudes[3]
       station_lon = longitudes[3]
    elif statid == 'Sod':
       pixellat = modislats[4]
       pixellon = modislons[4]
       long_name = longnames[4]
       substring = stations[4]
       station_lat = latitudes[4]
       station_lon = longitudes[4]
    elif statid == 'NuF':
       pixellat = modislats[5]
       pixellon = modislons[5]
       long_name = longnames[5]
       substring = stations[5]
       station_lat = latitudes[5]
       station_lon = longitudes[5]
    elif statid == 'ZaH':
       pixellat = modislats[6]
       pixellon = modislons[6]
       long_name = longnames[6]
       substring = stations[6]
       station_lat = latitudes[6]
       station_lon = longitudes[6]
    elif statid == 'Cok':
       pixellat = modislats[7]
       pixellon = modislons[7]
       long_name = longnames[7]
       substring = stations[7]
       station_lat = latitudes[7]
       station_lon = longitudes[7]
    elif statid == 'Adv':
       pixellat = modislats[8]
       pixellon = modislons[8]
       long_name = longnames[8]
       substring = stations[8]
       station_lat = latitudes[8]
       station_lon = longitudes[8]
    elif statid == 'Prr':
       pixellat = modislats[9]
       pixellon = modislons[9]
       long_name = longnames[9]
       substring = stations[9]
       station_lat = latitudes[9]
       station_lon = longitudes[9]
    elif statid == 'BZF':
       pixellat = modislats[10]
       pixellon = modislons[10]
       long_name = longnames[10]
       substring = stations[10]
       station_lat = latitudes[10]
       station_lon = longitudes[10]
    elif statid == 'Deg':
       pixellat = modislats[11]
       pixellon = modislons[11]
       long_name = longnames[11]
       substring = stations[11]
       station_lat = latitudes[11]
       station_lon = longitudes[11]
    elif statid == 'Oas':
       pixellat = modislats[12]
       pixellon = modislons[12]
       long_name = longnames[12]
       substring = stations[12]
       station_lat = latitudes[12]
       station_lon = longitudes[12]
    elif statid == 'SDF':
       pixellat = modislats[13]
       pixellon = modislons[13]
       long_name = longnames[13]
       substring = stations[13]
       station_lat = latitudes[13]
       station_lon = longitudes[13]
    elif statid == 'Ha2':
       pixellat = modislats[14]
       pixellon = modislons[14]
       long_name = longnames[14]
       substring = stations[14]
       station_lat = latitudes[14]
       station_lon = longitudes[14]
    elif statid == 'HaM':
       pixellat = modislats[15]
       pixellon = modislons[15]
       long_name = longnames[15]
       substring = stations[15]
       station_lat = latitudes[15]
       station_lon = longitudes[15]
    elif statid == 'Fyo':
       pixellat = modislats[16]
       pixellon = modislons[16]
       long_name = longnames[16]
       substring = stations[16]
       station_lat = latitudes[16]
       station_lon = longitudes[16]
    elif statid == 'Sor':
       pixellat = modislats[17]
       pixellon = modislons[17]
       long_name = longnames[17]
       substring = stations[17]
       station_lat = latitudes[17]
       station_lon = longitudes[17]
    elif statid == 'LP1':
       pixellat = modislats[18]
       pixellon = modislons[18]
       long_name = longnames[18]
       substring = stations[18]
       station_lat = latitudes[18]
       station_lon = longitudes[18]
    return path[0],pixellat,pixellon,long_name,substring,station_lat,station_lon

'''
Get Years From File Names
'''
def getyears(file_name):
    if "AMF" in file_name:
       start_year = file_name[54:58]
       end_year = file_name[59:63]
    else:
       start_year = file_name[106:110]
       end_year = file_name[111:115]
    return start_year,end_year

def getyears2(file_name):
    start_year = file_name[102:106]
    end_year = file_name[107:111]
    return start_year,end_year

'''
VODCA_L Custom Date Grouper
'''
def custom_grouper(x):
    month_start = x.replace(day=1)
    return pd.Timestamp(month_start) + pd.Timedelta(days=int((x.day - 2) / 10) * 10)

'''
Open Fluxnet File
'''
def open_fluxnet(year,filename):
    f = pd.read_csv(filename)
    dfname = ['TIMESTAMP','TA_F_MDS','VPD_F_MDS','SW_IN_F_MDS','TS_F_MDS_1','SWC_F_MDS_1',
              'TA_F_MDS_QC','VPD_F_MDS_QC','SW_IN_F_MDS_QC','TS_F_MDS_1_QC','SWC_F_MDS_1_QC']
    varnames = {}
    for i in dfname:
        if i in f:
           varnames[i] = f[i].values
        else:
           continue
    df = pd.DataFrame.from_dict(varnames)
    df['TIMESTAMP'] = pd.to_datetime(f["TIMESTAMP"],format='%Y%m%d').dt.date.to_numpy(dtype='str')
    yeardf = df[df.TIMESTAMP.str.startswith(year)]
    yeardf['TIMESTAMP'] = pd.to_datetime(yeardf['TIMESTAMP'])
    '''
    NDVI
    '''
    df_ndvi = yeardf.groupby(pd.Grouper(key='TIMESTAMP',freq='16D')).mean()
    df_ndvi = df_ndvi.reset_index()
    '''
    SIF
    '''
    df_sif = yeardf.groupby(pd.Grouper(key='TIMESTAMP',freq='SM')).mean()
    df_sif = df_sif.reset_index()
    df_sif = df_sif[1:]
    '''
    VOD
    '''
    grouped_vod = yeardf
    grouped_vod['TIMESTAMP'] = grouped_vod['TIMESTAMP'].apply(custom_grouper)
    df_vod = grouped_vod.groupby('TIMESTAMP').mean()
    df_vod = df_vod.reset_index()
    return df_ndvi,df_sif,df_vod
    
'''
Open MODIS NDVI
'''
def open_ndvi(year,coordlat,coordlon):
    path = sorted(glob('/Volumes/Expansion/MODIS_VEGETATION/MOD13C1.A'+year+'*'+'.hdf'))
    ndviak = []
    julianday = []
    aklat = lat[coordlat]
    aklon = lon[coordlon]
    ndvidates = []
    for i,f in enumerate(path):
        filedate = f[45:52]
        hdf = SD(f,SDC.READ)
        ndvi = hdf.select('CMG 0.05 Deg 16 days NDVI').get()
        ndvi = ndvi[coordlat,coordlon]
        ndvi = ndvi/10000
        ndviak.append(ndvi)
        dates = datetime.strptime(filedate,'%Y%j').date()
        ndvidates.append(dates)
        julianday.append(filedate[4:7])
    datearray = np.array(ndvidates)
    ndviarray = np.array(ndviak)
    ndvidf = pd.DataFrame(datearray,columns=['Date'])
    ndvidf['NDVI'] = ndviarray
    return ndvidf,aklat,aklon

'''
Open LCSIF
'''
def open_lcsif(year,coordlat,coordlon):
    path = sorted(glob('/Volumes/Expansion/LCSIF/'+year+'/*.nc'))
    sif = []
    for i,f in enumerate(path):
        nc = Dataset(f,'r')
        var = nc.variables['sif_clear_daily'][:,coordlat,coordlon]
        sif.append(var)
    sifdf = pd.DataFrame(sif,columns=['SIF'])
    return sifdf

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

'''
Open VODCA_L
'''
def open_vodcal(year,x_lat,x_lon):
    path = sorted(glob('/Volumes/Expansion/VODCA_L/'+year+'/*.nc'))
    vod = []
    for i,f in enumerate(path):
        nc = Dataset(f,'r')
        lats = nc.variables['lat'][:]
        lons = nc.variables['lon'][:]
        var_lat_idx,var_lat = find_nearest(lats,value=x_lat)
        var_lon_idx,var_lon = find_nearest(lons,value=x_lon)
        var = nc.variables['VODCA_L'][:,var_lat_idx,var_lon_idx]
        vod.append(var)
    voddf = pd.DataFrame(vod,columns=['VOD'])
    return voddf

'''
Create datasets for Granger-Causality Test (for NDVI and SIF)
'''
def create_ndvi_dataset(ndvidf,fluxdf):
    dfname = ['TA_F_MDS','VPD_F_MDS','SW_IN_F_MDS','TS_F_MDS_1','SWC_F_MDS_1',
              'TA_F_MDS_QC','VPD_F_MDS_QC','SW_IN_F_MDS_QC','TS_F_MDS_1_QC','SWC_F_MDS_1_QC']
    df = {}
    for j in dfname:
        if j in fluxdf:
           df[j] = fluxdf[j].values
        else:
           continue
    granger_dataset_NDVI = pd.DataFrame.from_dict(df)
    granger_dataset_NDVI[granger_dataset_NDVI < -1000] = np.nan
    granger_dataset_NDVI['Dates'] = ndvidf['Date'].values
    granger_dataset_NDVI['Dates'] = pd.to_datetime(granger_dataset_NDVI["Dates"],format='%Y-%m-%d').dt.date.to_numpy(dtype='str')
    granger_dataset_NDVI['NDVI'] = ndvidf['NDVI'].values
    return granger_dataset_NDVI

def create_sif_dataset(sifdf,fluxdf):
    dfname = ['TA_F_MDS','VPD_F_MDS','SW_IN_F_MDS','TS_F_MDS_1','SWC_F_MDS_1',
              'TA_F_MDS_QC','VPD_F_MDS_QC','SW_IN_F_MDS_QC','TS_F_MDS_1_QC','SWC_F_MDS_1_QC']
    df = {}
    for j in dfname:
        if j in fluxdf:
           df[j] = fluxdf[j].values
        else:
           continue
    granger_dataset_SIF = pd.DataFrame.from_dict(df)
    granger_dataset_SIF[granger_dataset_SIF < -1000] = np.nan
    granger_dataset_SIF['Dates'] = sifdf['Dates'].values
    granger_dataset_SIF['Dates'] = pd.to_datetime(granger_dataset_SIF["Dates"],format='%Y-%m-%d').dt.date.to_numpy(dtype='str')
    granger_dataset_SIF['SIF'] = sifdf['SIF'].values
    return granger_dataset_SIF

def create_vod_dataset(voddf,fluxdf):
    dfname = ['TA_F_MDS','VPD_F_MDS','SW_IN_F_MDS','TS_F_MDS_1','SWC_F_MDS_1',
              'TA_F_MDS_QC','VPD_F_MDS_QC','SW_IN_F_MDS_QC','TS_F_MDS_1_QC','SWC_F_MDS_1_QC']
    df = {}
    for j in dfname:
        if j in fluxdf:
           df[j] = fluxdf[j].values
        else:
           continue
    granger_dataset_VOD = pd.DataFrame.from_dict(df)
    granger_dataset_VOD[granger_dataset_VOD < -1000] = np.nan
    granger_dataset_VOD['Dates'] = voddf['Dates'].values
    granger_dataset_VOD['Dates'] = pd.to_datetime(granger_dataset_VOD["Dates"],format='%Y-%m-%d').dt.date.to_numpy(dtype='str')
    granger_dataset_VOD['VOD'] = voddf['VOD'].values
    return granger_dataset_VOD

'''
Run Granger Causality Test
'''
def run_granger_causality(df,yearlength,rangeofyears,predictand,varstring):
    p_value = []
    yr = []
    columns=[1,2,3,4,5,6]
    for m in range(yearlength):
        gd = df[df.Dates.str.startswith(rangeofyears[m])]
        check_nan = gd[varstring].isnull().values.any()
        if check_nan:
            continue
        else:
            gc = grangercausalitytests(gd[[predictand,varstring]],maxlag=6,verbose=False)
            for a in range(1,7):
                res = gc[a][0]['ssr_ftest'][1]
                p_value.append(res)
                yr.append(rangeofyears[m])
            pvaluedf = pd.DataFrame(p_value,columns=['p-value'])
            pvaluedf['Year'] = yr
    pvaluedf['Lag'] = list(islice(cycle(columns),len(pvaluedf)))
    pvaluedf = pvaluedf.iloc[:,[2,1,0]]
    pvaluedf['Lag'] = pvaluedf['Lag'].astype(str)
    pvaluedf = pvaluedf.set_index('Year')
    
    lag1 = pvaluedf[pvaluedf.Lag.str.startswith('1')]
    lag2 = pvaluedf[pvaluedf.Lag.str.startswith('2')]
    lag3 = pvaluedf[pvaluedf.Lag.str.startswith('3')]
    lag4 = pvaluedf[pvaluedf.Lag.str.startswith('4')]
    lag5 = pvaluedf[pvaluedf.Lag.str.startswith('5')]
    lag6 = pvaluedf[pvaluedf.Lag.str.startswith('6')]
    return lag1,lag2,lag3,lag4,lag5,lag6

def create_growing_dataset_NDVI(df,yearlength,rangeofyears):
    gs = []
    for i in range(yearlength):
        gd = df[df.Dates.str.startswith(rangeofyears[i])]
        gd[gd['NDVI'] < 0] = np.nan
        data = gd.dropna()
        gs.append(data)
    gs_df = pd.concat(gs)
    return gs_df

def create_growing_dataset_SIF(df,yearlength,rangeofyears):
    gs = []
    for i in range(yearlength):
        gd = df[df.Dates.str.startswith(rangeofyears[i])]
        gd[gd['SIF'] < 0.1] = np.nan
        data = gd.dropna()
        gs.append(data)
    gs_df = pd.concat(gs)
    return gs_df

def granger_growing_season(data,yearlength,rangeofyears,predictand,varstring):
    p_value = []
    yr = []
    columns=[1]
    for j in range(yearlength):
        df = data[data.Dates.str.startswith(rangeofyears[j])]
        gc = grangercausalitytests(df[[predictand,varstring]],maxlag=1,verbose=False)
        '''
        for a in range(1,2):
            res = gc[a][0]['ssr_ftest'][1]
            p_value.append(res)
            yr.append(rangeofyears[j])
        '''
        res = gc[1][0]['ssr_ftest'][1]
        p_value.append(res)
        yr.append(rangeofyears[j])
        pvaluedf = pd.DataFrame(p_value,columns=['p-value'])
        pvaluedf['Year'] = yr
        pvaluedf['Lag'] = list(islice(cycle(columns),len(pvaluedf)))
        pvaluedf = pvaluedf.iloc[:,[2,1,0]]
        pvaluedf['Lag'] = pvaluedf['Lag'].astype(str)
        pvaluedf = pvaluedf.set_index('Year')
    
    lag1 = pvaluedf[pvaluedf.Lag.str.startswith('1')]
    return lag1

def plot_results(ndvilag1,ndvilag2,ndvilag3,ndvilag4,ndvilag5,ndvilag6,siflag1,siflag2,siflag3,siflag4,siflag5,siflag6,
                 varlabel,varname,name,lat,lon,statid,predictand,threshold=0.05):
    fig,axes = plt.subplots(2,1,figsize=(14,8),sharex=True)
    #fig,axes = plt.subplots(2,1,figsize=(14,8))
    axes[0].plot(ndvilag1['p-value'],'--bo',label='Lag 1: 16 Days')
    axes[0].plot(ndvilag2['p-value'],'saddlebrown',linestyle='--',marker='o',label='Lag 2: 32 Days')
    axes[0].plot(ndvilag3['p-value'],'--go',label='Lag 3: 48 Days')
    axes[0].plot(ndvilag4['p-value'],'--ko',label='Lag 4: 64 Days')
    axes[0].plot(ndvilag5['p-value'],'--co',label='Lag 5: 80 Days')
    axes[0].plot(ndvilag6['p-value'],'--mo',label='Lag 6: 96 Days')
    axes[0].axhline(y=0.05, color='r', linestyle='-')
    axes[0].set_yscale('log')
    trans = transforms.blended_transform_factory(
    axes[0].get_yticklabels()[0].get_transform(), axes[0].transData)
    axes[0].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans, 
        ha="right", va="center")
    axes[0].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    axes[0].grid()
    axes[0].set_title(predictand[0])
    
    axes[1].plot(siflag1['p-value'],'--bo',label='Lag 1: 16 Days')
    axes[1].plot(siflag2['p-value'],'saddlebrown',linestyle='--',marker='o',label='Lag 2: 32 Days')
    axes[1].plot(siflag3['p-value'],'--go',label='Lag 3: 48 Days')
    axes[1].plot(siflag4['p-value'],'--ko',label='Lag 4: 64 Days')
    axes[1].plot(siflag5['p-value'],'--co',label='Lag 5: 80 Days')
    axes[1].plot(siflag6['p-value'],'--mo',label='Lag 6: 96 Days')
    axes[1].axhline(y=0.05, color='r', linestyle='-')
    axes[1].set_yscale('log')
    trans2 = transforms.blended_transform_factory(axes[1].get_yticklabels()[0].get_transform(), axes[1].transData)
    axes[1].text(0,threshold, "{:.2f}".format(threshold), color="red", transform=trans2, 
        ha="right", va="center")
    axes[1].grid()
    axes[1].set_title(predictand[1])
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Year")
    plt.ylabel("p-value")
    plt.tight_layout()
    
    fig.suptitle('Significance of %s at %s \n $%s^\circ$N, $%s^\circ$E' %(varlabel,name,lat,lon))
    fig.savefig('/Users/cassandracalderella/Library/CloudStorage/OneDrive-Personal/George Mason/CLIM 997/Codes/Images/%s_%s_GC.png' %(statid,varname),
                dpi=200,bbox_inches='tight')
    #fig.savefig('%s_%s_GC.png' %(statid,varname),dpi=200)
    
#def plot_growing(testlag1,testlag2,testlag3,testlag4,testlag5,siflag1,siflag2,siflag3,siflag4,siflag5,
#                 varlabel,name,lat,lon,statid,predictand,threshold=0.05):
def plot_growing(testlag1,testlag2,testlag3,testlag4,testlag5,siflag1,siflag2,siflag3,siflag4,siflag5,
                 varlabel,name,lat,lon,statid,predictand,threshold=0.05):
    fig,ax = plt.subplots(5,1,figsize=(25,20))
    ax[0].plot(testlag1['p-value'],'--bo')
    ax[0].plot(siflag1['p-value'],'--go')
    ax[1].plot(testlag2['p-value'],'--bo')
    ax[1].plot(siflag2['p-value'],'--go')
    ax[2].plot(testlag3['p-value'],'--bo')
    ax[2].plot(siflag3['p-value'],'--go')
    ax[3].plot(testlag4['p-value'],'--bo')
    ax[3].plot(siflag4['p-value'],'--go')
    ax[4].plot(testlag5['p-value'],'--bo')
    ax[4].plot(siflag5['p-value'],'--go')
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
    ax[0].legend([predictand[0],predictand[1]],bbox_to_anchor=(1.05, 1.0), loc='upper left')
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
    ax[0].set_title(varlabel[0])
    ax[1].set_title(varlabel[1])
    ax[2].set_title(varlabel[2])
    ax[3].set_title(varlabel[3])
    ax[4].set_title(varlabel[4])
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Year")
    plt.ylabel("p-value",labelpad=10)
    plt.tight_layout()
    fig.suptitle('Significance of Climate Drivers at %s \n $%s^\circ$N, $%s^\circ$E' %(name,lat,lon))
    #fig.suptitle('Significance of Climate Drivers on %s at %s \n $%s^\circ$N, $%s^\circ$E' %(predictand[0],name,lat,lon))
    fig.savefig('%s_GC.png' %(statid),dpi=200,bbox_inches='tight')
    #fig.savefig('%s_%s_GC.png' %(statid,predictand[0]),dpi=200,bbox_inches='tight')