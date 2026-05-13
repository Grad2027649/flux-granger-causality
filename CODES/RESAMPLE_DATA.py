#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 11:48:31 2024

@author: cassandracalderella
"""

import pandas as pd
import READ_DATA as RD

pd.options.mode.chained_assignment = None 

sites = pd.read_excel("Fluxnet_Sites.xlsx",sheet_name='Metadata',names=["Station", "Latitude", "Longitude", "Modis_Lat", "Modis_Lon", "Long_Name"])
stations = sites['Station'].values
latitudes = sites['Latitude'].values
longitudes = sites['Longitude'].values

units_dict = {'TA':'degC','SW':'W/m**2','LW':'W/m**2','VPD':'hPa','PA':'kPa','P':'kg/m**2/s',
          'WS':'m/s','WD':'degree','RH':'Pa/hPa','USTAR':'m/s','NETRAD':'W/m**2',
          'PPFD':'micromole/s/m**2','CO2':'micromole/mole','TS':'degC','SWC':'mm**2/cm**2',
          'G':'W/m**2','LE':'W/m**2','H':'W/m**2','NEE':'micromole/s/m**2',
          'RECO':'micromole/s/m**2','GPP':'micromole/s/m**2','SIF': 'mW m-2 nm-1 sr-1'}

years = ['1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006',
         '2007','2008','2009','2010','2011','2012','2013','2014','2015',
         '2016','2017','2018','2019','2020','2021']

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

Years
~~~~~
1996 [0]
1997 [1]
1998 [2]
1999 [3]
2000 [4]
2001 [5]
2002 [6]
2003 [7]
2004 [8]
2005 [9]
2006 [10]
2007 [11]
2008 [12]
2009 [13]
2010 [14]
2011 [15]
2012 [16]
2013 [17]
2014 [18]
2015 [19]
2016 [20]
2017 [21]
2018 [22]
2019 [23]
2020 [24]
2021 [25]
'''

'''
Get data
'''
filename,pixellat,pixellon,longname,substring,station_lat,station_lon = RD.read_fluxnet_station(stations[16])
start,end = RD.getyears(filename)
startindex = years.index(start)
endindex = years.index(end)
if ((substring == 'FLX_FI-Hyy') or (substring == 'FLX_FI-Sod') or (substring == 'FLX_GL-ZaH') or (substring == 'FLX_SE_Deg') or (substring == 'FLX_CA-Oas')
    or (substring == 'FLX_RU-Fyo') or (substring == 'FLX_DK-Sor')):
    yearrange = years[7:endindex+1]
    yearlen = len(yearrange)
else:
   yearrange = years[startindex:endindex+1]
   yearlen = len(yearrange)
   
'''
Get MODIS NDVI Data
'''

ndvi = []
for i in range(yearlen):
    ndviak,aklat,aklon = RD.open_ndvi(yearrange[i],pixellat,pixellon)
    ndvi.append(ndviak)
    
ndvi_df = pd.concat(ndvi)


'''
Get FLUXNET Data and scale the time to match MODIS NDVI and SIF
'''
modis_df = []
sif_df = []
vod_df = []
for j in range(yearlen):
    fluxndvi,fluxsif,fluxvod = RD.open_fluxnet(yearrange[j],filename)
    modis_df.append(fluxndvi)
    sif_df.append(fluxsif)
    vod_df.append(fluxvod)

fluxnet_df_ndvi = pd.concat(modis_df)
fluxnet_df_sif = pd.concat(sif_df)
fluxnet_df_vod = pd.concat(vod_df)

'''
Get LCSIF Data
'''

sif = []
for k in range(yearlen):
    lcsif = RD.open_lcsif(yearrange[k],pixellat,pixellon)
    sif.append(lcsif)
    
lcsif_df = pd.concat(sif)
lcsif_df['Dates'] = fluxnet_df_sif['TIMESTAMP'].values
lcsif_df = lcsif_df.iloc[:,[1,0]]

'''
Get VODCA_L Data
'''

vod = []
for m in range(yearlen):
    vodcal = RD.open_vodcal(yearrange[m],station_lat,station_lon)
    vod.append(vodcal)
    
vodcal_df = pd.concat(vod)
fluxnet_df_vod2 = fluxnet_df_vod[255:]
vodcal_df['Dates'] = fluxnet_df_vod2['TIMESTAMP'].values
vodcal_df = vodcal_df.iloc[:,[1,0]]

#print(fluxnet_df_vod.index[fluxnet_df_vod['TIMESTAMP'] == '2010-02-01'])
#fluxnet_df_vod3 = fluxnet_df_vod.reset_index(drop=True)

#print(fluxnet_df_ndvi.iloc[28:])

'''
NDVI
'''
gd_NDVI = RD.create_ndvi_dataset(ndvi_df,fluxnet_df_ndvi)

'''
SIF
'''

gd_SIF = RD.create_sif_dataset(lcsif_df,fluxnet_df_sif)
gd_VOD = RD.create_vod_dataset(vodcal_df,fluxnet_df_vod2)

'''
Save to File
'''
with pd.ExcelWriter("STATION_FILES_QC/%s.xlsx" %(substring)) as writer:
     gd_SIF.to_excel(writer,sheet_name='SIF',index=False)
     gd_NDVI.to_excel(writer,sheet_name='NDVI',index=False)
     gd_VOD.to_excel(writer,sheet_name='VOD',index=False)
     
