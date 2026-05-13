#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 11:48:00 2025

@author: cassandracalderella
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from datetime import datetime

today_date = datetime.today().strftime('%Y%m%d')

pd.options.mode.chained_assignment = None

shortvars = ['TA','VPD','SWR','TS','SWC']

varlabels = ['Air Temperature Anomaly','Vapor Pressure Deficit Anomaly','Incoming Shortwave Radiation Anomaly',
             'Soil Temperature Anomaly','Soil Water Content Anomaly']

units_dict = {'TA':'$^\circ$C','SW':'$\mathregular{W/m^{2}}$','LW':'W/m**2','VPD':'hPa','PA':'kPa','P':'kg/m**2/s',
          'WS':'m/s','WD':'degree','RH':'Pa/hPa','USTAR':'m/s','NETRAD':'W/m**2',
          'PPFD':'micromole/s/m**2','CO2':'micromole/mole','TS':'$^\circ$C','SWC':'%',
          'G':'W/m**2','LE':'W/m**2','H':'W/m**2','NEE':'micromole/s/m**2',
          'RECO':'micromole/s/m**2','GPP':'micromole/s/m**2'}

TA_units = units_dict.get('TA')
VPD_units = units_dict.get('VPD')
SWR_units = units_dict.get('SW')
TS_units = units_dict.get('TS')
SWC_units = units_dict.get('SWC')

units_list = [TA_units,VPD_units,SWR_units,TS_units,SWC_units]

def getdata_predictor(df,causingvar):
    decompose_predictor = seasonal_decompose(df[causingvar],model='additive',period=1,extrapolate_trend='freq')
    predictor_trend = decompose_predictor.trend
    predictor_residual = decompose_predictor.resid
    predictor_mean = np.mean(df[causingvar].values)
    new_predictor_trend = predictor_trend - predictor_mean
    predictor_anomaly = new_predictor_trend + predictor_residual 
    return predictor_anomaly

def linear_regression(df_x,df_y,col):
    R_matrix = np.corrcoef(df_x[col].values,df_y[col].values)
    R = np.round(R_matrix[0][1],2)
    return R

bzf_flux_file = 'AMF_US-BZF_LINEAR_REGRESSION_20250524.xlsx'
sod_flux_file = 'FLX_FI-Sod_LINEAR_REGRESSION_20250524.xlsx'

bzf_flux_df = pd.read_excel(bzf_flux_file,sheet_name='FLUX')
bzf_era5_df = pd.read_excel(bzf_flux_file,sheet_name='ERA5')
sod_flux_df = pd.read_excel(sod_flux_file,sheet_name='FLUX')
sod_era5_df = pd.read_excel(sod_flux_file,sheet_name='ERA5')

def get_correlation(flux_df,era5_df):
    R_TA = linear_regression(flux_df,era5_df,shortvars[0])
    R_VPD = linear_regression(flux_df,era5_df,shortvars[1])
    R_SWR = linear_regression(flux_df,era5_df,shortvars[2])
    R_TS = linear_regression(flux_df,era5_df,shortvars[3])
    R_SWC = linear_regression(flux_df,era5_df,shortvars[4])

    text_ta = 'r='+str(R_TA)
    text_vpd = 'r='+str(R_VPD)
    text_isr = 'r='+str(R_SWR)
    text_ts = 'r='+str(R_TS)
    text_swc = 'r='+str(R_SWC)

    m1,b1 = np.polyfit(flux_df['TA'].values,era5_df['TA'].values,1)
    m2,b2 = np.polyfit(flux_df['VPD'].values,era5_df['VPD'].values,1)
    m3,b3 = np.polyfit(flux_df['SWR'].values,era5_df['SWR'].values,1)
    m4,b4 = np.polyfit(flux_df['TS'].values,era5_df['TS'].values,1)
    m5,b5 = np.polyfit(flux_df['SWC'].values,era5_df['SWC'].values,1)
    return text_ta,text_vpd,text_isr,text_ts,text_swc,m1,b1,m2,b2,m3,b3,m4,b4,m5,b5
    
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

bzf_text_ta,bzf_text_vpd,bzf_text_isr,bzf_text_ts,bzf_text_swc,bzf_m1,bzf_b1,bzf_m2,bzf_b2,bzf_m3,bzf_b3,bzf_m4,bzf_b4,bzf_m5,bzf_b5 = get_correlation(bzf_flux_df,bzf_era5_df)
sod_text_ta,sod_text_vpd,sod_text_isr,sod_text_ts,sod_text_swc,sod_m1,sod_b1,sod_m2,sod_b2,sod_m3,sod_b3,sod_m4,sod_b4,sod_m5,sod_b5 = get_correlation(sod_flux_df,sod_era5_df)

fig,ax = plt.subplots(nrows=5,ncols=2,figsize=(10,10))
ax[0,0].plot(bzf_flux_df['TA'].values,bzf_era5_df['TA'].values,'go',alpha=0.15)
ax[0,0].plot(bzf_era5_df['TA'].values,bzf_b1+bzf_m1*bzf_era5_df['TA'].values,'g-')
ax[0,0].grid(color='k')
ax[0,0].set_ylim(-20,20)
ax[0,0].yaxis.set_ticks((-20,-10,0,10,20))
ax[0,0].plot([], label=bzf_text_ta)
ax[0,0].legend(handlelength=0,labelcolor='g')

ax[0,1].plot(sod_flux_df['TA'].values,sod_era5_df['TA'].values,'go',alpha=0.15)
ax[0,1].plot(sod_era5_df['TA'].values,sod_b1+sod_m1*sod_era5_df['TA'].values,'g-')
ax[0,1].grid(color='k')
ax[0,1].set_ylim(-20,20)
ax[0,1].yaxis.set_ticks((-20,-10,0,10,20))
ax[0,1].plot([], label=sod_text_ta)
ax[0,1].legend(handlelength=0,labelcolor='g')
ax[0,0].set_title(' ')
fig.text(-0.05, 1.15, varlabels[0]+' (%s)' %(TA_units), horizontalalignment='center',
  verticalalignment='center', transform=ax[0,1].transAxes)

ax[1,0].plot(bzf_flux_df['VPD'].values,bzf_era5_df['VPD'].values,'bo',alpha=0.15)
ax[1,0].plot(bzf_era5_df['VPD'].values,bzf_b2+bzf_m2*bzf_era5_df['VPD'].values,'b-')
ax[1,0].grid(color='k')
ax[1,0].set_ylim(-20,20)
ax[1,0].yaxis.set_ticks((-20,-10,0,10,20))
ax[1,0].set_xlim(-5,15)
ax[1,0].xaxis.set_ticks((-5,0,5,10,15))
ax[1,0].plot([], label=bzf_text_vpd)
ax[1,0].legend(handlelength=0,labelcolor='b')

ax[1,1].plot(sod_flux_df['VPD'].values,sod_era5_df['VPD'].values,'bo',alpha=0.15)
ax[1,1].plot(sod_era5_df['VPD'].values,sod_b2+sod_m2*sod_era5_df['VPD'].values,'b-')
ax[1,1].grid(color='k')
ax[1,1].set_ylim(-20,20)
ax[1,1].yaxis.set_ticks((-20,-10,0,10,20))
ax[1,1].xaxis.set_ticks((-5,0,5,10,15))
ax[1,1].set_xlim(-5,15)
ax[1,1].xaxis.set_ticks((-5,0,5,10,15))
ax[1,1].plot([], label=sod_text_vpd)
ax[1,1].legend(handlelength=0,labelcolor='b')
ax[1,0].set_title(' ')
fig.text(-0.05, 1.15, varlabels[1]+' (%s)' %(VPD_units), horizontalalignment='center',
  verticalalignment='center', transform=ax[1,1].transAxes)

ax[2,0].plot(bzf_flux_df['SWR'].values,bzf_era5_df['SWR'].values,'ro',alpha=0.15)
ax[2,0].plot(bzf_era5_df['SWR'].values,bzf_b3+bzf_m3*bzf_era5_df['SWR'].values,'r-')
ax[2,0].grid(color='k')
ax[2,0].set_ylim(-200,200)
ax[2,0].yaxis.set_ticks((-200,-100,0,100,200))
ax[2,0].set_xlim(-200,200)
ax[2,0].xaxis.set_ticks((-200,-150,-100,-50,0,50,100,150,200))
ax[2,0].plot([], label=bzf_text_isr)
ax[2,0].legend(handlelength=0,labelcolor='r')

ax[2,1].plot(sod_flux_df['SWR'].values,sod_era5_df['SWR'].values,'ro',alpha=0.15)
ax[2,1].plot(sod_era5_df['SWR'].values,sod_b3+sod_m3*sod_era5_df['SWR'].values,'r-')
ax[2,1].grid(color='k')
ax[2,1].set_ylim(-200,200)
ax[2,1].yaxis.set_ticks((-200,-100,0,100,200))
ax[2,1].set_xlim(-200,200)
ax[2,1].xaxis.set_ticks((-200,-150,-100,-50,0,50,100,150,200))
ax[2,1].plot([], label=sod_text_isr)
ax[2,1].legend(handlelength=0,labelcolor='r')
ax[2,0].set_title(' ')
fig.text(-0.05, 1.15, varlabels[2]+' (%s)' %(SWR_units), horizontalalignment='center',
  verticalalignment='center', transform=ax[2,1].transAxes)

ax[3,0].plot(bzf_flux_df['TS'].values,bzf_era5_df['TS'].values,'ko',alpha=0.15)
ax[3,0].plot(bzf_era5_df['TS'].values,bzf_b4+bzf_m4*bzf_era5_df['TS'].values,'k-')
ax[3,0].grid(color='k')
ax[3,0].set_xlim(-15,15)
ax[3,0].xaxis.set_ticks((-15,-10,-5,0,5,10,15))
ax[3,0].plot([], label=bzf_text_ts)
ax[3,0].legend(handlelength=0,labelcolor='k')

ax[3,1].plot(sod_flux_df['TS'].values,sod_era5_df['TS'].values,'ko',alpha=0.15)
ax[3,1].plot(sod_era5_df['TS'].values,sod_b4+sod_m4*sod_era5_df['TS'].values,'k-')
ax[3,1].grid(color='k')
ax[3,1].set_xlim(-15,15)
ax[3,1].xaxis.set_ticks((-15,-10,-5,0,5,10,15))
ax[3,1].plot([], label=sod_text_ts)
ax[3,1].legend(handlelength=0,labelcolor='k')
ax[3,0].set_title(' ')
fig.text(-0.05, 1.15, varlabels[3]+' (%s)' %(TS_units), horizontalalignment='center',
  verticalalignment='center', transform=ax[3,1].transAxes)

ax[4,0].plot(bzf_flux_df['SWC'].values,bzf_era5_df['SWC'].values,'mo',alpha=0.15)
ax[4,0].plot(bzf_era5_df['SWC'].values,bzf_b5+bzf_m5*bzf_era5_df['SWC'].values,'m-')
ax[4,0].grid(color='k')
ax[4,0].set_xlim(-0.8,0.4)
ax[4,0].xaxis.set_ticks((-0.8,-0.6,-0.4,-0.2,0,0.2,0.4))
ax[4,0].plot([], label=bzf_text_swc)
ax[4,0].legend(handlelength=0,labelcolor='m')
ax[4,0].text(0.5,-0.5,"(a)",ha='center',transform=ax[4,0].transAxes)

ax[4,1].plot(sod_flux_df['SWC'].values,sod_era5_df['SWC'].values,'mo',alpha=0.15)
ax[4,1].plot(sod_era5_df['SWC'].values,sod_b5+sod_m5*sod_era5_df['SWC'].values,'m-')
ax[4,1].grid(color='k')
ax[4,1].set_xlim(-0.8,0.4)
ax[4,1].xaxis.set_ticks((-0.8,-0.6,-0.4,-0.2,0,0.2,0.4))
ax[4,1].plot([], label=sod_text_swc)
ax[4,1].legend(handlelength=0,labelcolor='m')
ax[4,1].text(0.5,-0.5,"(b)",ha='center',transform=ax[4,1].transAxes)
ax[4,0].set_title(' ')
fig.text(-0.05, 1.15, varlabels[4]+' (%s)' %(SWC_units), horizontalalignment='center',
  verticalalignment='center', transform=ax[4,1].transAxes)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Fluxnet/Ameriflux",labelpad=20)
plt.ylabel("ERA5",labelpad=20)
plt.tight_layout()
#fig.savefig('BZF_SOD_LINEAR_REGRESSION_COMPARISON_%s.png' %(today_date),dpi=200,bbox_inches='tight')
fig.savefig('BZF_SOD_LINEAR_REGRESSION_COMPARISON_%s.pdf' %(today_date),format='pdf',bbox_inches='tight')
