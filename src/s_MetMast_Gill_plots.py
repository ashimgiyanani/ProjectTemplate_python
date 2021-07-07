# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:11:00 2020

@author: papalk
"""


import sys
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.dates import DateFormatter
from matplotlib.ticker import StrMethodFormatter
import pandas as pd
import numpy as np
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn-whitegrid')
SMALL_SIZE = 17
MEDIUM_SIZE = 22
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE,weight = 'bold')          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure', figsize =  (8, 8))


def addtimedecimal(l):
    for i,x in enumerate(l):    
        if '.' not in x:
            l[i] = x+'.00'
        else:
            l[i] = x
    return(l)
#%% Import data gill

# Set path
path  = r'Z:\Projekte\109797-TestfeldBHV\30_Technical_execution_Confidential\TP3\AP2_Aufbau_Infrastruktur\Infrastruktur_Windmessung\02_Equipment\01_Wartung Messmast GE-NET_DWG_20190226\Maintenance\MetMast Upgrade\Data\ASCII'

# Select device
device = 'gill'

# Select start and end date
dt_start ='2021-01-29 00:00:00' # Select start date in the form yyyy-mm-dd_HH-MM-SS

dt_end = dt.datetime.strptime(dt_start, '%Y-%m-%d %H:%M:%S')  + dt.timedelta(days=10)# Select end date in the form yyyy-mm-dd_HH-MM-SS
dt_end = dt_end.strftime('%Y-%m-%d %H:%M:%S')

#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(0, r'C:\Users\papalk\Desktop\04_Python\Testfeld')
# import data
from f_import_gill_thies_data import f_import_gill_thies_data
df = f_import_gill_thies_data(path,device,dt_start,dt_end) 

#Correct time lag
df.index = [date+dt.timedelta(seconds=0) for date in df.index] # Logger time unit: UTC+0 after 14/01/2021 11:00, if before seconds=-7200
#%%
# Resample files gill
dfr = df.resample('1H').mean() # Resample in hourly averages mean(skipna = True)
dfs = df.resample('1H').count()/72000*100 # Resample and count in hourly averages
df10 = df.resample('600S').mean() # Resample in 10 min averages mean(skipna = True)
df10s = df.resample('600S').count()/12000*100 # Resample and count in 10min averages

#%% Calculate Vhor & Dir

#55m
Vhor55 = np.sqrt(df.gill_55_u**2+df.gill_55_v**2).resample('600S').mean()
Vhor55_std =  np.sqrt(df.gill_55_u**2+df.gill_55_v**2).resample('600S').std()
Dir55 = np.mod(90-np.mod(np.arctan2(df.gill_55_v,df.gill_55_u)*180/np.pi,360).resample('600S').mean()+30.5,360)
Dir55_std = np.mod(90-np.mod(np.arctan2(df.gill_55_v,df.gill_55_u)*180/np.pi,360).resample('600S').std()+30.5,360)
#110m
Vhor110 = np.sqrt(df.gill_115_u**2+df.gill_115_v**2).resample('600S').mean()
Vhor110_std =  np.sqrt(df.gill_115_u**2+df.gill_115_v**2).resample('600S').std()
Dir110 = np.mod(90-np.mod(np.arctan2(df.gill_115_v,df.gill_115_u)*180/np.pi,360).resample('600S').mean()+30.6,360)
Dir110_std = np.mod(90-np.mod(np.arctan2(df.gill_115_v,df.gill_115_u)*180/np.pi,360).resample('600S').std()+30.6,360)

#%% Plots
# # raw gill
# for i in [1,3,4,5,7,8,11,12,13,15,16]:
#     # date-ws_free_avg
#     fig = plt.figure(figsize =  (20, 8)) 
#     ax = fig.add_subplot(111)
#     # ax.plot(df.ix[df['LOS index']==LOS,i],'.',label = 'Valid data');
#     ax.plot(df.TIMESTAMP[(df.gill_55_SpeedOfSound >100)&(df.gill_115_SpeedOfSound>100)],df.ix[(df.gill_55_SpeedOfSound >100)&(df.gill_115_SpeedOfSound>100),i],'.');
#     ax.set_xlabel('time ',labelpad=40,weight= 'bold')
#     ax.set_ylabel(df.columns[i],labelpad=40,weight= 'bold')
#     date_form = DateFormatter("%H:%M:%S")
#     ax.xaxis.set_major_formatter(date_form)
#     legend = ax.legend(frameon = 1)
#     frame = legend.get_frame()
#     frame.set_color('white')
#     plt.savefig(r'Z:\Projekte\109797-TestfeldBHV\30_Technical_execution_Confidential\TP3\AP2_Aufbau_Infrastruktur\Infrastruktur_Windmessung\02_Equipment\01_Wartung Messmast GE-NET_DWG_20190226\MetMast Upgrade\Data\plots\TS' + path[-20:-4]+'_'+df.columns[i]+'.png',bbox_inches='tight')


# # Hourly status
# for i in np.arange(len(dfr.columns)):
#     # date-ws_free_avg
#     fig = plt.figure(figsize =  (20, 8)) 
#     ax = fig.add_subplot(111)
#     ax.plot(dfr.index[(dfr.gill_55_SpeedOfSound >100)&(dfr.gill_115_SpeedOfSound>100)],dfr.ix[(dfr.gill_55_SpeedOfSound >100)&(dfr.gill_115_SpeedOfSound>100),i],'.');
#     ax.set_xlabel('time ',labelpad=40,weight= 'bold')
#     ax.set_ylabel(dfr.columns[i],labelpad=40,weight= 'bold')
#     date_form = DateFormatter("%H:%M:%S")
#     ax.xaxis.set_major_formatter(date_form)
#     legend = ax.legend(frameon = 1)
#     frame = legend.get_frame()
#     frame.set_color('white')
#     plt.savefig(r'Z:\Projekte\109797-TestfeldBHV\30_Technical_execution_Confidential\TP3\AP2_Aufbau_Infrastruktur\Infrastruktur_Windmessung\02_Equipment\01_Wartung Messmast GE-NET_DWG_20190226\MetMast Upgrade\Data\plots\TS_1H_2020-09-06_'+df.columns[i]+'.png',bbox_inches='tight')


#%% QM plots
fig,ax = plt.subplots(5,1, figsize =  (10, 15),sharex=True) 
ax[0].plot(df10.index[(df10.index>dt_start)&(df10.index<dt_end)],df10['gill_115_u'][(df10.index>dt_start)&(df10.index<dt_end)].values,'.');
ax[0].set_xlabel('date',labelpad=40,weight= 'bold')
ax[0].set_ylabel("u [m/s]",labelpad=40,weight= 'bold')
date_form = DateFormatter("%d/%m")
ax[0].xaxis.set_major_formatter(date_form)
ax[0].set_xlim([ dt.datetime.strptime(dt_start, '%Y-%m-%d %H:%M:%S'), dt.datetime.strptime(dt_end, '%Y-%m-%d %H:%M:%S')])
ax[0].set_ylim([-25,25])

ax[1].plot(df10.index[(df10.index>dt_start)&(df10.index<dt_end)],df10['gill_115_v'][(df10.index>dt_start)&(df10.index<dt_end)].values,'.');
ax[1].set_xlabel('date',labelpad=40,weight= 'bold')
ax[1].set_ylabel("v [m/s]",labelpad=40,weight= 'bold')
date_form = DateFormatter("%d/%m")
ax[1].xaxis.set_major_formatter(date_form)
ax[1].set_xlim([ dt.datetime.strptime(dt_start, '%Y-%m-%d %H:%M:%S'), dt.datetime.strptime(dt_end, '%Y-%m-%d %H:%M:%S')])
ax[1].set_ylim([-25,25])

ax[2].plot(df10.index[(df10.index>dt_start)&(df10.index<dt_end)],df10['gill_115_w'][(df10.index>dt_start)&(df10.index<dt_end)].values,'.');
ax[2].set_xlabel('date',labelpad=40,weight= 'bold')
ax[2].set_ylabel("w [m/s]",labelpad=40,weight= 'bold')
date_form = DateFormatter("%d/%m")
ax[2].xaxis.set_major_formatter(date_form)
ax[2].set_xlim([ dt.datetime.strptime(dt_start, '%Y-%m-%d %H:%M:%S'), dt.datetime.strptime(dt_end, '%Y-%m-%d %H:%M:%S')])
ax[2].set_ylim([-5,5])

ax[3].plot(df10.index[(df10.index>dt_start)&(df10.index<dt_end)],df10['gill_115_SonicTempC'][(df10.index>dt_start)&(df10.index<dt_end)].values,'.');
ax[3].set_xlabel('date',labelpad=40,weight= 'bold')
ax[3].set_ylabel("Temp [$^o$C]",labelpad=40,weight= 'bold')
date_form = DateFormatter("%d/%m")
ax[3].xaxis.set_major_formatter(date_form)
ax[3].set_xlim([ dt.datetime.strptime(dt_start, '%Y-%m-%d %H:%M:%S'), dt.datetime.strptime(dt_end, '%Y-%m-%d %H:%M:%S')])
ax[3].set_ylim([-10,30])

ax[4].plot(df10s.index[(df10s.index>dt_start)&(df10s.index<dt_end)],df10s['RECORD'][(df10s.index>dt_start)&(df10s.index<dt_end)].values,'.',color = 'limegreen');
ax[4].plot(df10s.index[(df10s.index>dt_start)&(df10s.index<dt_end)&(df10s['RECORD']<100)],df10s['RECORD'][(df10s.index>dt_start)&(df10s.index<dt_end)&(df10s['RECORD']<100)].values,'.',color = 'red');
ax[4].set_xlabel('date',labelpad=40,weight= 'bold')
ax[4].set_ylabel("Availability [%]",labelpad=40,weight= 'bold')
date_form = DateFormatter("%d/%m")
ax[4].xaxis.set_major_formatter(date_form)
ax[4].set_xlim([ dt.datetime.strptime(dt_start, '%Y-%m-%d %H:%M:%S'), dt.datetime.strptime(dt_end, '%Y-%m-%d %H:%M:%S')])
ax[4].set_ylim([0,100])
plt.savefig(r'Z:\Projekte\109797-TestfeldBHV\30_Technical_execution_Confidential\TP3\AP2_Aufbau_Infrastruktur\Infrastruktur_Windmessung\02_Equipment\01_Wartung Messmast GE-NET_DWG_20190226\Maintenance\MetMast Upgrade\Data\\'+dt_start[0:10]+'_'+dt_end[0:10]+'_115_gill.png',
            bbox_inches='tight',dpi = 100)


# QM plots
fig,ax = plt.subplots(5,1, figsize =  (10, 15),sharex=True) 
ax[0].plot(df10.index[(df10.index>dt_start)&(df10.index<dt_end)],df10['gill_55_u'][(df10.index>dt_start)&(df10.index<dt_end)].values,'.');
ax[0].set_xlabel('date',labelpad=40,weight= 'bold')
ax[0].set_ylabel("u [m/s]",labelpad=40,weight= 'bold')
date_form = DateFormatter("%d/%m")
ax[0].xaxis.set_major_formatter(date_form)
ax[0].set_xlim([ dt.datetime.strptime(dt_start, '%Y-%m-%d %H:%M:%S'), dt.datetime.strptime(dt_end, '%Y-%m-%d %H:%M:%S')])
ax[0].set_ylim([-25,25])

ax[1].plot(df10.index[(df10.index>dt_start)&(df10.index<dt_end)],df10['gill_55_v'][(df10.index>dt_start)&(df10.index<dt_end)].values,'.');
ax[1].set_xlabel('date',labelpad=40,weight= 'bold')
ax[1].set_ylabel("v [m/s]",labelpad=40,weight= 'bold')
date_form = DateFormatter("%d/%m")
ax[1].xaxis.set_major_formatter(date_form)
ax[1].set_xlim([ dt.datetime.strptime(dt_start, '%Y-%m-%d %H:%M:%S'), dt.datetime.strptime(dt_end, '%Y-%m-%d %H:%M:%S')])
ax[1].set_ylim([-25,25])

ax[2].plot(df10.index[(df10.index>dt_start)&(df10.index<dt_end)],df10['gill_55_w'][(df10.index>dt_start)&(df10.index<dt_end)].values,'.');
ax[2].set_xlabel('date',labelpad=40,weight= 'bold')
ax[2].set_ylabel("w [m/s]",labelpad=40,weight= 'bold')
date_form = DateFormatter("%d/%m")
ax[2].xaxis.set_major_formatter(date_form)
ax[2].set_xlim([ dt.datetime.strptime(dt_start, '%Y-%m-%d %H:%M:%S'), dt.datetime.strptime(dt_end, '%Y-%m-%d %H:%M:%S')])
ax[2].set_ylim([-5,5])

ax[3].plot(df10.index[(df10.index>dt_start)&(df10.index<dt_end)],df10['gill_55_SonicTempC'][(df10.index>dt_start)&(df10.index<dt_end)].values,'.');
ax[3].set_xlabel('date',labelpad=40,weight= 'bold')
ax[3].set_ylabel("Temp [$^o$C]",labelpad=40,weight= 'bold')
date_form = DateFormatter("%d/%m")
ax[3].xaxis.set_major_formatter(date_form)
ax[3].set_xlim([ dt.datetime.strptime(dt_start, '%Y-%m-%d %H:%M:%S'), dt.datetime.strptime(dt_end, '%Y-%m-%d %H:%M:%S')])
ax[3].set_ylim([-10,30])

ax[4].plot(df10s.index[(df10s.index>dt_start)&(df10s.index<dt_end)],df10s['RECORD'][(df10s.index>dt_start)&(df10s.index<dt_end)].values,'.',color = 'limegreen');
ax[4].plot(df10s.index[(df10s.index>dt_start)&(df10s.index<dt_end)&(df10s['RECORD']<100)],df10s['RECORD'][(df10s.index>dt_start)&(df10s.index<dt_end)&(df10s['RECORD']<100)].values,'.',color = 'red');
ax[4].set_xlabel('date',labelpad=40,weight= 'bold')
ax[4].set_ylabel("Availability [%]",labelpad=40,weight= 'bold')
date_form = DateFormatter("%d/%m")
ax[4].xaxis.set_major_formatter(date_form)
ax[4].set_xlim([ dt.datetime.strptime(dt_start, '%Y-%m-%d %H:%M:%S'), dt.datetime.strptime(dt_end, '%Y-%m-%d %H:%M:%S')])
ax[4].set_ylim([0,100])
plt.savefig(r'Z:\Projekte\109797-TestfeldBHV\30_Technical_execution_Confidential\TP3\AP2_Aufbau_Infrastruktur\Infrastruktur_Windmessung\02_Equipment\01_Wartung Messmast GE-NET_DWG_20190226\Maintenance\MetMast Upgrade\Data\\'+dt_start[0:10]+'_'+dt_end[0:10]+'_55_gill.png',
            bbox_inches='tight',dpi = 100)

#%% Compare MM
    
# from class_mm import C_MMprocess
# start = C_MMprocess(r'E:\113166_Testfeld\01_Instruments\03_MetMast\OneDAS_2021-01-04T00-00_600_s_063b14a8.zip','600S')
# data_mm = start.loadzip()

# # plot Vhor
# fig = plt.figure(figsize =  (20, 8)) 
# plt.plot(Vhor110,label = 'USA3D Gill 110m')
# plt.plot(Vhor55,label = 'USA3D Gill 55m')
# plt.plot(Vhor,label = 'USA3D Thies 25m')
# plt.plot(data_mm.M0000_V1,label = 'Cup Thies 115m')
# plt.plot(data_mm.M0030_V4,label = 'Cup Thies 55m')
# plt.plot(data_mm.M0040_V5,label = 'Cup Thies 25m')
# plt.ylim(0,20)
# plt.xlabel('date',labelpad=10,weight= 'bold')
# plt.ylabel('$V_{hor}$ [m/s]',labelpad=10,weight= 'bold')
# plt.legend()


# # plot Dir
# fig = plt.figure(figsize =  (20, 8)) 
# plt.plot(Dir110,label = 'USA3D Gill 110m')
# plt.plot(Dir55,label = 'USA3D Gill 55m')
# plt.plot(Dir,label = 'USA3D Thies 25m')
# plt.plot(data_mm.M0070_D1,label = 'Vane Thies 110m')
# plt.plot(data_mm.M0100_D4,label = 'Vane Thies 23.5m')
# plt.ylim(0,360)
# plt.xlabel('date',labelpad=10,weight= 'bold')
# plt.ylabel('$V_{hor}$ [m/s]',labelpad=10,weight= 'bold')
# plt.legend()

# # # date-ws_free_avg
# # fig = plt.figure(figsize =  (20, 8)) 
# # ax = fig.add_subplot(111)
# # ax.plot(df10.index[(df10.gill_55_SpeedOfSound >100)&(df10.gill_115_SpeedOfSound>100)],df10.ix[(df10.gill_55_SpeedOfSound >100)&(df10.gill_115_SpeedOfSound>100),1],'.',label = 'Gill110');
# # ax.plot(data_mm.index,data_mm.M0000_V1, '.',label = 'Cup115');
# # ax.set_xlabel('time ',labelpad=40,weight= 'bold')
# # ax.set_ylabel(["m/s"],labelpad=40,weight= 'bold')
# # date_form = DateFormatter("%H:%M:%S")
# # ax.xaxis.set_major_formatter(date_form)
# # legend = ax.legend(frameon = 1)
# # frame = legend.get_frame()
# # frame.set_color('white')
