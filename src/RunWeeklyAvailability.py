# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 19:41:21 2021

@author: giyash
"""
# Steps to include in the script:
# import all the channels from OneDAS [##### 100%]
# arrange them into one database and one datetime format [##### 100%]
# check if the timestamps are matching [0%]
# perform a standardized check using a function definition [##### 100%]
# perform an advanced check using a function definition [##### 100%]
# plot the results for each sensor [##### 100%]
# write the results to an excel file which serves as a logbook for all the sensors [##### 100%]
# save the pdf report generated on the Z drive [##### 100%]
# send the report to all recipents [0%]
 
#%% user modules
import pandas as pd
import glob
import os
import numpy as np
import sys
sys.path.append(r"../../userModules")
# sys.path.append(r"../fun") # change to this when sharing the data
sys.path.append(r"../../OneDasExplorer/Python Connector")

import runpy as rp
import matplotlib.pyplot as plt

from csv import writer
import matlab2py as m2p
from pythonAssist import *
from datetime import datetime, timezone, timedelta

from odc_exportData import odc_exportData
from FnImportOneDas import FnImportOneDas
from FnDataAvailability import FnDataAvailability

#%% user definitions
input = struct()
input.tiny = 12
input.Small = 14
input.Medium = 16
input.Large = 18
input.Huge = 22
plt.rc('font', size=input.Small)          # controls default text sizes
plt.rc('axes', titlesize=input.Small)     # fontsize of the axes title
plt.rc('axes', labelsize=input.Medium)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=input.Small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=input.Small)    # fontsize of the tick labels
plt.rc('legend', fontsize=input.Medium)    # legend fontsize
plt.rc('figure', titlesize=input.Huge)  # fontsize of the figure title

#%% import data from OneDAS
# begin = datetime(2021, 9, 6, 0, 0, tzinfo=timezone.utc)
# end   = datetime(2021, 9, 13, 0, 0, tzinfo=timezone.utc)
data = struct()
data.sampleRate = 1/600
input.AppendLog = 0

# must all be of the same sample rate
data.paths = [
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0000_V1/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/P0420_RotorSpdRaw/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/P1000_TotalPwMeas/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0010_V2/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0020_V3/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0030_V4/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0040_V5/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0050_V6/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0060_Precipitation/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0060_Precipitation/600 s_sum',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0230_H1/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0070_D1/600 s_mean_polar',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0100_D4/600 s_mean_polar',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0110_D5/600 s_mean_polar',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0200_B1/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0210_B2/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0220_T1/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0240_T2/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/ISPIN/SaDataValid/600 s',
    '/AIRPORT/AD8_PROTOTYPE/ISPIN/WS_free_avg/600 s',
    '/AIRPORT/AD8_PROTOTYPE/ISPIN/DataOK/600 s',
    '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/BlackTC_110m_HWS_hub/600 s',
    '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/BluePO_110m_HWS_hub/600 s',
    '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/GreenPO_110m_HWS_hub/600 s',
    '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/BlackTC_110m_HWS_hub_availability/600 s',
    '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/BluePO_110m_HWS_hub_availability/600 s',
    '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/GreenPO_110m_HWS_hub_availability/600 s',
    '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_115m_Wind_Speed/600 s'
                ]
# Provide the names that you want to give to the channel paths
data.names = [
        'v1',
        'omega',
        'Pw',
        'v2',
        'v3',
        'v4',
        'v5',
        'v6',
        'prec',
        'prec_sum',
        'RH',
        'd1',
        'd4',
        'd5',
        'b1',
        'b2',
        'T1',
        'T2',
        's_valid',
        's_V', 
        's_ok',
        'btc_v110',
        'bpo_v110', 
        'gpo_v110', 
        'btc_Av110', 
        'bpo_Av110', 
        'gpo_Av110', 
        'wc_v115'
        ]

# folder where data will be stored 
data.folder = r"../data"
# start and end datetime for data download
data.tstart = datetime.strptime('2021-10-18_00-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
# funktioniert
data.tend = data.tstart + timedelta(days=7) # Select start date in the form yyyy-mm-dd_HH-MM-SS
_, pdData, t = FnImportOneDas(data.tstart, data.tend, data.paths, data.names, data.sampleRate, data.folder)
# create a pandas dataframe
index = pd.date_range(data.tstart, periods=7, freq='D')
data.sensors = ['wt', 'ispin', 'btc', 'bpo', 'gpo', 'metmast', 'sonics', 'windcube']

# weekly availability variable for all sensors
avail = struct()
avail.weekly = pd.DataFrame(index=index, columns=data.sensors)
avail.weekly = avail.weekly.fillna(0)
# filtering condition dataframe for all sensors
avail.filter = pd.DataFrame(index=pdData.index, columns=data.sensors)
avail.filter = avail.filter.fillna(0)
# weekly statistics
avail.stat = pd.DataFrame(index=['Nvalid', 'Npts', 'N_avail'], columns=['Date', *data.sensors])
avail.stat['Date'] = [data.tend, data.tend, data.tend]
#%% Check the availability of the wind turbine
param = ['omega', 'Pw']
param_min = [0,0]
param_max = [30,8000]

# special conditions for filtering
cond1 = pd.DataFrame(index=pdData.index, columns=param)
cond1[param[0]] = (pdData[param[0]]) > 1 # allow only omega > 1 rpm
cond1[param[1]] = (pdData[param[1]]) >= 0 # # allow a power >0 kW

# loop through param to give Availability, filtering condition (combined) and statistics
for i in range(len(param)):
    data.xrange=[param_min[i], param_max[i]]
    data.sampleRate=1/600
    av, cond, Nstat = FnDataAvailability(data.sampleRate, param[i], pdData, data.xrange, cond2=cond1[param[i]])
    avail.weekly[param[i]] = av
    avail.filter[param[i]] = cond
    avail.stat[param[i]] = Nstat.transpose()
    del av, cond, Nstat

# assigning local variables to global
input.threshold = 0.25 # availability threshold of 25%
avail.weekly['wt'] = [int(round(x-input.threshold + 0.5)) for x in avail.weekly.loc[:, param].mean(axis=1)]
avail.filter['wt'] = [x for x in avail.filter.loc[:, param].all(axis=1)]
del cond1, param, param_min, param_max

#%% Check the availability of iSpin
## Filtering conditions
cond0 = pdData.s_V.notnull() # non-zero values
cond1 = (pdData.t>=data.tstart)&(pdData.t<=data.tend) # time interval filtering
cond2 = (pdData.s_valid==1) # 95% availability of 10 Hz data, wind vector +-90° in front of turbine within 10 min 
cond3 = (pdData.s_ok==True)  # data.TotalCountNoRotation=0, 95% availability, min & max rotor rpm, avg rotor rpm, free wind speed > 3.5 m/s, sample ID!= 0
cond4 = (pdData.s_V > 0) & (pdData.s_V < 50) # wind speed physical limits

## extra parameters for logbook [# to be integrated]
Nwt_valid = pdData.loc[(cond0 & cond1 & cond2 & cond3),'s_V'].shape[0]
Nyaw_valid = pdData.loc[(cond0 & cond1 & cond3),'s_V'].shape[0]

# filling in the weekly availabiliy as 1/0 based on number of points
param= 's_V'
param_min, param_max = 0, 50
data.xrange=[param_min, param_max]
av, cond, Nstat = FnDataAvailability(data.sampleRate, param, pdData, data.xrange, cond2=cond2)

avail.weekly['ispin'] = av
avail.filter['ispin'] = cond
avail.stat['ispin'] = Nstat.transpose()
del av, cond, Nstat
del cond0, cond1, cond2, cond3, cond4, param, param_min, param_max

#%% check the availability of Nacelle Lidars

# list of BlackTC, BluePO and GreenPO Nacelle Lidars
param = ['btc', 'bpo', 'gpo']
param_min, param_max = 0, 50

# loop through param to give Availability, filtering condition (combined) and statistics
for i in range(len(param)):
    av, cond = [], []
    param_v = param[i] + '_v110'
    param_Av = param[i] + '_Av110'
    ## Filtering conditions
    # Check if the Lidars have Availability > 25% within the 10 minutes
    nl_cond3 = (pdData[param_Av] >= input.threshold)
    # filling in the weekly availabiliy as 1/0 based on number of points
    data.xrange = [param_min, param_max]
    av, cond, Nstat = FnDataAvailability(data.sampleRate, param_v, pdData, data.xrange, cond3=nl_cond3)

    avail.weekly[param[i]] = av
    avail.filter[param[i]] = cond
    avail.stat[param[i]] = Nstat.transpose()
    del av, cond, Nstat

# delete useless variables
del nl_cond3, param, param_min, param_max, param_v, param_Av

#%% check the availability of the metmast sensors
param = ['v1','v2','v3','v4','v5','v6','prec','prec_sum','d1','d4','d5','b1','b2','T1']
param_min = [0,0,0,0,0,0,-0.1,0,0,0,0,0,0,-60]
param_max = [50,50,50,50,50,50,100,100,360,360,360,2000, 2000, 60]
data.xrange=[[param_min[i],param_max[i]] for i in range(len(param_min))]

# loop through param to give Availability, filtering condition (combined) and statistics
for i in range(len(param)):
    av, cond=[], []
    # Filtering conditions
    cond_sp = np.isfinite(pdData[param[i]])
    av, cond, Nstat = FnDataAvailability(data.sampleRate, param[i], pdData, data.xrange[i], cond0=cond_sp)
        
    avail.weekly[param[i]] = av
    avail.filter[param[i]] = cond
    avail.stat[param[i]] = Nstat.transpose()
    del av, cond, Nstat

# Combining the individual sensor availability in one variable
avail.weekly['metmast'] = [int(round(x-input.threshold + 0.5)) for x in avail.weekly.loc[:, param].mean(axis=1)]
avail.filter['metmast'] = [x for x in avail.filter.loc[:, param].all(axis=1)]
# delete useless variables
del cond_sp, param, param_min, param_max

#%% Check the availability of WindCube
param='wc_v115'
param_min, param_max = 0, 50
data.xrange=[param_min, param_max]

av, cond, Nstat = FnDataAvailability(data.sampleRate, param, pdData, data.xrange)

avail.weekly['windcube'] = av
avail.filter['windcube'] = cond
avail.stat['windcube'] = Nstat.transpose()
del av, cond, Nstat

#%% check the availability of sonic anemometers
data.sonics = struct()
data.sonics.paths =     [
    '/AIRPORT/AD8_PROTOTYPE/METMAST_EXTENSION/gill_115_u/20 Hz',
    '/AIRPORT/AD8_PROTOTYPE/METMAST_EXTENSION/gill_55_u/20 Hz',
    '/AIRPORT/AD8_PROTOTYPE/METMAST_EXTENSION/thies_25_Vx/20 Hz',
    ]
data.sonics.names = [
        'gill_u115',
        'gill_u55',
        'thies_u25',
            ]
data.sonics.sampleRate = 20
_, pdData_sonics, t_sonics = FnImportOneDas(data.tstart, data.tend, data.sonics.paths, data.sonics.names, data.sonics.sampleRate, data.folder)
param = data.sonics.names
param_min, param_max = -65, 65
data.xrange = [param_min, param_max]

# calculate 10 min averages
import more_itertools
for i in range(len(param)):
    step= int(data.sonics.sampleRate * 600)  # advance of the window
    length = int(data.sonics.sampleRate * 600) # width of the window
    N_win = np.int64(len(pdData_sonics.loc[:,param[i]])/step)
    window = np.transpose(list(more_itertools.windowed(np.ravel(pdData_sonics.loc[:,param[i]]), n=length, \
                        fillvalue=np.nan, step=step)))
    pdData[param[i]] = np.nanmean(window, axis=0)

for i in range(len(param)):
    av, cond, Nstat = FnDataAvailability(data.sampleRate, param[i], pdData, data.xrange)
    avail.weekly[param[i]] = av
    avail.filter[param[i]] = cond
    avail.stat[param[i]] = Nstat.transpose()
    del av, cond, Nstat

# averaging the gill and thies availability into one column 'sonics'
avail.weekly['sonics'] = [int(round(x-input.threshold + 0.5)) for x in avail.weekly.loc[:, param].mean(axis=1)]
avail.filter['sonics'] = [x for x in avail.filter.loc[:, param].all(axis=1)]

# delete useless variables
del param, param_min, param_max, step, length, N_win, window

#%% download data from DWD portal
from wetterdienst.provider.dwd.observation import DwdObservationRequest
import datetime as dt

request = DwdObservationRequest(
    parameter=["wind_speed", "wind_direction", "precipitation_duration", "precipitation_height", "precipitation_indicator_wr" ],
    resolution = "minute_10",
    start_date= (data.tstart + dt.timedelta(minutes = 10)).strftime("%Y-%m-%d %H:%M"),
    end_date = data.tend.strftime("%Y-%m-%d %H:%M"),
    tidy=False,
    humanize=False,
    si_units=True).filter_by_station_id(station_id=(701))

station_data = request.values.all().df
new_df = station_data.groupby(station_data['parameter'], observed=True)
param = station_data.parameter.unique()
param_abbr = ['ws', 'wd', 'prec_T', 'prec_H', 'prec_idx']
for i in range(len(param)):
    pdData['dwd_' + param_abbr[i]] = station_data.value[station_data.parameter== param[i]].values

# delete useless variables
del request, station_data, new_df, param, param_abbr

#%% Timestamp quaylity check [to be implemented]
# from xcorr import xcorr
# lags,c = xcorr(pdData.v1, pdData.btc_v110, normed=False, detrend=False, maxlags=len(pdData.v1)-1)
# plt.plot(lags, c)

#%% plot the figures to be input into the pdf report
from matplotlib.dates import DateFormatter
from matplotlib.ticker import StrMethodFormatter
fig,ax = plt.subplots(5,1, figsize =  (10, 16),sharex=True)

ax[0].plot(pdData.t[avail.filter['metmast']] , pdData.d1[avail.filter['metmast']] ,'.',label = 'd1')
ax[0].plot(pdData.t[avail.filter['metmast']] , pdData.d4[avail.filter['metmast']] ,'.',label = 'd4')
ax[0].plot(pdData.t[avail.filter['metmast']] , pdData.d5[avail.filter['metmast']] ,'.',label = 'd5')
ax[0].plot(pdData.t , pdData.dwd_wd ,'.',label = 'DWD BHV 10m')
ax[0].set_xlabel('date')
ax[0].set_ylabel("wind direction [°]")
date_form = DateFormatter("%d/%m")
ax[0].xaxis.set_major_formatter(date_form)
ax[0].set_xlim([ datetime.strptime(str(data.tstart), '%Y-%m-%d %H:%M:%S'), datetime.strptime(str(data.tend), '%Y-%m-%d %H:%M:%S')])
ax[0].set_ylim([-2,360])
ax[0].legend(loc=1)
ax[0].grid(axis = 'x', color='0.95')

# plot the lines
p1 = ax[1].plot(pdData.t[avail.filter['wt']] , pdData.omega[avail.filter['wt']] ,'k.',label = 'omega')
ax[1].set_xlabel('date')
ax[1].set_ylabel("AD8 omega [rpm]")
tw=ax[1].twinx()
p2 = tw.plot(pdData.t[avail.filter['wt']] , pdData.Pw[avail.filter['wt']] ,'g.',label = 'AD8-180 Pw')
tw.set_ylabel("AD8 Power [W]")
# arrange the legends together
p = p1+p2
labs = [l.get_label() for l in p]
color = [l.get_color() for l in  p]
ax[1].legend(p, labs, loc=1)
# axis properties
tw.yaxis.get_label().set_color(color[1])
ax[1].yaxis.get_label().set_color(color[0])
date_form = DateFormatter("%d/%m/%y")
ax[1].xaxis.set_major_formatter(date_form)
ax[1].set_xlim([ datetime.strptime(str(data.tstart), '%Y-%m-%d %H:%M:%S'), datetime.strptime(str(data.tend), '%Y-%m-%d %H:%M:%S')])
ax[1].set_ylim([-2,25])
ax[1].grid(axis = 'x', color='0.95')

ax[2].plot(pdData.t[avail.filter['metmast']] , pdData.T1[avail.filter['metmast']] ,'k.',label = 'Temperature sensor 1')
ax[2].plot(pdData.t[avail.filter['metmast']] , pdData.T2[avail.filter['metmast']] - 1 ,'g.',label = 'Temperature sensor 2')
ax[2].set_xlabel('date')
ax[2].set_ylabel("Temperature [°C]")
date_form = DateFormatter("%d/%m/%y")
ax[2].xaxis.set_major_formatter(date_form)
ax[2].set_xlim([ datetime.strptime(str(data.tstart), '%Y-%m-%d %H:%M:%S'), datetime.strptime(str(data.tend), '%Y-%m-%d %H:%M:%S')])
ax[2].set_ylim([-10,40])
ax[2].legend(loc=1)
ax[2].grid(axis = 'x', color='0.95')

p1a = ax[3].plot(pdData.t , pdData.dwd_prec_H ,'r.',label = 'precipitation DWD')
p1b = ax[3].plot(pdData.t, pdData.prec_sum, 'b.', label='prec_sum')
# p1c = ax[3].plot(pdData.t[avail.filter['metmast']] , pdData.prec[avail.filter['metmast']]*1000 ,'k.',label = 'precipitation')
ax[3].set_xlabel('date')
ax[3].set_ylabel("precipitation [mm]")
# arrange the legends together
p = p1a+p1b
labs = [l.get_label() for l in p]
color = [l.get_color() for l in  p]
ax[3].legend(p, labs, ncol=3)
date_form = DateFormatter("%d/%m/%y")
ax[3].xaxis.set_major_formatter(date_form)
ax[3].set_xlim([ datetime.strptime(str(data.tstart), '%Y-%m-%d %H:%M:%S'), datetime.strptime(str(data.tend), '%Y-%m-%d %H:%M:%S')])
ax[3].set_ylim([-1,3])
ax[3].legend(loc=1, ncol=3, mode='expand')
tw.legend(loc=1)
ax[3].grid(axis = 'x', color='0.95')

ax[4].plot(pdData.t[avail.filter['metmast']] , pdData.b1[avail.filter['metmast']] ,'k.',label = 'Barometer 1')
ax[4].plot(pdData.t[avail.filter['metmast']] , pdData.b2[avail.filter['metmast']] - 1 ,'g.',label = 'Barometer 2')
ax[4].set_xlabel('date')
ax[4].set_ylabel("Pressure [Pa]")
date_form = DateFormatter("%d/%m/%y")
ax[4].xaxis.set_major_formatter(date_form)
ax[4].set_xlim([ datetime.strptime(str(data.tstart), '%Y-%m-%d %H:%M:%S'), datetime.strptime(str(data.tend), '%Y-%m-%d %H:%M:%S')])
ax[4].set_ylim([900,1100])
ax[4].legend(loc=1)
ax[4].grid(axis = 'x', color='0.95')
plt.xticks(rotation=30)

fig.savefig("../results/results1.png", format='png')


#%% New plot with sensors
fig,ax = plt.subplots(5,1, figsize =  (10, 16),sharex=True)
ax[0].plot(pdData.t[avail.filter['windcube']] , pdData.wc_v115[avail.filter['windcube']] ,'.',label = 'WLS7-119 WC')
ax[0].set_xlabel('date')
ax[0].set_ylabel("wind speed [m/s]")
date_form = DateFormatter("%d/%m/%y")
ax[0].xaxis.set_major_formatter(date_form)
ax[0].set_xlim([ datetime.strptime(str(data.tstart), '%Y-%m-%d %H:%M:%S'), datetime.strptime(str(data.tend), '%Y-%m-%d %H:%M:%S')])
ax[0].set_ylim([-2,25])
ax[0].legend(loc=1)
ax[0].grid(axis = 'x', color='0.95')

ax[1].plot(pdData.t[avail.filter['metmast']] , pdData.v1[avail.filter['metmast']] ,'.',label = 'v1')
ax[1].plot(pdData.t[avail.filter['metmast']] , pdData.v2[avail.filter['metmast']] ,'.',label = 'v2')
ax[1].plot(pdData.t[avail.filter['metmast']] , pdData.v3[avail.filter['metmast']] ,'.',label = 'v3')
ax[1].plot(pdData.t[avail.filter['metmast']] , pdData.v4[avail.filter['metmast']] ,'.',label = 'v4')
ax[1].plot(pdData.t[avail.filter['metmast']] , pdData.v5[avail.filter['metmast']] ,'.',label = 'v5')
ax[1].plot(pdData.t[avail.filter['metmast']] , pdData.v6[avail.filter['metmast']] ,'.',label = 'v6')
ax[1].set_xlabel('date')
ax[1].set_ylabel("wind speed [m/s]")
date_form = DateFormatter("%d/%m/%y")
ax[1].xaxis.set_major_formatter(date_form)
ax[1].set_xlim([ datetime.strptime(str(data.tstart), '%Y-%m-%d %H:%M:%S'), datetime.strptime(str(data.tend), '%Y-%m-%d %H:%M:%S')])
ax[1].set_ylim([-2,25])
ax[1].legend(loc=1)
ax[1].grid(axis = 'x', color='0.95')

ax[2].plot(pdData.t, pdData.s_V, 'k.', lw=0.5, label='iSpin')
ax[2].plot(pdData.t, pdData.dwd_ws, 'b.', lw=0.25, label='DWD BHV') 
# ax[2].plot(pdData.t[avail.filter['ispin']] , pdData.s_V[avail.filter['ispin']] ,'.',label = 'Valid data')
ax[2].set_xlabel('date')
ax[2].set_ylabel("wind speed [m/s]")
date_form = DateFormatter("%d/%m/%y")
ax[2].xaxis.set_major_formatter(date_form)
ax[2].set_xlim([ datetime.strptime(str(data.tstart), '%Y-%m-%d %H:%M:%S'), datetime.strptime(str(data.tend), '%Y-%m-%d %H:%M:%S')])
ax[2].set_ylim([-2,25])
ax[2].legend(loc=1)
ax[2].grid(axis = 'x', color='0.95')

ax[3].plot(pdData.t[avail.filter['sonics']] , pdData.gill_u115[avail.filter['sonics']] ,'k.',label = 'gill u115')
ax[3].plot(pdData.t[avail.filter['sonics']] , pdData.gill_u55[avail.filter['sonics']] - 1 ,'g.',label = 'gill u55')
ax[3].plot(pdData.t[avail.filter['sonics']] , pdData.thies_u25[avail.filter['sonics']] + 1 ,'b.',label = 'thies u25')
ax[3].set_xlabel('date')
ax[3].set_ylabel("wind speed [m/s]")
date_form = DateFormatter("%d/%m/%y")
ax[3].xaxis.set_major_formatter(date_form)
ax[3].set_xlim([ datetime.strptime(str(data.tstart), '%Y-%m-%d %H:%M:%S'), datetime.strptime(str(data.tend), '%Y-%m-%d %H:%M:%S')])
ax[3].set_ylim([-25,25])
ax[3].legend(loc=1)
ax[3].grid(axis = 'x', color='0.95')

ax[4].plot(pdData.t[avail.filter['btc']] , pdData.btc_v110[avail.filter['btc']] ,'k.',label = 'BlackTC')
ax[4].plot(pdData.t[avail.filter['gpo']] , pdData.btc_v110[avail.filter['gpo']] - 1 ,'g.',label = 'GreenPO-1m/s')
ax[4].plot(pdData.t[avail.filter['bpo']] , pdData.btc_v110[avail.filter['bpo']] + 1 ,'b.',label = 'BluePO+1m/s')
ax[4].set_xlabel('date')
ax[4].set_ylabel("wind speed [m/s]")
date_form = DateFormatter("%d/%m/%y")
ax[4].xaxis.set_major_formatter(date_form)
ax[4].set_xlim([ datetime.strptime(str(data.tstart), '%Y-%m-%d %H:%M:%S'), datetime.strptime(str(data.tend), '%Y-%m-%d %H:%M:%S')])
ax[4].set_ylim([-2,25])
ax[4].legend(loc=1)
ax[4].grid(axis = 'x', color='0.95')
plt.xticks(rotation=30)

fig.savefig("../results/results2.png", format='png')

del ax, color, date_form, fig, labs, p, p1, p1a, p2, tw, i, index, 
#%% write the turbine availability to an excel file
import openpyxl as opl
from openpyxl.utils.dataframe import dataframe_to_rows

output = struct()

# choose local files here (for Ashim)
# csv1_path =r'../results/Availability.csv'
# csv2_path = r'../results/Stat.csv'

# choose files on Z Drive (others)
output.csv1_path =r'z:/Projekte/109797-TestfeldBHV/30_Technical_execution_Confidential/TP3/AP2_Aufbau_Infrastruktur/Infrastruktur_Windmessung/02_Equipment/Data Management/WeeklyReports/Availability.csv'
output.csv2_path = r'z:/Projekte/109797-TestfeldBHV/30_Technical_execution_Confidential/TP3/AP2_Aufbau_Infrastruktur/Infrastruktur_Windmessung/02_Equipment/Data Management/WeeklyReports/Stat.csv'

# option 2 using to_csv
with open(output.csv1_path, 'a+') as f:
    avail.weekly.to_csv(f, header=f.tell()==0,escapechar='\n', index=True, line_terminator='\n')

with open(output.csv2_path, 'a+') as f:
    avail.stat[avail.stat.index=='N_avail'].to_csv(f, header=f.tell()==0,escapechar='\n', index=True, line_terminator='\n')

#%% publish a report of Wind turbine availability
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph, Spacer, Table, Image, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.units import inch

output.report = struct()

output.report.styles = getSampleStyleSheet()
output.report.ps = ParagraphStyle('title', fontSize=20, leading=24)
output.report.doc = SimpleDocTemplate("../results/WeeklyReport.pdf")

# create a Title
output.report.title = Paragraph("Testfeld BHV: Weekly Availability report", output.report.styles["h1"])

# create Text
output.report.par1 = Paragraph(" <br/> The weekly report presents the availability of wind measurements mainly \
        based on 10-min. average data from OneDAS. The threshold for data availability within a day is 25% \
            i.e. if the sensor availability is 50%, it is termed as 1 (green) <br/> \
        Note: <br/>\
            Data Availability > 25% ---> 1,  (else 0) <br/> \
            wt - wind turbine AD8-180 sensors (omega, Pw) combined into one variable <br/>\
            metmast - Availability combination of sensors on metmast except sonic anemometers <br/> \
            sonics - Availability combination of gill_u115, gill_u55 and thies_u25 <br/> \
                     ", output.report.styles["Normal"])
# create a Table
# generate a table borders
# add data
head1 = [['Date'] + avail.weekly.T.columns.strftime('%Y-%m-%d').astype(str).tolist()]
head2 = [['Day'] + avail.weekly.T.columns.strftime('%a').astype(str).tolist()]
head3 = [['Sensors'] + len(avail.weekly.T.columns)*[None]] 
output.report.headers = head1 + head2 + head3
output.report.table_data = output.report.headers + avail.weekly.T.reset_index().values.tolist()

def FnColorCodeTable(table_data):
    table_style = TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), 
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ])

    for row, values in enumerate(table_data):
        for column, value in enumerate(values):
        # for j in len(values):
            if value == 0:
                table_style.add('BACKGROUND', (column, row), (column, row), colors.red)
            elif value == 1:
                table_style.add('BACKGROUND', (column, row), (column, row), colors.green)
            else:
                table_style.add('BACKGROUND', (column, row), (column, row), colors.white)
    return table_style, table_data

output.report.table_style, output.report.table_data = FnColorCodeTable(output.report.table_data)

output.report.table = Table(data=output.report.table_data, style=output.report.table_style)

del head1, head2, head3
#%% add images into pdf
img1_file = "../results/results1.png"
img2_file = "../results/results2.png"
img3_file = "../results/MetmastSensors_17062021.png"


im1 = Image(img1_file, 6*inch, 9*inch)
im2 = Image(img2_file, 6*inch, 9*inch)
im3 = Image(img3_file, 7*inch, 8*inch)

info = []
info.append(output.report.title)
info.append(output.report.par1)
# info.append(Spacer(1,0.25*inch))
info.append(output.report.table)
info.append(im3)
info.append(Paragraph('', output.report.ps))
info.append(im1)
info.append(im2)

output.report.doc.build(info)

import glob
import shutil

# Sukshata shall send the pdfs with email
for f in glob.glob('../results/WeeklyReport.pdf'):
    shutil.copy(f, 'z:/Projekte/109797-TestfeldBHV/30_Technical_execution_Confidential/TP3/AP2_Aufbau_Infrastruktur/Infrastruktur_Windmessung/02_Equipment/Data Management/WeeklyReports')

for f in glob.glob('../results/*.csv'):
    shutil.copy(f, 'z:/Projekte/109797-TestfeldBHV/30_Technical_execution_Confidential/TP3/AP2_Aufbau_Infrastruktur/Infrastruktur_Windmessung/02_Equipment/Data Management/WeeklyReports')

del img1_file, img2_file, img3_file, im1, im2, im3, info, 

sys.exit('manual Stop')

#%% Sending Email to recipents with the report attached [## to be completed later]
# def FnSendEmail
import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

subject = "An email with attachment from Python"
body = "This is an email with attachment sent from Python"
# sender_email = "my@gmail.com"
# receiver_email = "your@gmail.com"
# password = input("Type your password and press enter:")

# expected settings
port = 25  # For SSL
smtp_server = '153.96.93.5'
sender_email = "ashim.giyanani@iwes.fraunhofer.de"  # Enter your address
receiver_email = "ashim.giyanani@iwes.fraunhofer.de"  # Enter receiver address
password = ''
# message = """\
# Subject: Hi there

# This message is sent from Python."""

# Create a multipart message and set headers
message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = subject
message["Bcc"] = receiver_email  # Recommended for mass emails

# Add body to email
message.attach(MIMEText(body, "plain"))

filename = "../results/WeeklyReport.pdf"  # In same directory as script

# Open PDF file in binary mode
with open(filename, "rb") as attachment:
    # Add file as application/octet-stream
    # Email client can usually download this automatically as attachment
    part = MIMEBase("application", "octet-stream")
    part.set_payload(attachment.read())

# Encode file in ASCII characters to send by email    
encoders.encode_base64(part)

# Add header as key/value pair to attachment part
part.add_header(
    "Content-Disposition",
    f"attachment; filename= {filename}",
)

# Add attachment to message and convert message to string
message.attach(part)
text = message.as_string()

# connect using ssl
# context = ssl.create_default_context()
# with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
#     server.login(sender_email, password)
#     server.sendmail(sender_email, receiver_email, message)

# connect using .starttls
context = ssl.create_default_context()
with smtplib.SMTP(smtp_server, port) as server:
    server.ehlo()  # Can be omitted
    server.starttls(context=context)
    server.ehlo()  # Can be omitted
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)    


#%% Removed code
# path = r'c:\Users\giyash\OneDrive - Fraunhofer\Python\Data\OneDAS_2019-01-01T00-00_600_s_56c64a34'
# filename = glob.glob(path+'\AIRPORT_AD8_PROTOTYPE*.csv')
# df=pd.read_csv(filename[0], sep = ';',decimal = ',',header=9, skiprows=[10,11], usecols=[0,1,2], names=['index','cupWs', 'omega'])
# time = [begin + timedelta(seconds=i/data.sampleRate) for i in range(len(df['index']))]

# cond0 = pdData.omega.notnull()
# cond1 = pd.to_numeric(pdData.omega) > 0
# Avail = (cond0 & cond1).astype('uint8')
# daily_avail = []
# for i in np.arange(int(len(pdData.omega)/144)):
#     avail = Avail[144*i:144*(i+1)].mean()
#     if avail > 0.25:
#         daily_avail.append(1)
#     else:
#         daily_avail.append(0)
