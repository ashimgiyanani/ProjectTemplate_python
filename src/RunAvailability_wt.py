# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 19:41:21 2021

@author: giyash
"""
## user modules
import pandas as pd
import glob
import os
import numpy as np
import sys
sys.path.append(r"../../userModules")
sys.path.append(r"../../OneDasExplorer/Python Connector")

import runpy as rp
from FnWsRange import *
# import altair as alt
# import datapane as dp
import matplotlib.pyplot as plt
# from sanitize_dataframe import sanitize_dataframe
#datapane login --token=cebae30aca77e1422a72442c8c7113378fca57b3
# import altair_viewer as av

from csv import writer
import matlab2py as m2p
# from detecta import detect_peaks
from pythonAssist import *
import tikzplotlib as tz
from AD180 import AD180
from datetime import datetime
import re

from odc_exportData import odc_exportData
from datetime import *

begin = datetime(2021, 7, 7, 0, 0, tzinfo=timezone.utc)
end   = datetime(2021, 8, 9, 0, 0, tzinfo=timezone.utc)
sampleRate = 1/600
AppendLog = 1

# must all be of the same sample rate
channel_paths = [
    "/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0000_V1/600 s_mean",
    "/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/P0420_RotorSpdRaw/600 s_mean",
                ]
# Provide the names that you want to give to the channel paths
ch_names = ['v1','omega']
target_folder = r"c:/Users/giyash/OneDrive - Fraunhofer/Python/Scripts/OneDasExplorer/Python Connector/data"
data = odc_exportData(begin, end, channel_paths, target_folder)
odcData = struct()
# assign the data values to variable names that are given by ch_names
i = 0
for keys in data:
    setattr(odcData,ch_names[i], data[keys].values)        
    i = i+1
    time = [begin + timedelta(seconds=i/sampleRate) for i in range(len(odcData.v1))]
    tnew = pd.DatetimeIndex(time) # concert to datetime64 format
    setattr(odcData,'t',time)

odcData.omega = pd.DataFrame(odcData.omega)
cond0 = odcData.omega.notnull()
cond1 = (odcData.omega) > 1
Avail = (cond0 & cond1).astype('uint8')
daily_avail = []
for i in np.arange(int(len(odcData.omega)/144)):
    avail = Avail[144*i:144*(i+1)].mean(axis=0)
    if avail[0] > 0.25:
        daily_avail.append(1)
    else:
        daily_avail.append(0)
    del(avail)
    
# ## Import data from csv
# path = r'c:\Users\giyash\OneDrive - Fraunhofer\Python\Data\OneDAS_2019-01-01T00-00_600_s_56c64a34'
# filename = glob.glob(path+'\AIRPORT_AD8_PROTOTYPE*.csv')
# df=pd.read_csv(filename[0], sep = ';',decimal = ',',header=9, skiprows=[10,11], usecols=[0,1,2], names=['index','cupWs', 'omega'])
# time = [begin + timedelta(seconds=i/sampleRate) for i in range(len(df['index']))]

# cond0 = df.omega.notnull()
# cond1 = pd.to_numeric(df.omega) > 0
# Avail = (cond0 & cond1).astype('uint8')
# daily_avail = []
# for i in np.arange(int(len(df.omega)/144)):
#     avail = Avail[144*i:144*(i+1)].mean()
#     if avail > 0.25:
#         daily_avail.append(1)
#     else:
#         daily_avail.append(0)

import openpyxl as opl
xl_filepath = r'C:\Users\giyash\Documents\trial.xlsx'
wb = opl.load_workbook(xl_filepath)
#  grab the active worksheet
ws = wb['Sheet1']
# grab the workshet title
ws_title = ws.title
print('Active sheet title: {} \n'.format(ws_title))

# appending the worksheet

# iteration to find the last row with values in it
nrows = ws.max_row
lastrow = 0
if nrows > 1000:
    nrows = 1000
while True:
    if ws.cell(nrows, 3).value != None:
        lastrow = nrows
        break
    else:
        nrows -= 1

    # appending to the worksheet and saving it
if AppendLog==1: # if AppendLog is wished at start
    count=0
    # iterate over all entries in appData array which contains variables for appending the excel
    for ncol, entry in enumerate(daily_avail,start=1):
        # print(ncol, entry)
        ws.cell(row=1+nrows, column=ncol, value=entry)
        count += 1
    print('[{}] - No. of entries made: {} \n'.format(now(), count))
    wb.save(xl_filepath) # file should be closed to save
    print('[{}] - Changes saved to: {} \n'.format(now(), ws_title))
else:
    print('[{}] - Changes not saved to: {} \n'.format(now(), ws_title))

# #  Compare the wind speeds across different sensors
# fig, ax = plt.subplots()
# ax.plot(odcData.t,U[:,0],'m-', label = 'nac Lid BTC')
# ax.plot(odcData.t,odcData.wc115m_U,'k--', label='WC@115m')
# ax.plot(odcData.t,odcData.v1,'b-', label='cup1')
# ax.plot(odcData.t,odcData.v2,'g-', label='cup2')
# ax.plot(odcData.t,odcData.usa1_v,'r-', label= 'sonic1')
# plt.xlabel('Time')
# plt.ylabel('wind speeds [m/s]')
# plt.legend()
# plt.show()
