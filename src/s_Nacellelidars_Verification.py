# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 13:32:35 2020

@author: papalk
"""

import sys
import datetime as dt
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.dates import DateFormatter
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D
import glob

sys.path.insert(1, r'c:\Users\giyash\ownCloud\Data\TestfeldBHV')
sys.path.append(r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\testfeld-bhv\userModules")

plt.style.use('seaborn-whitegrid')
SMALL_SIZE = 22
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

#%% Import data - Set import parameters

# Change parameters to select the data
device = 'BluePO' #Select between GreenPO, BluePO, BlackTC
dt_start ='2021-01-25_00-00-00' # Select start date in the form yyyy-mm-dd_HH-MM-SS
dt_end = dt.datetime.strptime(dt_start, '%Y-%m-%d_%H-%M-%S')  + dt.timedelta(days=60)# Select end date in the form yyyy-mm-dd_HH-MM-SS
dt_end = dt_end.strftime('%Y-%m-%d_%H-%M-%S')
i_RaworAverage = 1 #Select if you want to look into raw (0) or average data (1)

# Set path
path = r'Z:\Projekte\109797-TestfeldBHV\30_Technical_execution_Confidential\TP3\AP2_Aufbau_Infrastruktur\Infrastruktur_Windmessung\02_Equipment\04_Nacelle-Lidars-Inflow_CONFIDENTIAL\30_Data\\'

# Load all data for all three instruments
df_all = {}
for device in ["GreenPO","BluePO","BlackTC"]:
    # import data
    from f_import_nacelle_lidar_data import f_import_nacelle_lidar_data
    df_all[device] = f_import_nacelle_lidar_data(path,device,i_RaworAverage,dt_start,dt_end)

#%% Verification - Between WindIris

 
# Select instruments
dev0 = 'GreenPO' # Instrument 1
dev1 = 'BluePO' # Instrument 2  

# Select parameter to verify
sensor = 'RWS2'

# Select range
# for i_range in [50, 80, 120, 140, 160, 180, 200]:  # 
for i_range in [50,80,120,140,160,180,200,220,240,260,280,320,360,400,450,500,550,600,650,700 ]:  
    # Select filter threshold
    c_minavail = 0.8 # Minimum availability (change if needed) (for TC 0.8 and for PO 0.95)
    c_minCNR = -30 # Minimum CNR threshold (change if needed)
    c_minWSP = 3 # Minimum wind speed threshold (change if needed)
    c_maxWSP = 16 # Maximum wind speed threshold (change if needed)
    
    
    # Filter BAD data
    df0 = df_all[dev0]
    df1 = df_all[dev1]
    combined = pd.merge(left=df0, left_on='Date and Time',
             right=df1, right_on='Date and Time')
    
    combined_fil = combined
    combined_fil = combined_fil.dropna(subset=[sensor+'_x', sensor+'_y']) # Drop rows with nans
    combined_fil = combined_fil.drop(combined_fil[combined_fil[sensor+' availability_x'] < c_minavail].index) # Drop rows with x low availability
    combined_fil = combined_fil.drop(combined_fil[combined_fil[sensor+' availability_y'] < c_minavail].index) # Drop rows with y low availability
    combined_fil = combined_fil.drop(combined_fil[(combined_fil[sensor+'_x'] < c_minWSP)|(combined_fil[sensor+'_x'] > c_maxWSP)].index) # Drop rows with x high or low wind speed
    combined_fil = combined_fil.drop(combined_fil[(combined_fil[sensor+'_y'] < c_minWSP)|(combined_fil[sensor+'_y'] > c_maxWSP)].index) # Drop rows with y high or low wind speed
    # combined_fil = combined_fil.drop(combined_fil[combined_fil['CNR3'+'_x'] < c_minCNR].index) # Drop rows with x low CNR
    # combined_fil = combined_fil.drop(combined_fil[combined_fil['CNR3'+'_y'] < c_minCNR].index) # Drop rows with y low CNR
    combined_fil = combined_fil.drop(combined_fil[combined_fil['Distance_x'] != i_range].index) # Drop rows not matching i_range x
    combined_fil = combined_fil.drop(combined_fil[combined_fil['Distance_y'] != i_range].index) # Drop rows not matching i_range y
    
    
    # Calculate correlation
    x =  combined_fil[sensor+'_x'].values.reshape(-1, 1)
    y =  combined_fil[sensor+'_y'].values.reshape(-1, 1)
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(x,y)
    beta = model.intercept_ # intercept
    alpha = model.coef_ # slope
    R2 = model.score(x, y) # correlation coefficient
    y_pred = model.predict(x)  # make predictions
    
    
    # Plot correlation
    fig = plt.figure(figsize =  (10, 8)) 
    plt.plot(combined_fil[sensor+'_x'],combined_fil[sensor+'_y'],'.')
    plt.plot(x.flatten(), y_pred, color = 'k',linewidth=3);
    plt.plot( [0,18],[0,18],'r--' )
    plt.xlim(2,18)
    plt.ylim(2,18)
    plt.text(12, 6, 'y = {a:.2f}x+{b:.2f}'.format(a = float(alpha[0]),b = float(beta)))
    plt.text(12, 5, 'R$^2$ = {a:.3f}'.format(a = R2))
    plt.text(12, 4, '#data = {a:.0f} '.format(a = len(y)))
    plt.xlabel(sensor +' '+ dev0,labelpad=10,weight= 'bold')
    plt.ylabel(sensor +' '+ dev1,labelpad=10,weight= 'bold')
    plt.title('range = '+str(i_range))
    plt.savefig(r'Z:\Projekte\109797-TestfeldBHV\30_Technical_execution_Confidential\TP3\AP2_Aufbau_Infrastruktur\Infrastruktur_Windmessung\02_Equipment\04_Nacelle-Lidars-Inflow_CONFIDENTIAL\30_Data\Verification\Cor_'+
                sensor.replace(" ", "")+'_'+dev0+'_'+dev1+'_'+
                str(i_range)+'_'+dt_start[0:10]+'_'+dt_end[0:10]+'.png',bbox_inches='tight',dpi = 100)


#%% Compare to MM
# Met mast data
from class_mm import C_MMprocess
start = C_MMprocess(r'C:\Users\papalk\Desktop\OneDAS_2020-07-10T00-00_600_s_3efeff60.zip','600S')
alldata = start.loadzip()

# date-ws_free_avg
fig = plt.figure(figsize =  (20, 8)) 
ax = fig.add_subplot(111)
ax.plot(alldata.M0000_V1,'.',label = 'Gill110');
ax.plot(data_mm.index,data_mm.M0000_V1, '.',label = 'Cup115');
ax.set_xlabel('time ',labelpad=40,weight= 'bold')
ax.set_ylabel(["m/s"],labelpad=40,weight= 'bold')
date_form = DateFormatter("%H:%M:%S")
ax.xaxis.set_major_formatter(date_form)
legend = ax.legend(frameon = 1)
frame = legend.get_frame()
frame.set_color('white')
