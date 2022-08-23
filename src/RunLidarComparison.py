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

# sys.path.insert(1, r'c:\Users\giyash\ownCloud\Data\TestfeldBHV')
sys.path.append(r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\userModules")

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


# Change parameters to select the data
device = 'GreenPO' #Select between GreenPO, BluePO, BlackTC
dt_start ='2021-01-25_00-00-00' # Select start date in the form yyyy-mm-dd_HH-MM-SS
dt_end = dt.datetime.strptime(dt_start, '%Y-%m-%d_%H-%M-%S')  + dt.timedelta(days=320)# Select end date in the form yyyy-mm-dd_HH-MM-SS
dt_end = dt_end.strftime('%Y-%m-%d_%H-%M-%S')
i_RaworAverage = 1 #Select if you want to look into raw (0) or average data (1)

# Set path
path = r'Z:\Projekte\109797-TestfeldBHV\30_Technical_execution_Confidential\TP3\AP2_Aufbau_Infrastruktur\Infrastruktur_Windmessung\02_Equipment\04_Nacelle-Lidars-Inflow_CONFIDENTIAL\30_Data\\'
# Load all data for all three instruments
df_all = {}
for device in ["GreenPO"]:
    # import data
    from f_import_nacelle_lidar_data import f_import_nacelle_lidar_data
    df_all[device] = f_import_nacelle_lidar_data(path,device,i_RaworAverage,dt_start,dt_end)

path = r'Z:\Projekte\109797-TestfeldBHV\30_Technical_execution_Confidential\TP3\AP2_Aufbau_Infrastruktur\Infrastruktur_Windmessung\02_Equipment\04_Nacelle-Lidars-Inflow_CONFIDENTIAL\30_Data\HourlyData\\'
df_all['NewConfig'] = f_import_nacelle_lidar_data(path,device,i_RaworAverage,dt_start,dt_end)

plt.plot(df_all['GreenPO']['Date and Time'], df_all['GreenPO']['HWS hub'], 'k.')
plt.plot(df_all['NewConfig']['Date and Time'], df_all['NewConfig']['HWS hub'], 'g.')
plt.show()

plt.plot(df_all['GreenPO']['HWS hub'],df_all['NewConfig']['HWS hub'] )
plt.show()