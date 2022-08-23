# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:33:36 2021

@author: giyash
"""

def FnImportOneDas(tstart, tend, channel_paths, ch_names,sampleRate, target_folder):
	import pandas as pd
	import numpy as np
	import sys, os
	sys.path.append(r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\userModules")
	import matlab2py as m2p
	from pythonAssist import now, struct
	sys.path.append("c:/Users/giyash/OneDrive - Fraunhofer/Python/Scripts/OneDasExplorer/Python Connector")
	from odc_exportData import odc_exportData
	import datetime as dt
	from datetime import datetime

    ## Initializations
	try:
		tstart, tend
	except NameError:
		sys.exit('tstart and tend necessary')

	begin = tstart 
	end   = tend #- dt.timedelta(0,0,0,0,10)
	try:
		sampleRate
	except NameError:
		sampleRate = 1/600

	if sampleRate == 1:
		Fs = 's'
	elif sampleRate == 1/4:
		Fs = '4s'
	elif sampleRate == 1/600:
		Fs = '10T'
	elif (sampleRate == 20):
		Fs = '50ms'
	else:
		Fs = 'H'

	from pandas.tseries.frequencies import to_offset
	t = pd.date_range(start=begin + pd.to_timedelta(to_offset(Fs)),  end=end, freq = Fs)    

	# ch_names =['Shead', 'Struehead', 'Swd_scada', 'Syaw']  #
	try:
		target_folder
	except NameError:
		# target folder points to default folder
		target_folder = r"c:/Users/giyash/OneDrive - Fraunhofer/Python/Scripts/OneDasExplorer/Python Connector/data"

	try:
		ch_names, channel_paths
	except NameError:
		print('[{}]: Channel names and paths are missing'.format(now()))
		sys.exit('Rerun the script by providing the channel names and path')

	data = odc_exportData(begin, end, channel_paths, target_folder)
	odcData = struct()
	setattr(odcData,'t',np.array(t))
	pdData = pd.DataFrame(dtype=np.float64(),index=t)

	# assign the data values to variable names that are given by ch_names
	i = 0
	for keys in data:
		pdData[ch_names[i]] = np.array(data[keys].values)
		setattr(odcData,ch_names[i], np.array(data[keys].values))        
		i = i+1

	t = pd.DatetimeIndex(t) # concert to datetime64 format
	setattr(odcData,'t',t)
	pdData['t'] = t
	return odcData, pdData, t


# Example:
# must all be of the same sample rate
# channel_paths = [
#    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0000_V1/600 s_mean',
#     '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/P0420_RotorSpdRaw/600 s_mean',
#     '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0010_V2/600 s_mean',
#     '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0020_V3/600 s_mean',
#     '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0030_V4/600 s_mean',
#     '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0040_V5/600 s_mean',
#     '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0050_V6/600 s_mean',
#     '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0060_Precipitation/600 s_mean',
#     '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0070_D1/600 s_mean_polar',
#     '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0100_D4/600 s_mean_polar',
#     '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0110_D5/600 s_mean_polar',
#     '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0200_B1/600 s_mean',
#     '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0210_B2/600 s_mean',
#     '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0220_T1/600 s_mean',
#     '/AIRPORT/AD8_PROTOTYPE/ISPIN/SaDataValid/600 s',
#     '/AIRPORT/AD8_PROTOTYPE/ISPIN/WS_free_avg/600 s',
#     '/AIRPORT/AD8_PROTOTYPE/ISPIN/DataOK/600 s',
#     '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/BlackTC_110m_HWS_hub/600 s',
#     '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/BluePO_110m_HWS_hub/600 s',
#     '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/GreenPO_110m_HWS_hub/600 s',
#     '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/BlackTC_110m_HWS_hub_availability/600 s',
#     '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/BluePO_110m_HWS_hub_availability/600 s',
#     '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/GreenPO_110m_HWS_hub_availability/600 s',
#     '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_115m_Wind_Speed/600 s',
# 	]

# ch_names = [
#         'v1',
#         'omega',
#         'v2',
#         'v3',
#         'v4',
#         'v5',
#         'v6',
#         'prec',
#         'd1',
#         'd4',
#         'd5',
#         'b1',
#         'b2',
#         'T1',
#         's_valid',
#         's_V', 
#         's_ok',
#         'btc_v110',
#         'bpo_v110', 
#         'gpo_v110', 
#         'btc_Av110', 
#         'bpo_Av110', 
#         'gpo_Av110', 
#         'wc_v115',
# 	]

# import datetime as dt
# from datetime import datetime
# tstart = datetime.strptime('2021-01-13_00-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
# # funktioniert
# tend = datetime.strptime('2021-01-19_00-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
# # funktioniert nicht
# # tend = datetime.strptime('2021-02-01_00-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
# target_folder = r"c:/Users/giyash/OneDrive - Fraunhofer/Python/Scripts/OneDasExplorer/Python Connector/data"
# sampleRate= 600
# odc, df, t = FnImportOneDas(tstart, tend, channel_paths, ch_names, sampleRate, target_folder)
