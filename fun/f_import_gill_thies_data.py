# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:31:56 2020

@author: papalk
"""


def f_import_gill_thies_data(path_in,device,dt_start,dt_end):
    import pandas as pd
    import numpy as np
    import glob 
    import datetime as dt
    import os
    import os.path, time
    
    def addtimedecimal(l):
            for i,x in enumerate(l):    
                if '.' not in x:
                    l[i] = x+'.00'
                else:
                    l[i] = x
            return(l)

    # Construct  path
    path = path_in
    
    dt_start = dt.datetime.strptime(dt_start, '%Y-%m-%d %H:%M:%S') #Convert dt_start to timestamp
    dt_end = dt.datetime.strptime(dt_end, '%Y-%m-%d %H:%M:%S') #Convert dt_end to timestamp

    
    # Import csv
    if device == 'gill':
        all_filenames = np.asarray([i for i in glob.glob(path+'\*'+device+'*.dat')])
        all_filenames_TF = [(dt.datetime.strptime(i[-19:-4], '%Y_%m_%d_%H%M')>dt_start)&(dt.datetime.strptime(i[-19:-4], '%Y_%m_%d_%H%M')<=dt_end) for i in all_filenames]
        all_filenames = all_filenames[all_filenames_TF]
    else:
        all_filenames = np.asarray([i for i in glob.glob(path+'\*'+device+'*.dat')])
        all_filenames_TF = [(dt.datetime.strptime(i[-19:-4], '%Y_%m_%d_%H%M')>dt_start)&(dt.datetime.strptime(i[-19:-4], '%Y_%m_%d_%H%M')<=dt_end) for i in all_filenames]
        all_filenames = all_filenames[all_filenames_TF]            
        
    if all_filenames.size>0:
        if device == 'gill':
            df= pd.concat([pd.read_csv(f, sep = ',',decimal = '.',header=1,skiprows= [2,3],na_values = 'NAN',dtype = {'TIMESTAMP': object, 'RECORD': int, 'gill_115_id': object, 
                                                                               'gill_115_u': np.float64, 'gill_115_v': np.float64, 'gill_115_w': np.float64,
                                                                               'gill_115_unit': object, 'gill_115_SpeedOfSound': np.float64, 'gill_115_SonicTempC': np.float64,
                                                                               'gill_115_status': object,
                                                                               'gill_55_id': object,
                                                                               'gill_55_u': np.float64, 'gill_55_v': np.float64, 'gill_55_w': np.float64,
                                                                               'gill_55_unit': object, 'gill_55_SpeedOfSound': np.float64, 'gill_55_SonicTempC': np.float64,
                                                                               'gill_55_status': object})  for f in all_filenames ])
            
        else: 
            df= pd.concat([pd.read_csv(f, sep = ',',decimal = '.',header=1,skiprows= [2,3],na_values = 'NAN',dtype = {'TIMESTAMP': object, 'RECORD': int, 'thies_Vx': np.float64, 
                                                                               'thies_Vy': np.float64, 'thies_Vz': np.float64, 'thies_AvTc': np.float64,
                                                                               'thies_TiX':  np.float64, 'thies_TiY': np.float64, 'thies_TiZ': np.float64,
                                                                               'thies_ThiesStatus': np.float64,
                                                                               'thies_CheckSum': np.float64})  for f in all_filenames ])


        df['TIMESTAMP'] = addtimedecimal(df.TIMESTAMP.values)
        df['TIMESTAMP'] = [dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f') for date in df['TIMESTAMP']]
        df.index = df.TIMESTAMP # Remove this to keep the index as as an increasing number

    else:
        if device == 'gill':
            df = pd.DataFrame(np.nan, index=pd.date_range(dt_start,dt_end,freq = '1S'), columns=[ 'RECORD', 'gill_115_id', 
                                                                               'gill_115_u', 'gill_115_v', 'gill_115_w',
                                                                               'gill_115_unit', 'gill_115_SpeedOfSound', 'gill_115_SonicTempC',
                                                                               'gill_115_status',
                                                                               'gill_55_id',
                                                                               'gill_55_u', 'gill_55_v', 'gill_55_w',
                                                                               'gill_55_unit', 'gill_55_SpeedOfSound', 'gill_55_SonicTempC',
                                                                               'gill_55_status'
                                                                               ])
        else:
            df = pd.DataFrame(np.nan, index=pd.date_range(dt_start,dt_end,freq = '1S'), columns=[  'RECORD', 'thies_Vx', 
                                                                               'thies_Vy', 'thies_Vz', 'thies_AvTc',
                                                                               'thies_TiX', 'thies_TiY', 'thies_TiZ',
                                                                               'thies_ThiesStatus',
                                                                               'thies_CheckSum'])
        df.index.rename('TIMESTAMP', inplace=True)
                                                                    

    return(df)

