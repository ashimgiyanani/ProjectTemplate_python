# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:31:56 2020

@author: papalk
"""


def f_import_nacelle_lidar_data(path_in,device,i_RaworAverage,dt_start,dt_end):
    import pandas as pd
    import numpy as np
    import glob 
    import datetime as dt

    # Construct  path
    path = path_in+device
    
    # Import csv
    if i_RaworAverage:
        all_filenames = np.asarray([i for i in glob.glob(path+'\*\*average_data'+'*.csv')])
        all_filenames_TF = [(i[-23:-4]>dt_start)&(i[-23:-4]<=dt_end) for i in glob.glob(path+'\*\*average_data'+'*.csv')]
        all_filenames = all_filenames[all_filenames_TF]
    else:
        all_filenames = np.asarray([i for i in glob.glob(path+'\*\*real_time_data'+'*.csv')])
        all_filenames_TF = [(i[-23:-4]>dt_start)&(i[-23:-4]<=dt_end) for i in glob.glob(path+'\*\*real_time_data'+'*.csv')]
        all_filenames = all_filenames[all_filenames_TF]
    
    if all_filenames.size>0:
        df= pd.concat([pd.read_csv(f, sep = ';',decimal = '.',header=0)  for f in all_filenames ])

        if i_RaworAverage:
             # 10 min
             df['Date and Time'] = [dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:00.000+00:00') for date in df['Date and Time']]
        else:
            # raw
            df['Timestamp'] = [dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f+00:00') for date in df['Timestamp']]
            df.index = df.Timestamp # Remove this to keep the index as as an increasing number

    else:
        df = []
                                                                    

    return(df)
