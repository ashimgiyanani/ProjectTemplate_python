def FnDataAvailability(sampleRate, param, df, xrange, **kwargs):
# filling in the weekly availabiliy as 1/0 based on number of points
# Inputs:
##################
# sampleRate - sampling freq [Hz]
# param - parameters within the pandas dataframe, in case of more than one sensor
# df - input pandas dataframe including the data to be evaluated for weekly availability
# tstart - (optional) provide the starting time explicitely, default df.t[0]
# cond0 - (optional) default: df[param].notnull()
# cond1 - (optional) check for data availaibility with the timerange [tstart, tend]
# cond2 - availability signal, if included in DataFrame
# cond3 - any other conditions
# cond4 - filtering the physical limits, within the range [x_min, x_max]

# Outputs:
#################
# weekly_avail - weekly availaibility of the param, same size as param


    from datetime import datetime
    import pandas as pd
    import numpy as np

    # filtering conditions
    cond0 = kwargs.pop('cond0', df[param].notnull()) # checks null values
    try:
        tstart
    except NameError:
        tstart = kwargs.pop('tstart', df.t[0]) 
    tend = tstart + timedelta(days=7)
    cond1 = kwargs.pop('cond1', (df.t>=tstart)&(df.t<=tend)) # checks date ranges
    cond2 = kwargs.pop('cond2', df[param].notnull()) # quality filter
    cond3 = kwargs.pop('cond3', df[np.isfinite(df[param]) | np.isnan(df[param])]) # quality filter
    cond4 = kwargs.pop('cond4', (df[param] > xrange[0]) & (df[param] < xrange[1]) ) # checks physical ranges

    # find the availability of param over a windowed data
    step= sampleRate * 24 * 60 * 60  # advance of the window
    length = sampleRate * 24 * 60 * 60 # width of the window
    N_win = np.int64(len(df.loc[:,param])/step)
    window = np.transpose(list(more_itertools.windowed(df.loc[:,param], n=length, \
                     fillvalue=np.nan, step=step)))
    condn = np.transpose(list(more_itertools.windowed(np.array(cond0 & cond1 & cond2 \
                & cond2 & cond3 & cond4), n=length, fillvalue=np.nan,\
                     step=step))).astype(bool)
    for j in np.arange(N_win):
        daily_avail = window[condn[:,j],j].shape[0]/length
        print('{:.1f}'.format(daily_avail))
        if daily_avail > 0.3:
            avail.append(1)
        else:
            avail.append(0)
    return avail

# Example:
from datetime import datetime
from datetime import *
import sys
sys.path.append(r"../../userModules")
from FnImportOneDas import FnImportOneDas

sampleRate = 1/600
param = ['s_V']
channel_paths = [    
    '/AIRPORT/AD8_PROTOTYPE/ISPIN/SaDataValid/600 s',
    '/AIRPORT/AD8_PROTOTYPE/ISPIN/WS_free_avg/600 s',
    '/AIRPORT/AD8_PROTOTYPE/ISPIN/DataOK/600 s'
]
ch_names = [
        's_valid',
        's_V', 
        's_ok'
]
tstart = datetime.strptime('2021-09-13_00-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
# funktioniert
tend = tstart + timedelta(days=7) # Select start date in the form yyyy-mm-dd_HH-MM-SS
# folder where data will be stored 
target_folder = r"../data"
odcData, pdData, t = FnImportOneDas(tstart, tend, channel_paths, ch_names, sampleRate, target_folder)
df = pdData
xrange = [0, 50]
avail = FnDataAvailability(sampleRate, param, df, xrange)
