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


    import datetime
    import pandas as pd
    import numpy as np
    import itertools
    import more_itertools

    # filtering conditions
    cond0 = kwargs.pop('cond0', df[param].notnull()) # checks null values
    tstart = kwargs.pop('tstart', df.t[0]) 
    tend = tstart + datetime.timedelta(days=7)
    cond1 = kwargs.pop('cond1', np.array([(df.t>=tstart)&(df.t<=tend)]).reshape(-1,1)) # checks date ranges
    cond2 = kwargs.pop('cond2', (df[param] - df[param].mean()).notnull()) # quality filter
    cond3 = kwargs.pop('cond3', [~np.isfinite(df[param]) | ~np.isnan(df[param])]) # quality filter
    cond4 = kwargs.pop('cond4', (df[param] >= xrange[0]) & (df[param] <= xrange[1]) ) # checks physical ranges

    # bring all condn in one format
    cond0 = np.array(cond0).reshape(-1,1).astype('bool')
    cond1 = np.array(cond1).reshape(-1,1).astype('bool')
    cond2 = np.array(cond2).reshape(-1,1).astype('bool')
    cond3 = np.array(cond3).reshape(-1,1).astype('bool')
    cond4 = np.array(cond4).reshape(-1,1).astype('bool')

    # find the availability of param over a windowed data
    avail=[]
    step= int(sampleRate * 24 * 60 * 60)  # advance of the window
    length = int(sampleRate * 24 * 60 * 60) # width of the window
    N_win = np.int64(len(df.loc[:,param])/step)
    window = np.transpose(list(more_itertools.windowed(np.ravel(df.loc[:,param]), n=length, \
                     fillvalue=np.nan, step=step)))
    condn = np.transpose(list(more_itertools.windowed(np.ravel(cond0 & cond1 & cond2 & cond2 & cond3 & cond4), n=length, fillvalue=np.nan,step=step))).astype(bool)
    
    for j in np.arange(N_win):
        daily_avail = window[condn[:,j],j].shape[0]/length
        # print('{:.1f}'.format(daily_avail))
        if daily_avail > 0.3:
            avail.append(1)
        else:
            avail.append(0)
    
    # count the number of valid points and no. of input points
    Nvalid = np.sum(condn.reshape(-1,1)) # number of valid points
    Npts = len(df[param]) # all points
    N_avail = np.round(Nvalid/Npts * 100, decimals=0).astype(int) # Availability in %
    Nstat = pd.DataFrame(data=np.array([[int(Nvalid)], [int(Npts)], [N_avail]]).T, columns=['Nvalid', 'Npts', 'N_avail'] )
    return avail, condn.reshape(-1,1), Nstat

# Example:
# from datetime import datetime, timedelta
# import sys
# import pandas as pd
# import numpy as np

# sys.path.append(r"../../userModules")
# sys.path.append(r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\OneDasExplorer\Python Connector")
# from FnImportOneDas import FnImportOneDas

# sampleRate = 1/600
# param = ['s_V']
# channel_paths = [    
#     '/AIRPORT/AD8_PROTOTYPE/ISPIN/SaDataValid/600 s',
#     '/AIRPORT/AD8_PROTOTYPE/ISPIN/WS_free_avg/600 s',
#     '/AIRPORT/AD8_PROTOTYPE/ISPIN/DataOK/600 s'
# ]
# ch_names = [
#         's_valid',
#         's_V', 
#         's_ok'
# ]
# tstart = datetime.strptime('2021-09-06_00-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
# # funktioniert
# tend = tstart + timedelta(days=7) # Select start date in the form yyyy-mm-dd_HH-MM-SS
# # folder where data will be stored 
# target_folder = r"../data"
# odcData, pdData, t = FnImportOneDas(tstart, tend, channel_paths, ch_names, sampleRate, target_folder)
# df = pdData
# xrange = [0, 50]
# cond3 = (df[param][np.isfinite(df[param]) | np.isnan(df[param])]) # quality filter
# avail, cond, Nstat  = FnDataAvailability(sampleRate, param, df, xrange, cond3=cond3)
