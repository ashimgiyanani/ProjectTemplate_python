# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 23:35:51 2020

@author: giyash
"""
def odc_exportData(begin, end, channel_paths, target_folder):
# Function definition to export variables in the channel paths to python target folder and loads the data into the python

# Inputs:
# begin - begin timestamp in the following format-> datetime.datetime(year, month, day[, hour[, minute[, second[, microsecond[, tzinfo]]]]])
# end - end timestamp in the following format -> datetime.datetime(year, month, day[, hour[, minute[, second[, microsecond[, tzinfo]]]]])
# channel_paths - exact names and formats as in OneDas explorer
# target_folder - target folder, where the data needs to be saved, if not to be saved, target = None

# Output:
# data - exported data

# Example:
# usr = "ashim.giyanani@iwes.fraunhofer.de"
# pswd = "" # password = input("Please enter you password: ")
# begin = datetime(2020, 8, 9, 10, 0, tzinfo=timezone.utc)
# end   = datetime(2020, 8, 10, 11, 0, tzinfo=timezone.utc)
# channel_paths = [
#     "/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_110m_CNR/600 s",
# ]
# target_folder = r"c:/Users/giyash/OneDrive - Fraunhofer/Python/Scripts/OneDasExplorer/Python Connector/data"
# sample_import_ag(usr,pswd,begin,end, channel_paths, target)

    import asyncio
    from datetime import datetime, timedelta, timezone
    
    import matplotlib.pyplot as plt
    
    from OneDasConnector import OneDasConnector
    import nest_asyncio
    nest_asyncio.apply()
    import warnings
    warnings.filterwarnings("ignore")
    
    # settings
    scheme = "https"
    host = "onedas.iwes.fraunhofer.de"
    port = 443
    username = "ashim.giyanani@iwes.fraunhofer.de" # Fraunhofer username
    password = "xxx" # password = input("Please enter your password: ")
    
   # load data
    connector = OneDasConnector(scheme, host, port, username, password) 
    # without authentication: connector = OneDasConnector(scheme, host, port)
    
    params ={
            "FileGranularity": "SingleFile",
            "FileFormat": "CSV",
            "ChannelPaths": channel_paths,
            "CsvRowIndexFormat": "Unix"
            }
   
    try:
        if not target_folder:
            data = asyncio.run(connector.load(begin, end, params))
        else:
            data = asyncio.run(connector.load(begin, end, params))
            asyncio.run(connector.export(begin, end, params, target_folder))
    except:
        try:
            loop = asyncio.get_running_loop()
            data = loop.run_until_complete(connector.load(begin, end, params))
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            print('Async event loop already running')
            task = loop.create_task(connector.load(begin, end, params))
            data = asyncio.run(connector.load(begin, end, params))
        else:
            print('Starting new loop')
            data = asyncio.run(connector.load(begin, end, params))
    
    return data

# # Example:
# from datetime import datetime, timedelta, timezone
# begin = datetime(2021, 7, 19, 00, 0, tzinfo=timezone.utc)
# # funktioniert
# # end   = datetime(2021, 1, 19, 00, 0, tzinfo=timezone.utc)
# # funktioniert nicht
# end   = datetime(2021, 7, 20, 00, 0, tzinfo=timezone.utc)

# # # must all be of the same sample rate
# channel_paths = [
#  	'/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/GreenPO_450m_HWS_hub/600 s',
#  	]
# target_folder = r"c:/Users/giyash/OneDrive - Fraunhofer/Python/Scripts/OneDasExplorer/Python Connector/data"
# target_folder = ""
# data = odc_exportData(begin, end, channel_paths, target_folder)
