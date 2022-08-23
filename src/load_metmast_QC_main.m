%% settings
scheme          = 'https';
host            = 'onedas.iwes.fraunhofer.de';
port         	= 443;
% Edit your username here
username        = input('Please enter your username: ');
% Enter your password here
password        = input('Please enter your password: ');

% Choose the period of interest
dateTimeBegin 	= datetime(2021, 09, 01, 0, 0, 0, 'TimeZone', 'UTC');
dateTimeEnd 	= datetime(2021, 09, 13, 0, 0, 0, 'TimeZone', 'UTC');

% must all be of the same sample rate
channelPaths = { ...
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/15fe4966-2aa2-4090-82b6-3474d1416ed8/600 s_mean'
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/eba3bafd-d92e-4180-bb56-7158c86aa084/600 s_mean'
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/7fc45398-8cfd-47a6-8939-d8ee12f81e7a/600 s_mean'
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/c302298a-b6cb-47ed-95d0-22442973e85d/600 s_mean'
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/21e6ade8-3682-4948-9175-60d8dfc2ed72/600 s_mean'
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/aeb1cb78-c038-4569-b66e-02a15967caee/600 s_mean'
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/b579cd20-393b-4b37-8c08-262d22020139/600 s_sum'
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/88a8a5bf-a609-4fea-8e92-ae382330d878/600 s_mean_polar'
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/05886edb-7463-425d-b53b-9f521d56ec71/600 s_mean_polar'
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/c90a8a7a-7b8f-48c5-a1b7-6328b8628619/600 s_mean_polar'
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/40a12845-36c0-477a-86f4-2e97902e213c/600 s_mean'
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/8ef3db0d-624d-4c9f-9038-0d114fac3ebe/600 s_mean'
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/08c2f0db-bd7a-42a5-aac6-294cadc70cf2/600 s_mean'
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/6d218486-4a5f-4637-9837-342f5d6ca245/600 s_mean'
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/9d48ce34-6dcd-4d31-82ad-4182a885d97a/600 s_mean'
    };

%% load data
connector = OneDasConnector(scheme, host, port, username, password);
% without authentication: connector = OneDasConnector(scheme, host, port)
params.ChannelPaths = channelPaths;
data                = connector.Load(dateTimeBegin, dateTimeEnd, params);

% Save data in a structure
for k=1:length(channelPaths)
    AA{k,1}=data(channelPaths{k});
    vars{k,1}=AA{k,1}.Description;
end

%% Create a time table
% Make a timestamp array
sampleRate  = 1/600; % 1/600 Hz or 10-min (adapt to your needs)
dt          = 1 / sampleRate / 86400;
time        = (datenum(dateTimeBegin) : dt : datenum(dateTimeEnd) - dt).';

% Compile the time table
TT=timetable(datetime(datevec(time)),AA{1}.Values,...
    'VariableNames',{AA{1}.Description});

for i=2:length(channelPaths)
    TT=addvars(TT,AA{i}.Values,'NewVariableNames',{AA{i}.Description});
end

%% Count the daily number of valid data
TT1=retime(TT,'daily','count');
% store in an array
aa=timetable2table(TT1);
dailyrec=table2array(aa(:,2:end));

%% Plot results
figure
bar(TT1.Time,dailyrec./144)
legend(vars,'Location','eastoutside')
hline(.7,'r')
ylabel('Daily data availability [-]')



