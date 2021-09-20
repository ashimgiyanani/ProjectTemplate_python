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
    '/AIRPORT/AD8_PROTOTYPE/METMAST_EXTENSION/73011db3-9136-4d7b-9959-1f987a78299d/20 Hz'
    '/AIRPORT/AD8_PROTOTYPE/METMAST_EXTENSION/e1b85d5e-f732-4eed-82fb-565f41ccf1e4/20 Hz'
    '/AIRPORT/AD8_PROTOTYPE/METMAST_EXTENSION/448a1afd-6a2e-45f8-b614-5a8117d4e2b5/20 Hz'
    };

aa=[];


%% Load data and loop over days

period=dateTimeBegin;
while period<dateTimeEnd
% connect to OneDAS
connector = OneDasConnector(scheme, host, port, username, password);
% without authentication: connector = OneDasConnector(scheme, host, port)
params.ChannelPaths = channelPaths;
data                = connector.Load(period, period+hours(24), params);

% Save data in a structure
for k=1:length(channelPaths)
    AA{k,1}=data(channelPaths{k});
    vars{k,1}=AA{k,1}.Name;
end

% Create a time table
% Make a timestamp array
sampleRate  = 20; % 20Hz (adapt to your needs)
dt          = 1 / sampleRate / 86400;
time        = (datenum(period) : dt : datenum(period+hours(24)) - dt).';

% Compile the time table
try
TT=timetable(datetime(datevec(time)),AA{1}.Values,...
    'VariableNames',{AA{1}.Name});

for i=2:length(channelPaths)
    TT=addvars(TT,AA{i}.Values,'NewVariableNames',{AA{i}.Name});
end

% Count the daily number of valid data
TT1=retime(TT,'daily','count');
% store in an array
aa=vertcat(aa,timetable2table(TT1));
end

period=period+hours(24);
disp(strcat('Processed ','>>',datestr(period)));

end

%% Plot results
dailyrec=table2array(aa(:,2:end));

figure
bar(table2array(aa(:,1)),dailyrec./(20*60*60*24))
legend(vars,'Location','eastoutside')
hline(.7,'r')
hline(0.9,'k--')
ylabel('Daily data availability [-]')



