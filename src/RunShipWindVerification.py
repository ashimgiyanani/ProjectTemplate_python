# Script to verify the algorithm to correct the motion of the ship

# import modules
import sys
sys.path.append(r"c:/Users/giyash/OneDrive - Fraunhofer/Python/Scripts/userModules/")
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import pythonAssist as pa # definitions useful in python in general
import numpy.matlib
import scipy.linalg
import matlab2py as m2p # includes some definitions helping matlab users 
from scipy.spatial.distance import cdist # euclidean distance between grid points
from KaimalSpec import KaimalSpec
from wind1D import wind1D


# generate a wind time series
T = 600 # total time [s]
Nfft = np.power(2,12) # no. of fft terms, always in power of 2 [-]
Uhub = 10 # hubheight wind speed [m/s]
H = 100 # hub height [m]
z0 = 0.001                   # roughness length [m] should be > 0
lat = 52.4                 # lattiutude of measurement [deg]
sigmau = 1               # sigma from measurements, if available
sigmav = 0.8*sigmau
xLu = 100                  # length scale from measurements / guess
xLv = xLu
df = 1/T
t, u, f, Su = wind1D(T,Nfft,Uhub,H,z0,lat,sigmau,xLu)
_, v, _, Sv = wind1D(T,Nfft,Uhub,H,z0,lat,sigmav,xLv)

u, v = np.ravel(u), np.ravel(v)

# generate a time series for ship movement
ship_U = 5 + np.random.randn(len(t))
ship_Dir = 230 + np.random.randn(len(t))
ship_u = ship_U*np.cos(np.deg2rad(ship_Dir))
ship_v = ship_U*np.sin(np.deg2rad(ship_Dir))

# derived from an example of pendulum
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# given:-
g = 9.81  # accl due to gravity
L = 1 # length of swivel
b = 2 # damping kg/s
m = 100 # weight of boat

# 1st order equations 
def pendulum_ode(t, Theta):
    dTheta2_dt = (-b/m)*Theta[1] + (-g/L)* np.sin(Theta[0])
    dTheta1_dt = Theta[1]
    return [dTheta1_dt, dTheta2_dt]



Theta1_0, Theta2_0 = 0, 3
Theta_0 = [Theta1_0, Theta2_0]
t_range = [0, 600]
Theta12 = solve_ivp(pendulum_ode, t_range, Theta_0, t_eval = t)
Theta1 = Theta12.y[0,:]    
Theta2 = Theta12.y[1,:]



# plot of angular displacement and velocity
import pandas as pd
import plotly.express as px
df = pd.DataFrame()
df['Theta1'] = Theta1
df['Theta2'] = Theta2
df['t'] = t
df.set_index(['t'])

fig = px.line(df, x='t', y = 'Theta1')
fig.show()


plt.plot(t,Theta1*20,label='Angular Displacement (rad)')
plt.plot(t,Theta2,label='Angular velocity (rad/s)')
plt.xlabel('Time(s)')
plt.ylabel('Angular Disp.(rad) and Angular Vel.(rad/s)')
plt.legend()
plt.show()




