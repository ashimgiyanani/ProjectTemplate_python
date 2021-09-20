# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# Tasks:
# compare a time series at a point in the grid vs the time series in scanning pattern
# Compare the spectrum
# Compare the std dev
# Compare the length scales affecting

# %%
# import modules
%matplotlib qt
import numpy as np
import scipy as sp
import sys

from scipy.spatial.ckdtree import cKDTree
sys.path.insert(0,r"../fun")

from FnKaimal3D import FnKaimal3D
import tikzplotlib as tz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.spatial.transform import Rotation as R
import matlab2py as m2p
import pythonAssist as pa
import time
start_time = time.time()


# %%
# initialize and generate contrained turbulence
PlotFig = 1
SaveTex = 0
SaveFig = 0
saveArr = 1 # load array from pickle (saveArr=0), save array to pickle (saveArr=1)
skipGen = 0 # save uk, vk and wk to pickle (skipGen=0), (load from pickle skipGen=1)

Uhub = 10 # hubheight wind speed [m/s]
H = 115 # hub height [m]
z0 = 0.1                   # roughness length [m]
lat = 52.4                 # lattiutude of measurement [deg]
Nx, Ny, Nz = 11, 11, 11 
X = np.linspace(0, 400, Nx) # longitudinal dir, (row vector), [m] 
Y = np.linspace(-100, 100,Ny) # lateral direction, (row vector), [m]
Z = np.linspace(0, 230, Nz) # grid vertical direction (row vector), [m]
dx, dy, dz = np.diff(X)[0], np.diff(Y)[0], np.diff(Z)[0] 
dt = 0.01  # discrete sampling time [s]
dT = 0.2 # cycle time to complete one scanning pattern
Np = int(dT/dt) # number of sampling points in one scanning pattern
dX = dT*Uhub # step in longitudinal direction after 1 scanning pattern

try:
	dt
	Nfft = np.power(2, 11)
	T = Nfft*dt # total time [s]
except NameError:
	dt = dx/Uhub
	T = 600 # total simulation time
	Nt = T/dt
	pow = np.ceil(np.log2(abs(Nt)))
	Nfft = int(np.power(2,pow)) # no. of fft terms, always in power of 2 [-]

Fs = 1/dt
t = np.linspace(dt,T,Nfft)
X = (t*Uhub).round(decimals=3)
fs = 1/t
sigmau = 1               # sigma from measurements, if available
sigmav = 0.8*sigmau
sigmaw = 0.5*sigmau
u, v, w, Su, Sv, Sw, xr, Lu, Lv, Lw, f =  FnKaimal3D(X,Y,Z,Nfft, T,dt,H,z0, Uhub,sigmau,lat, seed=1985) 
U = Uhub + u
Iu = sigmau/Uhub
Iv = sigmav/Uhub
Iw = sigmaw/Uhub
print('{}: Kaimal 3D Wind generated'.format(pa.now()))

#%% generate turbulent wind fields from Turbsim


# %%
## Adding wind turbine dimensions to the domain
theta = np.linspace(0,2*np.pi,101)
# rotor
rc = 90 # rotor radius
H = 115 # hub height
rx = np.zeros(len(theta)) # rotor x-coord
ry = rc*np.sin(theta) # rotor y-coord
rz = rc*np.cos(theta) # rotor z-coord
# tower
ty = np.asarray([-3,-2,2,3,-3])
tz = np.asarray([0,H,H,0,0])
tx = np.zeros(len(ty))
# Adding metmast tower to the domain
my = np.asarray([-5,-0.5,0.5,5, -5])
mz =  np.asarray([0,H,H,0,0])
mx = np.asarray([450,445,450,455, 450])

#%% 

#%% plane slices after dX distance
px0, py0, pz0 = 0, 0, 115 # centre position for consideration
x_slices = np.arange(px0, X.max(), dX) + np.around(X.max() % dX, decimals=3) # k should match x_slices
t_slices = np.array(x_slices/Uhub/dt, dtype=int) 

#%% Selecting a grid point and extracting time series from the wind field at this point
idx_y = np.where(Y==0)[0] # refers to rotor center
idx_z = np.where(Z==115)[0] # refers to 116 m height
idx_x = t_slices-1

plt.plot(uk[0,:], 'k', label='uk')
plt.plot(u[idx_y, idx_z, idx_x], 'b', label= 'u')
plt.xlabel('Time [s]')
plt.ylabel('u comp [m/s]')
plt.legend()
plt.show()

plt.plot(vk[0,:], 'k', label='vk')
plt.plot(v[idx_y, idx_z, idx_x], 'b', label= 'v')
plt.xlabel('Time [s]')
plt.ylabel('v comp [m/s]')
plt.legend()
plt.show()

plt.plot(wk[0,:], 'k', label='wk')
plt.plot(w[idx_y, idx_z, idx_x], 'b', label= 'w')
plt.xlabel('Time [s]')
plt.ylabel('w comp [m/s]')
plt.legend()
plt.show()

#%% saving and loading the variables
if (saveArr == 1) & (skipGen==0):
	import shelve
	from datetime import date
	filename = r'../data/{0}ScanData_{1}x{2}_{3}m_{4}.out'.format(pattern, Ny, Nz, dX, date.today().strftime('%Y%m%d'))
	my_shelf = shelve.open(filename, 'n')
	for key in dir():
		try:
			my_shelf[key] = globals()[key]
		except:
			#
			# __builtins__, my_shelf, and imported modules can not be shelved.
			#
			print('ERROR shelving: {0}'.format(key))			
	my_shelf.close()

	import dill
	filename = r'../data/{0}ScanData_{1}x{2}_{3}m_{4}.pkl'.format(pattern, Ny, Nz, dX, date.today().strftime('%Y%m%d'))
	dill.dump_session(filename)

	# with open(r'../data/LidarScanData.npy','wb') as f:
	# 	np.save(f, uk)
	# 	np.save(f, vk)
	# 	np.save(f, wk)
	# 	np.save(f, u)
	# 	np.save(f, v)
	# 	np.save(f, w)
	print('[{0}]:windfield saved'.format(pa.now()) )
else:
	print('[{0}]:saving array is not selected'.format(pa.now()) )
	end_time = time.time()
	print('{0}: Completed script in {1} seconds'.format(time.time(), end_time-start_time))

# %% Extracting the length scales from the windfield, xLu shall match Lu
sys.path.append(r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\userModules")
from FnTurbLengthScale import *
expfn='excludeDecay'
xLu = np.zeros((len(Y), len(Z)))
xLv, xLw = xLu.copy(), xLu.copy()
for i in range(len(Y)):
	for j in range(len(Z)):
		xTu, xLu[i,j], lagx_uu, cor_uu = FnTurbLengthScale(u[i,j,:],dt,Uhub,expfn)
		xTv, xLv[i,j], lagx_vv, cor_vv = FnTurbLengthScale(v[i,j,:],dt,Uhub,expfn)
		xTw, xLw[i,j], lagx_ww, cor_ww = FnTurbLengthScale(w[i,j,:],dt,Uhub,expfn)

## Extracting length scales for the scan pattern
xT_up, xL_up, lagx_up, cor_up = FnTurbLengthScale(uk[0,:],dT,Uhub,expfn)
xT_vp, xL_vp, lagx_vp, cor_vp = FnTurbLengthScale(vk[0,:],dT,Uhub,expfn)
xT_wp, xL_wp, lagx_wp, cor_wp = FnTurbLengthScale(wk[0,:],dT,Uhub,expfn)


#%% Spectral Characteristics for the windfield
from scipy import signal
# extract spectrum from the complete wind field
psd_u=np.nan*np.ones((len(Y), len(Z), int(Ns/2)+1))
psd_v=np.nan*np.ones((len(Y), len(Z), int(Ns/2)+1))
psd_w=np.nan*np.ones((len(Y), len(Z), int(Ns/2)+1))

for i in np.arange(len(Y)):
	for j in np.arange(len(Z)):
		fu, psd_u[i, j, :] = signal.welch(u[i,j,:], 1/dt, window='hann', nperseg=Ns, scaling='density', detrend=False)
		fv, psd_v[i, j, :] = signal.welch(v[i,j,:], 1/dt, window='hann', nperseg=Ns, scaling='density', detrend=False)
		fw, psd_w[i, j, :] = signal.welch(w[i,j,:], 1/dt, window='hann', nperseg=Ns, scaling='density', detrend=False)

#  extract spectrum from the scanning pattern
fup, psd_up = signal.welch(uk[0,:], 1/dT, window='hann',nperseg= Ns, scaling='density', detrend=False)
fvp, psd_vp = signal.welch(vk[0,:], 1/dT, window='hann',nperseg= Ns, scaling='density', detrend=False)
fwp, psd_wp = signal.welch(wk[0,:], 1/dT, window='hann',nperseg= Ns, scaling='density', detrend=False)

# Spectra u-component
fig = plt.figure(figsize=(6,6))
ax = plt.gca()
plt.loglog(fu, (psd_u[idx_y, idx_z, :]).ravel(), label='mean $S_{uu}$ wind-field')
plt.loglog(fup, psd_up, label='Scan pattern $S_{uu}$')
plt.loglog(f, Su, label='theoretical Kaimal $S_{uu}$')
# plt.axis((10e-3, 10, 10e-9, 10e2 ))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spectra $S_{uu}(f)$ [$(m^2/s^2)/Hz$]')
plt.legend()
plt.tight_layout()
plt.show()

# Spectra v-component
fig = plt.figure(figsize=(6,6))
ax = plt.gca()
plt.loglog(fv, (psd_v[idx_y, idx_z, :]).ravel(), label='mean $S_{vv}$ wind-field')
plt.loglog(fvp, psd_vp, label='Scan pattern $S_{vv}$')
plt.loglog(f, Sv, label='theoretical Kaimal $S_{vv}$')
# plt.axis((10e-3, 10, 10e-9, 10e2 ))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spectra $S_{vv}(f)$ [$(m^2/s^2)/Hz$]')
plt.legend()
plt.tight_layout()
plt.show()

# # Spectra w-component
fig = plt.figure(figsize=(6,6))
ax = plt.gca()
plt.loglog(fw, (psd_w[idx_y, idx_z, :]).ravel(), label='mean $S_{ww}$ wind-field')
plt.loglog(fwp, psd_wp, label='Scan pattern $S_{ww}$')
plt.loglog(f, Sw, label='theoretical Kaimal $S_{ww}$')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spectra $S_{ww}(f)$ [$(m^2/s^2)/Hz$]')
# plt.axis((10e-3, 10, 10e-9, 10e2 ))
plt.legend()
plt.tight_layout()
plt.show()

sys.exit('Manual stop')
# %% plot figure
# https://stackoverflow.com/questions/28420504/adding-colors-to-a-3d-quiver-plot-in-matplotlib
if PlotFig==1:
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(xlg,ylg,zlg, color='r', marker='.', label='Lidar points')
	ax.plot3D(xrg,yrg,zrg,color='b',label='turbine')
	ax.plot3D(xtg,ytg,ztg, color='b')
	ax.scatter(k[:,0], k[:,1], k[:,2], color='g', label='nearest')
	# q = ax.quiver(k[:,0], k[:,1], k[:,2],uk,vk,wk,length=1,cmap='Reds', lw=10)
	# q.set_array(np.random.rand(np.prod(uk.shape)))
	#Labeling
	ax.set_xlabel('X Axes')
	ax.set_ylabel('Y Axes')
	ax.set_zlabel('Z Axes')
	# setting tick labels
	ax.xaxis.set_major_locator(plt.MaxNLocator(8))
	ax.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax.zaxis.set_major_locator(plt.MaxNLocator(8))
	# setting axis limits
	# ax.set_xticks([0,2,4,6,8,10])
	# ax.set_yticks([0,2,4,6,8,10])
	# ax.set_zticks([0,2,4,6,8,10])
	ax.axes.set_xlim3d(left=np.min(X), right=np.max(X)) 
	ax.axes.set_ylim3d(bottom=np.min(Y), top=np.max(Z)) 
	ax.axes.set_zlim3d(bottom=np.min(Z), top=np.max(Z)) 
	ax.legend()
	plt.draw()
	if SaveTex==1:
		tz.clean_figure()
		tz.save(r"c:\Users\giyash\ownCloud\IWES\Reports\tikz\Lissajous3D.tikz", float_format=".6f")
	plt.show()

# %%
## plot the contour plot
fig, ax = plt.subplots(constrained_layout=True)
uu,vv = np.meshgrid(uk,vk, sparse=True)
uv = np.sqrt(uu**2 + vv**2)
cs = ax.contourf( k[:,0], k[:,1],uv)
fig.colorbar(cs,ax=ax,shrink=0.9)
ax.set_title('contour plot')
plt.xlabel('x-axis [m](longitudinal)')
plt.ylabel('y-axis [m](lateral)')
ax.locator_params(nbins=4)
plt.show()

fig, ax = plt.subplots(constrained_layout=True)
uu,vv = np.meshgrid(uk, vk, sparse=False, indexing='xy')
uv = np.sqrt(uu**2 + vv**2)
cs = ax.contourf( k[:,1], k[:,2],uv)
fig.colorbar(cs,ax=ax,shrink=0.9)
ax.set_title('contour plot')
plt.xlabel('y-axis [m](lateral)')
plt.ylabel('z-axis [m](vertical)')
ax.locator_params(nbins=4)
plt.show()

