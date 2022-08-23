# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# Tasks:
# compare a time series at a point in the grid vs the time series in scanning pattern
# Compare the spectrum
# Compare the std dev
# Compare the length scales affecting

# ToDos:
# save the wind fields to pkl files in a pandas dataframe for different mean wind speeds i.e. 5, 10, 15, 20, 25, 30, 30
# Plot the Uhub*Sxx Vs f/Uhub plot and compare with Taylor's  result.
# plot the autocorrelation function for different wind speeds 

# On decorrelation of simulated frozen turbulence time series
# the decay factor a decreases according to radioactive decay properties \
# 	as the sampling freuquency decreases. This effect is implemented in the\
#  		Cholesky decomposition
# frozen turbulence observed until 50ms

# side task:
# understand pycontrub - custom wind speed and sig profiles

# %%
# import modules
%matplotlib inline
import numpy as np
import scipy as sp
import sys

from scipy.spatial import cKDTree
sys.path.insert(0,r"../../userModules")

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
input = pa.struct()
input.PlotFig = 1
input.SaveTex = 0
input.SaveFig = 0
input.saveArr = 0 # load array from pickle (saveArr=0), save array to pickle (saveArr=1)
input.skipGen = 0 # save uk, vk and wk to pickle (skipGen=0), (load from pickle skipGen=1)
input.TurbModel = 'Kaimal' # options 'Kaimal' or 'Mann'
input.pattern = 'bowtie'

wind=pa.struct()
wind.Uhub = 5 # hubheight wind speed [m/s]
wind.H = 115 # hub height [m]
wind.z0 = 0.1                   # roughness length [m]
wind.lat = 52.4                 # lattiutude of measurement [deg]
wind.Nx, wind.Ny, wind.Nz = 3, 5, 5 
wind.X = np.linspace(0, 400, wind.Nx) # longitudinal dir, (row vector), [m] 
wind.Y = np.linspace(-100, 100,wind.Ny) # lateral direction, (row vector), [m]
wind.Z = np.linspace(0, 230, wind.Nz) # grid vertical direction (row vector), [m]
wind.dx, wind.dy, wind.dz = np.diff(wind.X)[0], np.diff(wind.Y)[0], np.diff(wind.Z)[0] 
wind.dt = 0.001  # discrete sampling time [s]
wind.dT = 0.1 # cycle time to complete one scanning pattern
# Np = int(dT/dt) # number of sampling points in one scanning pattern
# dX = dT*Uhub # step in longitudinal direction after 1 scanning pattern
wind.dX = wind.dt*wind.Uhub # step in longitudinal direction

try:
	wind.dt
	wind.Nfft = np.power(2, 12)
	wind.T = wind.Nfft*wind.dt # total time [s]
except NameError:
	wind.dt = wind.dx/wind.Uhub
	wind.T = 600 # total simulation time
	wind.Nt = wind.T/wind.dt
	pow = np.ceil(np.log2(abs(wind.Nt)))
	wind.Nfft = int(np.power(2,pow)) # no. of fft terms, always in power of 2 [-]

wind.Fs = 1/wind.dt
wind.t = np.linspace(wind.dt,wind.T,wind.Nfft)
wind.X = (wind.t*wind.Uhub).round(decimals=3)
wind.fs = 1/wind.t
wind.sigmau = 1               # sigma from measurements, if available
wind.sigmav = 0.8*wind.sigmau
wind.sigmaw = 0.5*wind.sigmau
if input.TurbModel == 'Kaimal':
	wind.u, wind.v, wind.w, wind.Su, wind.Sv, wind.Sw, wind.xr, wind.Lu, wind.Lv, wind.Lw, wind.f =  \
		FnKaimal3D(wind.X,wind.Y,wind.Z,wind.Nfft, wind.T,wind.dt,wind.H,wind.z0, wind.Uhub, \
			wind.sigmau, wind.lat, seed=858585) 
elif input.TurbModel == 'Mann':
	from FnMannSpec import FnMannSpec
	wind.L = 30 # [m] length scale parameter, unsheared, isotropic length parameter
	wind.ae = 1.453 # ref: Veldkamp2006, Appendix C
	wind.Gamma = 3.9 # [-] non-dimensional shear distortion (Anisotropy) parameter
	wind.sigma = [wind.sigmau, wind.sigmav, wind.sigmaw]
	wind.u,wind.v,wind.w,wind.Su,wind.Sv,wind.Sw,wind.Fu,wind.Fv,wind.Fw,wind.kw,wind.f = \
			FnMannSpec(wind.H,wind.T,wind.Uhub,wind.Nfft,wind.X,wind.Y,wind.Z,\
					wind.Gamma, wind.L, wind.ae, wind.sigma)
wind.U = wind.Uhub + wind.u
wind.Iu = wind.sigmau/wind.Uhub
wind.Iv = wind.sigmav/wind.Uhub
wind.Iw = wind.sigmaw/wind.Uhub
print('{}: Kaimal 3D Wind generated'.format(pa.now()))

#%% generate turbulent wind fields from Turbsim

#%% generate turbulent wind fields from pyconturb
from pyconturb import gen_turb, gen_spat_grid  # generate turbulence, useful helper
from pyconturb.sig_models import iec_sig  # IEC 61400-1 turbulence std dev
from pyconturb.spectral_models import kaimal_spectrum  # Kaimal spectrum
from pyconturb.wind_profiles import constant_profile, power_profile  # wind-speed profile functions
from _nb_utils import plot_slice

# generate domain for turbulence generation
# wind.X = np.linspace(0, 400, wind.Nx) # longitudinal dir, (row vector), [m] 
spat_df = gen_spat_grid(wind.Y, wind.Z+10)
# specify mean wind speed profile function (constant_profile or power_profile)
wsp = power_profile
# specify standard deviation profile function
sig = iec_sig
# specify power spectrum profile function
spec = kaimal_spectrum
turb_df = gen_turb(spat_df, T=wind.T, dt=wind.dt, u_ref=wind.Uhub, wsp_func=wsp, \
					 sig_func=sig, spec_fun=spec)

t = 0
ax = plot_slice(spat_df, turb_df, val=t)
ax.set_title(f'Turbulence slice at t = {t}');

#%% Include custom wind speed and sig profiles
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

#%% plane slices after dX distance
px0, py0, pz0 = 0, 0, 115 # centre position for consideration

x_slices = np.arange(px0, wind.X.max(), wind.dX) + np.around(wind.X.max() % wind.dX, decimals=3) # k should match x_slices
t_slices = np.array(x_slices/wind.Uhub/wind.dt, dtype=int) 
Ns = len(x_slices)
Np = wind.Ny*wind.Nz
uk = np.zeros((Np, Ns))
vk = np.zeros((Np, Ns))
wk = np.zeros((Np, Ns))
ZZ = np.zeros((Np, Ns), dtype=np.int32)
yg, zg = np.asarray(py0, dtype=np.float16), np.asarray(pz0, dtype=np.float16)
k = np.zeros((Np, 3, Ns))
# iterate over slices
from FnFindNearestGridPts import FnFindNearestGridPts
for ix in range(Ns):
	XG = np.asarray(x_slices[ix]*np.ones(np.asarray(px0).shape), dtype=np.float16)
	Xd, Yd, Zd, pos, ZZ, k[:,:,ix] = FnFindNearestGridPts(wind.X, wind.Y, wind.Z, XG, yg, zg)
	uk[:,ix] = wind.u.flatten()[ZZ]
	vk[:,ix] = wind.v.flatten()[ZZ]
	wk[:,ix] = wind.w.flatten()[ZZ]
print('[{0}]: Wind speed at nearest grid points is assigned'.format(pa.now()))

#%% Selecting a grid point and extracting time series from the wind field at this point
idx_y = np.where(wind.Y==0)[0] # refers to rotor center
idx_z = np.where(wind.Z==115)[0] # refers to 116 m height
idx_x = t_slices-1

plt.plot(uk[0,:], 'k', label='uk')
plt.plot(wind.u[idx_y, idx_z, idx_x], 'b', label= 'u')
plt.xlabel('Time [s]')
plt.ylabel('u comp [m/s]')
plt.legend()
plt.show()

plt.plot(vk[0,:], 'k', label='vk')
plt.plot(wind.v[idx_y, idx_z, idx_x], 'b', label= 'v')
plt.xlabel('Time [s]')
plt.ylabel('v comp [m/s]')
plt.legend()
plt.show()

plt.plot(wk[0,:], 'k', label='wk')
plt.plot(wind.w[idx_y, idx_z, idx_x], 'b', label= 'w')
plt.xlabel('Time [s]')
plt.ylabel('w comp [m/s]')
plt.legend()
plt.show()

#%% saving and loading the variables
if (input.saveArr == 1) & (input.skipGen==0):
	import shelve
	from datetime import date
	filename = r'../data/{0}ScanData_{1}x{2}_{3}m_{4}.out'.format(input.pattern, wind.Ny, wind.Nz, wind.dX, date.today().strftime('%Y%m%d'))
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
	filename = r'../data/{0}ScanData_{1}x{2}_{3}m_{4}.pkl'.format(input.pattern, wind.Ny, wind.Nz, wind.dX, date.today().strftime('%Y%m%d'))
	dill.dump_session(filename)
	print('[{0}]:windfield saved'.format(pa.now()) )
else:
	print('[{0}]:saving array is not selected'.format(pa.now()) )
	end_time = time.time()
	print('{0}: Completed script in {1} seconds'.format(time.time(), end_time-start_time))

# %% Extracting the length scales from the windfield, xLu shall match Lu
sys.path.append(r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\userModules")
from FnTurbLengthScale import *
expfn='includeDecay'
xLu = np.zeros((len(wind.Y), len(wind.Z)))
xLv, xLw = xLu.copy(), xLu.copy()
for i in range(len(wind.Y)):
	for j in range(len(wind.Z)):
		xTu, xLu[i,j], lagx_uu, cor_uu = FnTurbLengthScale(wind.u[i,j,:],wind.dt,wind.Uhub,expfn)
		xTv, xLv[i,j], lagx_vv, cor_vv = FnTurbLengthScale(wind.v[i,j,:],wind.dt,wind.Uhub,expfn)
		xTw, xLw[i,j], lagx_ww, cor_ww = FnTurbLengthScale(wind.w[i,j,:],wind.dt,wind.Uhub,expfn)

## Extracting length scales for the scan pattern
xT_up, xL_up, lagx_up, cor_up = FnTurbLengthScale(uk[0,:],wind.dT,wind.Uhub,expfn)
xT_vp, xL_vp, lagx_vp, cor_vp = FnTurbLengthScale(vk[0,:],wind.dT,wind.Uhub,expfn)
xT_wp, xL_wp, lagx_wp, cor_wp = FnTurbLengthScale(wk[0,:],wind.dT,wind.Uhub,expfn)

#%% Spectral Characteristics for the windfield
from scipy import signal
# extract spectrum from the complete wind field
psd_u=np.nan*np.ones((len(wind.Y), len(wind.Z), int(Ns/2)+1))
psd_v=np.nan*np.ones((len(wind.Y), len(wind.Z), int(Ns/2)+1))
psd_w=np.nan*np.ones((len(wind.Y), len(wind.Z), int(Ns/2)+1))

for i in np.arange(len(wind.Y)):
	for j in np.arange(len(wind.Z)):
		fu, psd_u[i, j, :] = signal.welch(wind.u[i,j,:], 1/wind.dt, window='hamming', nperseg=Ns, scaling='spectrum', detrend=False)
		fv, psd_v[i, j, :] = signal.welch(wind.v[i,j,:], 1/wind.dt, window='hamming', nperseg=Ns, scaling='spectrum', detrend=False)
		fw, psd_w[i, j, :] = signal.welch(wind.w[i,j,:], 1/wind.dt, window='hamming', nperseg=Ns, scaling='spectrum', detrend=False)

#  extract spectrum from the scanning pattern
fup, psd_up = signal.welch(uk[0,:], 1/wind.dT, window='hamming',nperseg= Ns, scaling='spectrum', detrend=False)
fvp, psd_vp = signal.welch(vk[0,:], 1/wind.dT, window='hamming',nperseg= Ns, scaling='spectrum', detrend=False)
fwp, psd_wp = signal.welch(wk[0,:], 1/wind.dT, window='hamming',nperseg= Ns, scaling='spectrum', detrend=False)

# Spectra u-component
fig = plt.figure(figsize=(6,6))
ax = plt.gca()
plt.loglog(fu/wind.Uhub, wind.Uhub*(psd_u[idx_y, idx_z, :]).ravel(), label='mean $S_{uu}$ wind-field')
# plt.loglog(fup, psd_up, label='Scan pattern $S_{uu}$')
plt.loglog(wind.f/wind.Uhub, wind.Uhub*wind.Su, label='theoretical Kaimal $S_{uu}$')
plt.axis((0, 2, 0, 5 ))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spectra $S_{uu}(f)$ [$(m^2/s^2)/Hz$]')
plt.legend()
plt.tight_layout()
plt.show()

# Spectra v-component
fig = plt.figure(figsize=(6,6))
ax = plt.gca()
plt.loglog(fv/wind.Uhub, wind.Uhub*(psd_v[idx_y, idx_z, :]).ravel(), label='mean $S_{vv}$ wind-field')
# plt.loglog(fvp, psd_vp, label='Scan pattern $S_{vv}$')
plt.loglog(wind.f/wind.Uhub, wind.Uhub*wind.Sv, label='theoretical Kaimal $S_{vv}$')
# plt.axis((10e-3, 10, 10e-9, 10e2 ))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spectra $S_{vv}(f)$ [$(m^2/s^2)/Hz$]')
plt.legend()
plt.tight_layout()
plt.show()

# # Spectra w-component
fig = plt.figure(figsize=(6,6))
ax = plt.gca()
plt.loglog(fw/wind.Uhub, wind.Uhub*(psd_w[idx_y, idx_z, :]).ravel(), label='mean $S_{ww}$ wind-field')
# plt.loglog(fwp, psd_wp, label='Scan pattern $S_{ww}$')
plt.loglog(wind.f/wind.Uhub, wind.Uhub*wind.Sw, label='theoretical Kaimal $S_{ww}$')
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

