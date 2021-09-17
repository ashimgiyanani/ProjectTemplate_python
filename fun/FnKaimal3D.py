def FnKaimal3D(xg,yg,zg,Nfft,T,dt,H,z0,Uhub,sigmau,lat, **kwargs):
# Script to generate a synthetic wind according to Kaimal spectrum and Veers method in 1D
# kaimal1d
# Function to determine the Kaimal spectrum in 1 dimension
# optional - Kaimal general / ESDU1975 / DS1972 / IEC1999 / Eurocode1997
# reference - Wind energy handbook
# Matlab file reference -> FnKaimal1d

# Inputs required
# T, Nfft, Uhub, H

	import sys
	sys.path.append(r"../fun")
	import importlib 
	import os
	import pandas as pd
	import numpy as np
	import scipy as sp
	import matplotlib.pyplot as plt 
	import pythonAssist as pa # definitions useful in python in general
	import gc
	from numpy import complex64, float64
	from numpy.core.fromnumeric import searchsorted

	# there you go: use os, signal, ... from whatever place of the module
	import numpy.matlib
	import scipy.linalg
	import matlab2py as m2p # includes some definitions helping matlab users 
	from scipy.spatial.distance import cdist # euclidean distance between grid points
	from importlib import reload # reload a session on ipython
	import matlab2py as m2p
	import time
	start_time = time.time()

	# define grid dimensions
	dth = T*Uhub/Nfft                                              # distances vector based on mean wind speed [m]
	t = np.linspace(dt,T,int(T/dt))                                       # time vector [s]
	xr = t*Uhub                                                   # distance vector for calculation Lidar
	y,z = np.meshgrid(yg,zg, sparse=False, indexing='xy') # order changed to match meshgrid in Matlab
	YZ = np.matrix([y.flatten(), z.flatten()]).T
	dr = cdist(YZ,YZ, metric='euclidean')
	dx = np.abs(xg[1]-xg[0])
	dy = np.abs(yg[1]-yg[0])
	dz = np.abs(zg[1]-zg[0])
	Nt =  len(yg)*len(zg) # total np. of grid points
	try:
		seed
		np.random.seed(seed)
	except NameError:
		print('Random seed is selected')
		
	# Initializations, most values taken from Wind energy handbook, Burton et al. 2001, ESDU 85020 and ESDU 86010
	df = 1/T                                                           # discrete freq [Hz]
	Fs = 1/dt                                                          # sampling frequency [Hz]
	Fn = Fs/2                                                          # nyquist freq [Hz]
	f = np.linspace(1,int(Nfft/2),int(Nfft/2))*df                    # freq vector one-sided [Hz]
	kw = 2*np.pi*f/Uhub                                                   # wavenumbers vector

	# Calcuation of sigmau based on lattitude, friction vel and surface roughness, if not from measurements
	omega = 7.29e-05                        # angular rotation of the earth [s^-1]
	fc = 2*omega*np.sin(np.deg2rad(lat))    # Coriolis force
	fricU = (0.4*Uhub - (34.5*fc*H))/np.log(H/z0)  # friction  velocity assuming neutral conditions
	h = fricU/(6*fc)     # height of the boundary layer
	eta = 1-6*fc*H/fricU
	p = eta**16
	try:
		sigmau
	except NameError:
		sigmau = (7.5*eta*(0.538+0.09*np.log(H/z0))**p)*fricU/(1+0.156*np.log(fricU/(fc*z0)))    # std deviation, longitudinal from friction vel and surface roughness wrt ref height


	# Turbulence intensity in u,v,w components
	try:
		sigmav, sigmaw
	except NameError:
		sigmav = 0.8*sigmau
		sigmaw = 0.5*sigmau

	Iu = (sigmau/Uhub)*100                                               # turbulence intensity, longitudinal
	Iv = (sigmav/Uhub)*100
	Iw = (sigmaw/Uhub)*100
	# Iv = Iu*(1-0.22*(np.cos(np.pi*H/(2*h)))**4)                                # TI, lateral
	# Iw = Iu*(1 - 0.45*(np.cos(np.pi*H/(2*h)))**4)                              # TI vertical
	# sigmav = Iv*Uhub/100                                                   # std dev lateral
	# sigmaw = Iw*Uhub/100                                                   # std dev vertical

	A = 0.115*(1 + 0.315*((1-H/h)**6))**(2/3) # Kaimal spectrum parameter in (ESDU,1985) large heights ~ 0.115, close to ground ~ 0.142
	alpha = 0.535+2.76*((0.138-A)**0.68)                                 #
	beta1 = 2.357*alpha-0.761
	beta2 = 1-beta1
	#Vz = 2.5*fricU*(log(H/z0)+5.75*H/h-1.875*(H/h)**2 ...               # wins speed hub height
	#       -1.333*(H/h)**3 +  0.25*(H/h)**4);


	# Determine the longitudinal length scale
	try:
		xLu
	except NameError:
		Ro = fricU/(fc*z0)                                                 # surface Rossby no.    u*/fc*z0
		B = 24*Ro**0.155
		N = 1.24*(Ro**0.008)                                                #
		K0 = 0.39/(Ro**0.11)                                                # value for Kz for z->0
		Kz = 0.19 - ((0.19-K0)*np.exp(-B*((H/h)**N)))                        # Kolmogorov parameter
		# Kinf = 0.188                                                     # old model ESDU 1975
		# zc = 0.39*h*((fricU/(fc*z0))**(-1/8))                             # old model ESDU 1975
		# Kz = Kinf*(1-(1-H/zc)**2)**(1/2)                                 # old model ESDU 1975
		xLu = (A**(3/2)*((sigmau/fricU)**3)*H)/ \
			 (2.5*(Kz**(3/2))*((1-H/h)**2)*(1 + 5.75*H/h))               # length scale u direction

	xLv = 0.5*xLu*((sigmav/sigmau)**3)                                 # length scale lateral, existence of isotropty in high freq regions, (ESDU, 1985)
	xLw = 0.5*xLu*((sigmaw/sigmau)**3)                                 # length scale vertical, existence of isotropty in high freq regions, (ESDU, 1985)
	yLu = 0.5*xLu*(1 - 0.46*np.exp(-35*((H/h)**1.7)))                     # source ??
	yLv = 2*yLu*((sigmav/sigmau)**3)
	yLw = yLu*((sigmaw/sigmau)**3)
	zLu = 0.5*xLu*(1 - 0.68*np.exp(-35*((H/h)**1.7)))                     # source ??
	zLv = zLu*((sigmav/sigmau)**3)
	zLw = 2*zLu*((sigmaw/sigmau)**3)

	fu = min(1.0, 0.04*(fc**(-2/3)))

	# mean length scales 
	Lu = np.sqrt(((yLu*dy)**2 + (zLu*dz)**2)/(dy**2+dz**2))
	Lv = np.sqrt(((xLv*dx)**2 + (zLv*dz)**2)/(dx**2+dz**2))
	Lw = np.sqrt(((xLw*dx)**2 + (yLw*dy)**2)/(dx**2+dy**2))

	# wavenumber domain
	nu = f*(xLu/Uhub)
	nv = f*(xLv/Uhub)
	nw = f*(xLw/Uhub)
	F1 = 1 + 0.455*np.exp( - (0.760*nu)/(alpha**(-0.8)))           # Gauss hypergeometric fn used in psd,u estimate
	F2 = 1 + 2.88 *np.exp( - (0.218*nv)/(alpha**(-0.9)))           # "", v estimate
	F3 = 1 + 2.88 *np.exp( - (0.218*nw)/(alpha**(-0.9)))           # "", w estimate

	# Generate the random phases between 0 and 2pi as in Veer's simulation
	mu = np.array(np.exp(1j*2*np.pi*np.matlib.rand(Nt,int(Nfft/2))))
	mv = np.array(np.exp(1j*2*np.pi*np.matlib.rand(Nt,int(Nfft/2))))
	gc.collect()
	mw = np.array(np.exp(1j*2*np.pi*np.matlib.rand(Nt,int(Nfft/2))))


	# Power spectral densities from ESDU1985 (Characteristics of atmospheric turbulence near the ground. Part II, Single point data for strong winds (neutral atmosphere))
	# power spectral density, u comp
	Su = ((sigmau**2)/f) * ( beta1*(2.987*nu/alpha)/ \
							 ((1+(2*np.pi*nu/alpha)**2)**(5/6))+ \
							 (F1*beta2*((1.294*nu/alpha)/((1+(np.pi*nu/alpha)**2)**(5/6)))) ) 
	# power spectral density, v comp
	Sv = ((sigmav**2)/f) * ( beta1*(2.987*(1+(8/3)*(4*np.pi*nv/alpha)**2)*(nv/alpha))/ \
							 ((1+(4*np.pi*nv/alpha)**2)**(11/6)) +  \
							 (F2*beta2*((1.294*nv/alpha) / ((1+(2*np.pi*nv/alpha)**2)**(5/6)) ) ) )   # psd, v comp
	# power spectral density, w comp
	Sw = ((sigmaw**2)/f)*(beta1*(2.987*(1+(8/3)*(4*np.pi*nw/alpha)**2)*(nw/alpha))/ \
						  ((1+(4*np.pi*nw/alpha)**2)**(11/6)) +  \
						  (F3*beta2*((1.294*nw/alpha) /  \
									 ((1+(2*np.pi*nw/alpha)**2)**(5/6)))))                           # psd, w comp

	# Coherence according to IEC definition
	a = 12                                                           # coherence decrement or decay, Turbsim=12(IEC)
	Lc = 5.67*min(30,H)                                              # coherence scale parameter
	Hu = np.array(np.empty((Nt,int(f.size))), dtype=complex64)
	Hv = np.array(np.empty((Nt,int(f.size))), dtype=complex64)
	Hw = np.array(np.empty((Nt,int(f.size))), dtype=complex64)
	theta = np.empty((Nt, int(f.size)))
	C = np.empty((Nt, Nt))
	C[:] = np.nan

	# Decaying coherence according to Kristensen and Bossanyi (Bossanyi2012a)
	def unfreezeTurb(Iu, Iv, Iw, Uhub, dr, Li,f,):
	# Input:
		# Iu, Iv, Iw - turbulence intensity in u, v and w comp. of wind
		# Uhub - mean wind speed
		# dr - distance np.matrix
		# Li - integral length scale, i refers to u,v and w components
		# f - frequency vector
	# Output: 
		# fi - decay fraction used in Bossany-Kristensen unfrozen turbulence model to decorrelate

		sig = Uhub * np.sqrt( Iu**2 + Iv**2 + Iw**2)/100
		Alpha = (sig * dr) / (Uhub * Li)
		eta = f*Li/Uhub
		G = eta**2 * (eta + 1/121)**0.5  / (eta + 1/33)**11/6
		fi = np.exp(-Alpha*G) * ( 1 - np.exp(-1 / (2*Alpha*np.min(Alpha,1)*eta**2) ) )
		return fi

	# Cx = np.exp( -a*dr*np.sqrt((f/Uhub)**2 + (0.12/Lc)**2) + )

	def choleskyDecompose(a, dr, f, Uhub, Lc, Sx,mx, T):
		Cx = np.exp(-a*dr*np.sqrt((f/Uhub)**2 + (0.12/Lc)**2))      # Coherence in Turbsim, IEC guidelines
		Cxtemp = sp.linalg.cholesky(Cx,lower=True)*np.sqrt(Sx/T)    # Cholesky decomposition
		Hx = Cxtemp.dot(mx)                  # multiplying amplitude with the random phases
		theta = np.arctan(np.imag(Hx)/np.real(Hx))
		return Hx, theta

	# need to optimize this section, due to memory and speed problems, currently a memory solution
	# Using Dask
	# from dask.distributed import Client
	# client = Client(n_workers=6)

	# futures = []

	# for i in np.arange(0, f.size):
	# 	future = client.submit(choleskyDecompose, a, dr, f, Uhub, Lc, Su, T, i)
	# 	futures.append(future)
	
	# results = client.gather(futures)
	# client.close()

	# Using joblib
	# from joblib import Parallel, delayed
	# element_run = Parallel(n_jobs=-1)(delayed(choleskyDecompose)(a, dr,f, Uhub, Lc, Su,mu, T, i) for i in np.arange(0,f.size))

	# # using asyncio 
	# import asyncio

	# def background(f):
	# 	def wrapped(*args, **kwargs):
	# 		return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

	# 	return wrapped

	# @background
	# def choleskyDecompose(a, dr,f, Uhub, Lc, Su,mu, T):
	# 	C = np.exp(-a*dr*np.sqrt((f/Uhub)**2 + (0.12/Lc)**2))      # Coherence in Turbsim, IEC guidelines
	# 	Cutemp = sp.linalg.cholesky(C,lower=True)*np.sqrt(Su/T)    # Cholesky decomposition
	# 	Hu = Cutemp.dot(mu)                                  # multiplying amplitude with the random phases
	# 	return Hu

	# for i in np.arange(0,f.size):
	# 	Hu[:,i] = choleskyDecompose(a, dr,f[i], Uhub, Lc, Su[i],mu[:i], T, i)

	# using numba
	# from numba import jit, prange

	# @jit
	# def choleskyDecompose(a, dr,f, Uhub, Lc, Su,mu, T):
	# 	for i in prange(f.size):
	# 		C = np.exp(-a*dr*np.sqrt((f[i]/Uhub)**2 + (0.12/Lc)**2))      # Coherence in Turbsim, IEC guidelines
	# 		Cutemp = sp.linalg.cholesky(C,lower=True)*np.sqrt(Su[i]/T)    # Cholesky decomposition
	# 		Hu[:,i] = Cutemp.dot(mu[:,i])                                  # multiplying amplitude with the random phases

	# 	return Hu

	for i in np.arange(0,f.size):
		Hu[:,i],theta[:,i] = choleskyDecompose(a, dr, f[i], Uhub, Lc, Su[i], mu[:,i], T)
		Hv[:,i],_ = choleskyDecompose(a, dr, f[i], Uhub, Lc, Sv[i], mv[:,i], T)
		Hw[:,i],_ = choleskyDecompose(a, dr, f[i], Uhub, Lc, Sw[i], mw[:,i], T)

	# pad the Hermitian matrix and rotate the Hermitian matrix  to get real H matrix
	Hu = np.hstack((np.zeros((Nt,1)), Hu, np.fliplr(np.conj(Hu[:,0:-1]))))                    #
	Hv = np.hstack((np.zeros((Nt,1)), Hv, np.fliplr(np.conj(Hv[:,0:-1]))))                    #
	Hw = np.hstack((np.zeros((Nt,1)), Hw, np.fliplr(np.conj(Hw[:,0:-1]))))                    #

	Hu[:,int(Nfft/2)] = np.real(Hu[:,int(Nfft/2)])
	Hv[:,int(Nfft/2)] = np.real(Hv[:,int(Nfft/2)])
	Hw[:,int(Nfft/2)] = np.real(Hw[:,int(Nfft/2)])

	u = Nt*np.real(np.fft.ifft(Hu,axis=1))
	v = Nt*np.real(np.fft.ifft(Hv,axis=1))
	w = Nt*np.real(np.fft.ifft(Hw,axis=1))
	del Hu, Hv, Hw
	gc.collect()

	# reshaping the time series in grid dimensions
	u = np.reshape(u,(len(yg),len(zg),Nfft),order='F')
	v = np.reshape(v,(len(yg),len(zg),Nfft),order='F')
	w = np.reshape(w,(len(yg),len(zg),Nfft),order='F')

# extracting U, V and W components of wind speed Uhub^2 = U^2 + V^2 + W^2
# https://github.com/old-NWTC/TurbSim/blob/master/Source/TSsubs.f90
  #   try:
  #       thetaH
  #   except NameError:
  #       C = 2*np.pi*np.ones(np.shape(u))
  #       thetaH = (360/C)*np.arctan2(v,u) + 180
  #       thetaV =(360/C)*np.arctan2(w,u) + 180
 
  #   Uc = Uhub + u
 	# U = Uc*np.cos(thetaH)*np.sin(thetaV) - v*np.sin(thetaH) - w*np.cos(thetaH)*np.sin(thetaV)
 	# V = Uc*np.sin(thetaH)*np.cos(thetaV) + v*np.cos(thetaH) - w*np.sin(thetaH)*np.sin(thetaV)
 	# W = Uc*np.sin(thetaV) + w*np.cos(thetaV)

	# scaling the time series
	u = u * sigmau/np.std(u.flatten())
	v = v * sigmav/np.std(v.flatten())
	w = w * sigmaw/np.std(w.flatten())

	end_time = time.time()
	print('[{0}]: Completed script in {1} seconds'.format(end_time, end_time-start_time))
	
	return u, v, w, Su, Sv, Sw, xr, Lu, Lv, Lw, f

# Example:
import numpy as np
xg = np.linspace(-200, 200,3) # longitudinal dir, (row vector), [m]
yg = np.linspace(-100, 100,3) # lateral direction, (row vector), [m]
zg = np.linspace(0, 200,3) # gridvertical direction (row vector), [m]
T = 600 # total time [s]
Nfft = np.power(2,9) # no. of fft terms, always in power of 2 [-]
dt = T/Nfft                                                        # discrete time [s]
Uhub = 10 # hubheight wind speed [m/s]
H = 100 # hub height [m]
z0 = 0.1                   # roughness length [m]
lat = 52.4                 # lattiutude of measurement [deg]
sigmau = 1               # sigma from measurements, if available
u, v, w, Su, Sv, Sw, xr, Lu, Lv, Lw, f =  FnKaimal3D(xg,yg,zg,Nfft,T,dt,H,z0,Uhub,sigmau,lat)
