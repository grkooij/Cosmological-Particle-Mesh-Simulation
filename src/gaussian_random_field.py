import numpy as np
from numba import njit
import pyfftw
from configure_me import LCDM_TRANSFER_FUNCTION, RANDOM_SEED, A_INIT, OMEGA_LAMBDA0, OMEGA_K0
from configure_me import POWER, H0, OMEGA_B0, OMEGA_M0, N_PARTS, BOX_SIZE, N_CPU

from cosmology import Dt

def gaussian_random_field():
	#This function generates a Gaussian Random Field for any power spectrum at a given
	# epoch through Dt
	print('Preparing the Gaussian random field...')
	lin_growth_factor = Dt(A_INIT, [OMEGA_M0, OMEGA_LAMBDA0, OMEGA_K0])

	f1, f2 = gaussian_random_numbers()
	

	p = power_spectrum()

	#Multiplying by the sqrt of the power spectrum and linear growth factor
	f1 = np.sqrt(p*lin_growth_factor**2) * f1
	f2 = np.sqrt(p*lin_growth_factor**2) * f2

	rho_k = f1 + 1j*f2

	#Real space density field
	fft_grid = np.zeros([N_PARTS, N_PARTS, N_PARTS], dtype='cfloat')
	ifft_object = pyfftw.FFTW(rho_k, fft_grid, direction = 'FFTW_BACKWARD',axes=(0,1,2), threads = N_CPU)
	return (ifft_object().real).astype('float32')

@njit(parallel = True)
def gaussian_random_numbers():

	#Generate two sets of uniform, independent random values for a given seed
	np.random.seed(RANDOM_SEED)
	u = np.zeros((N_PARTS**3), dtype = 'float32')
	v = np.zeros((N_PARTS**3), dtype = 'float32')
	
	n_parts_accepted = 0
	
	while n_parts_accepted < N_PARTS**3:
		u1 = np.random.uniform(-1,1, N_PARTS**3-n_parts_accepted)
		v1 = np.random.uniform(-1,1, N_PARTS**3-n_parts_accepted)
		sq = u1**2 + v1**2
		u1 = u1[np.nonzero((0. < sq) & (sq < 1.))]
		v1 = v1[np.nonzero((0. < sq) & (sq < 1.))]

		u[n_parts_accepted:n_parts_accepted+len(u1)] = u1
		v[n_parts_accepted:n_parts_accepted+len(v1)] = v1
		n_parts_accepted += len(u1)
			
	#Transformation from uniform to Gaussian Random numbers
	s = u**2+v**2

	u = np.reshape(u, (N_PARTS, N_PARTS, N_PARTS))
	v = np.reshape(v, (N_PARTS, N_PARTS, N_PARTS))
	s = np.reshape(s, (N_PARTS, N_PARTS, N_PARTS))

	#Polar Box-Muller transform
	f1 = u*(-2*np.log(s)/s)**0.5
	f2 = v*(-2*np.log(s)/s)**0.5

	return f1, f2

def lcdm_transfer_function(k_grid):

	#Transfer function for LCDM
	Gamma = OMEGA_M0 * H0 * np.exp(-OMEGA_B0 - OMEGA_B0/OMEGA_M0)
	q = k_grid / Gamma
	factor1 = np.sqrt(1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)
	factor2 = np.divide(np.log(1 + 2.34*q)**2, (2.34*q)**2, where=q!=0)
	lcdm = factor2 / factor1

	return lcdm

def fourier_grid():

	#Finding the fourier frequency axes  
	scale = 2*np.pi*N_PARTS/BOX_SIZE
	lxaxis = scale*np.fft.fftfreq(N_PARTS)
	lyaxis = scale*np.fft.fftfreq(N_PARTS)
	lzaxis = scale*np.fft.fftfreq(N_PARTS)
	
	#Make a Fourier grid
	lz, ly, lx = np.meshgrid(lzaxis, lyaxis, lxaxis, indexing='ij')   
	
	k_grid = np.sqrt(lx**2 + ly**2 + lz**2)

	return k_grid

def power_spectrum():

	k_grid = fourier_grid()

	if LCDM_TRANSFER_FUNCTION:
		lcdm = lcdm_transfer_function(k_grid)
	else:
		lcdm = 0.

	#Number of pixels used for normalisation
	Npix = N_PARTS**3

	#Variance of density fluctuations
	sigma2fluxt = 64*H0**2

	#Calculating the power spectrum
	if POWER >= 0.:
		if LCDM_TRANSFER_FUNCTION:
			summ = np.sum(np.power(k_grid, POWER, where=k_grid!=0)*lcdm)
			A = sigma2fluxt*Npix**2/summ
			p = A*(k_grid)**POWER*lcdm
		else:
			summ = np.sum(np.power(k_grid, POWER, where=k_grid!=0))
			A = sigma2fluxt*Npix**2/summ
			p = A*(k_grid)**POWER
	else:
		kgrid_inverse = np.power(k_grid,-POWER,where=k_grid!=0)
		div_kgrid = np.divide(1, kgrid_inverse, where=kgrid_inverse!=0)
		summ = np.sum(div_kgrid)
		A = sigma2fluxt*Npix**2/summ
		p = A*div_kgrid

	return p