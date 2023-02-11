import numpy as np
import random

def gaussian_random_field(box, cosm):
	#This function generates a Gaussian Random Field for a LCDM universe at a
	#given epoch through Dt
	Dt = cosm.Dt

	f1, f2 = gaussian_random_numbers(box)
	p = power_spectrum(box, cosm, lcdm_transfer_function(box, cosm))

	#Multiplying by the sqrt of the power spectrum and linear growth factor
	f1 = np.sqrt(p*Dt**2) * f1
	f2 = np.sqrt(p*Dt**2) * f2
	
	#Real space density field
	return np.fft.ifftn(f1 + 1j*f2).real

def gaussian_random_numbers(box):
	Npart = box.Npart
	seed = box.seed

	#Generate two sets of uniform, independent random values for a given seed
	np.random.seed(seed)
	u = np.zeros((Npart**3), dtype = 'float32')
	v = np.zeros((Npart**3), dtype = 'float32')
	
	n_parts_accepted = 0
	
	while n_parts_accepted < Npart**3:
		u1 = np.random.uniform(-1,1, Npart**3-n_parts_accepted)
		v1 = np.random.uniform(-1,1, Npart**3-n_parts_accepted)
		sq = u1**2 + v1**2
		u1 = u1[np.nonzero((0. < sq) & (sq < 1.))]
		v1 = v1[np.nonzero((0. < sq) & (sq < 1.))]

		u[n_parts_accepted:n_parts_accepted+len(u1)] = u1
		v[n_parts_accepted:n_parts_accepted+len(v1)] = v1
		n_parts_accepted += len(u1)
			
	#Transformation from uniform to Gaussian Random numbers
	s = u**2+v**2

	#Reshaping to a grid
	u = np.reshape(u, (Npart, Npart, Npart))
	v = np.reshape(v, (Npart, Npart, Npart))
	s = np.reshape(s, (Npart, Npart, Npart))

	#Polar Box-Muller transform
	f1 = u*(-2*np.log(s)/s)**0.5
	f2 = v*(-2*np.log(s)/s)**0.5

	return f1, f2

def lcdm_transfer_function(box, cosm):
	k_grid = fourier_grid(box)

	h = cosm.H0
	Omega_0 = cosm.omega_m0
	Omega_b = cosm.omega_b
	
	#Transfer function for LCDM
	Gamma = Omega_0 * h * np.exp(-Omega_b - Omega_b/Omega_0)
	q = k_grid / Gamma
	factor1 = np.sqrt(1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)
	factor2 = np.divide(np.log(1 + 2.34*q)**2, (2.34*q)**2, where=q!=0)
	lcdm = factor2 / factor1

	return lcdm

def fourier_grid(box):
	Npart = box.Npart
	Lx = box.Lx

	#Finding the fourier frequency axes  
	scale = 2*np.pi*Npart/Lx
	lxaxis = scale*np.fft.fftfreq(Npart)
	lyaxis = scale*np.fft.fftfreq(Npart)
	lzaxis = scale*np.fft.fftfreq(Npart)
	
	#Make a Fourier grid
	lz, ly, lx = np.meshgrid(lzaxis, lyaxis, lxaxis, indexing='ij')   
	
	k_grid = np.sqrt(lx**2 + ly**2 + lz**2)

	return k_grid

def power_spectrum(box, cosm, lcdm):
	Npart = box.Npart

	n = cosm.powspec
	h = cosm.H0

	k_grid = fourier_grid(box)

	#Number of pixels used for normalisation
	Npix = Npart**3

	#Variance of density fluctuations
	sigma2fluxt = 64*h**2

	#Calculating the power spectrum
	if n >= 0.:

		#Sum of power spectrum
		summ = np.sum(np.power(k_grid, n, where=k_grid!=0)*lcdm)
		
		#Amplitude of fluctuations
		A = sigma2fluxt*Npix**2/summ

		#Power spectrum
		p = A*(k_grid)**n*lcdm

	else:
		#Sum of power spectrum
		kgrid_inverse = np.power(k_grid,-n,where=k_grid!=0)
		div_kgrid = np.divide(1, kgrid_inverse, where=kgrid_inverse!=0)
		summ = np.sum(div_kgrid)

		#Amplitude of fluctuations
		A = sigma2fluxt*Npix**2/summ

		#Power spectrum
		p = A*div_kgrid*lcdm

	return p