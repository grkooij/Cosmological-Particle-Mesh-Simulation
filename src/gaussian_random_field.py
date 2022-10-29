import numpy as np
import random

def gaussian_random_field(box, cosm):
	#This function generates a Gaussian Random Field for a LCDM universe at a
	#given epoch through Dt

	#Unpack variables
	Npart = box.Npart
	Ngrid = box.Ngrid
	Lx = box.Lx
	n_cpu = box.n_cpu
	seed = box.seed

	n = cosm.powspec
	h = cosm.H0
	Omega_0 = cosm.omega_m0
	Omega_b = cosm.omega_b
	Dt = cosm.Dt

	#Generate two sets of uniform, independent random values for a given seed
	np.random.seed(box.seed)
	u = np.zeros((Npart**3), dtype = 'float')
	v = np.zeros((Npart**3), dtype = 'float')
	
	n_parts = 0
	
	while n_parts < Npart**3:
		u1 = np.random.uniform(-1,1)
		v1 = np.random.uniform(-1,1)
		if 0. < u1**2 + v1**2 < 1.: 
			u[n_parts] = u1
			v[n_parts] = v1
			n_parts += 1
			
	#Transformation from uniform to Gaussian Random numbers
	s = u**2+v**2

	#Reshaping to a grid
	u = np.reshape(u, (Npart, Npart, Npart))
	v = np.reshape(v, (Npart, Npart, Npart))
	s = np.reshape(s, (Npart, Npart, Npart))

	#Polar Box-Muller transform
	f1 = u*(-2*np.log(s)/s)**0.5
	f2 = v*(-2*np.log(s)/s)**0.5

	#Finding the fourier frequency axes  
	scale = 2*np.pi*Npart/Lx
	lxaxis = scale*np.fft.fftfreq(Npart)
	lyaxis = scale*np.fft.fftfreq(Npart)
	lzaxis = scale*np.fft.fftfreq(Npart)
	
	#Make a Fourier grid
	lz, ly, lx = np.meshgrid(lzaxis, lyaxis, lxaxis, indexing='ij')   
	
	kgrid = np.sqrt(lx**2 + ly**2 + lz**2)

	#Number of pixels used for normalisation
	Npix = Npart**3

	#Variance of density fluctuations
	sigma2fluxt = 64*h**2

	#Transfer function for LCDM
	Gamma = Omega_0 * h * np.exp(-Omega_b - Omega_b/Omega_0)
	q = kgrid / Gamma
	factor1 = np.sqrt(1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)
	factor2 = np.divide(np.log(1 + 2.34*q)**2, (2.34*q)**2, where=q!=0)
	LCDM = factor2 / factor1

	q1d = lxaxis / Gamma
	factor11d = np.sqrt(1 + 3.89*q1d + (16.1*q1d)**2 + (5.46*q1d)**3 + (6.71*q1d)**4)
	factor21d = np.divide(np.log(1 + 2.34*q1d)**2, (2.34*q1d)**2, where=q1d!=0)
	LCDM1d = factor21d / factor11d

	#Calculating the power spectrum
	if n >= 0.:

		#Sum of power spectrum
		summ = np.sum(np.power(kgrid, n, where=kgrid!=0)*LCDM)
		
		#Amplitude of fluctuations
		A = sigma2fluxt * Npix**2 / summ

		#Power spectrum
		P = A * (kgrid)**n*LCDM
		P1d = A*lxaxis**n*LCDM1d

	else:
		#Sum of power spectrum
		kgrid_inverse = np.power(kgrid,-n,where=kgrid!=0)
		div_kgrid = np.divide(1, kgrid_inverse, where=kgrid_inverse!=0)
		summ = np.sum(div_kgrid)

		#Amplitude of fluctuations
		A = sigma2fluxt * Npix**2 / summ

		#Power spectrum
		P = A * div_kgrid * LCDM

	#Multiplying by the sqrt of the power spectrum and linear growth factor
	f1 = np.sqrt(P*Dt**2) * f1
	f2 = np.sqrt(P*Dt**2) * f2
	
	#Density in Fourier space
	density_k = f1 + 1j*f2
	
	#Real space density field
	density_real = np.fft.ifftn(density_k).real
	return density_real