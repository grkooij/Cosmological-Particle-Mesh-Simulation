import numpy as np
import pyfftw
from numba import njit

def potential(box, cosm, density, fgrid, a):
	solution_grid = density_k(box, density)
	solution_grid = potential_k(cosm.omega_m0, solution_grid, fgrid, a)
	return potential_real(box, solution_grid)

@njit(parallel = True)
def potential_k(omega_m0, density_k, fgrid, a):
	#Convolving Fourier densities with Green's function to obtain the potential field in Fourier space
	return -3*omega_m0/8/a*fgrid*density_k

def density_k(box, density):
	Ngrid = box.Ngrid
	n_cpu = box.n_cpu
	
	#Create an auxiliary FFT required for PyFFTW module - creating wisdom in the fft_object 
	#and calling the object thereby performing the FFT of the density to Fourier space
	fft_grid = np.zeros([Ngrid, Ngrid, Ngrid], dtype='cfloat')
	fft_object = pyfftw.FFTW(density.astype('cfloat'), fft_grid, direction = 'FFTW_FORWARD', axes=(0,1,2), threads = n_cpu) 
	return fft_object()

def potential_real(box, potential_k):
	Ngrid = box.Ngrid
	n_cpu = box.n_cpu
	#Performing the inverse Fourier transform to obtain potential in real space
	fft_grid = np.zeros([Ngrid, Ngrid, Ngrid], dtype='cfloat')
	ifft_object = pyfftw.FFTW(potential_k, fft_grid, direction = 'FFTW_BACKWARD',axes=(0,1,2), threads = n_cpu)
	
	return (ifft_object().real).astype('float32')
