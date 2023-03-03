import numpy as np
import pyfftw
from numba import njit

from configure_me import N_CELLS, N_CPU, OMEGA_M0

def potential(density, fgrid, a):
	solution_grid = density_k(density)
	solution_grid = potential_k(solution_grid, fgrid, a)
	return potential_real(solution_grid)

@njit(parallel = True)
def potential_k(density_k, fgrid, a):
	#Convolving Fourier densities with Green's function to obtain the potential field in Fourier space
	return -3*OMEGA_M0/8/a*fgrid*density_k

def density_k(density):
	
	fft_grid = np.zeros([N_CELLS, N_CELLS, N_CELLS], dtype=np.cdouble)
	fft_object = pyfftw.FFTW(density.astype(np.cdouble), fft_grid, direction = 'FFTW_FORWARD', axes=(0,1,2), threads = N_CPU) 
	return fft_object()

def potential_real(potential_k):

	#Performing the inverse Fourier transform to obtain potential in real space
	fft_grid = np.zeros([N_CELLS, N_CELLS, N_CELLS], dtype=np.cdouble)
	ifft_object = pyfftw.FFTW(potential_k, fft_grid, direction = 'FFTW_BACKWARD',axes=(0,1,2), threads = N_CPU)
	
	return (ifft_object().real).astype('float32')
