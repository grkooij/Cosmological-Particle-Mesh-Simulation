import numpy as np
import random
import pyfftw

from cosmology import Dt, H, f

from configure_me import N_CPU, RANDOM_SEED, BOX_SIZE, N_PARTS, N_CELLS, A_INIT
from configure_me import OMEGA_M0, OMEGA_LAMBDA0, OMEGA_K0, H0

def zeldovich(density):
	print('Calculating Zeldovich displacements...')
	positions = np.zeros((3, N_PARTS**3), dtype=np.float64)
	velocities = np.zeros((3, N_PARTS**3), dtype=np.float64)

	directions = [0, 1, 2]

	density = np.fft.fftn(density)

	for direction in directions:
		pot_k = potential_k(density)
		positions[direction,:], velocities[direction] = zeldovich_one_direction(pot_k, direction)
	return positions, velocities

def potential_k(density_k):

	scale = 2*np.pi*N_PARTS/BOX_SIZE
	lxaxis = scale*np.fft.fftfreq(N_PARTS)
	lyaxis = scale*np.fft.fftfreq(N_PARTS)
	lzaxis = scale*np.fft.fftfreq(N_PARTS)

	#3D Fourier axes
	lz, ly, lx = np.meshgrid(lzaxis, lyaxis, lxaxis, indexing='ij')   

	#-k squared operator where k = sqrt(lx**2 + ly**2 + lz**2)
	del_sq = -(lx**2 + ly**2 + lz**2)
	
	#Calculating potential and correcting for scale with mass resolution
	return np.divide(density_k, del_sq, where=del_sq!=0)

def zeldovich_one_direction(potential_k, direction):
	displacement_field = displacement_field_one_direction(potential_k, direction)

	return zeldovich_positions(displacement_field, direction), zeldovich_velocities(displacement_field)

def displacement_field_one_direction(potential_k, direction):
	force_resolution = N_CELLS/BOX_SIZE

	#FFT to obtain the real displacement field
	fft_grid = np.zeros([N_PARTS, N_PARTS, N_PARTS], dtype='cfloat')
	df_k = displacement_field_k(potential_k, direction).astype('cfloat')
	fft_ZAx_obj = pyfftw.FFTW(df_k, fft_grid, direction = 'FFTW_BACKWARD', axes=(0,1,2), threads = N_CPU)

	displacement_field = np.reshape(fft_ZAx_obj(), (N_PARTS**3)).real*force_resolution
	return displacement_field

def displacement_field_k(potential_k, direction):

	resolution = N_CELLS/N_PARTS

	#Creating the Fourier axes 
	scale = 2*np.pi*N_PARTS/BOX_SIZE
	lxaxis = scale*np.fft.fftfreq(N_PARTS)
	lyaxis = scale*np.fft.fftfreq(N_PARTS)
	lzaxis = scale*np.fft.fftfreq(N_PARTS)

	#3D Fourier axes for direction
	l_direction = np.meshgrid(lxaxis, lyaxis, lzaxis, indexing='ij')[direction]
	
	return -1.j*l_direction*potential_k*resolution

def zeldovich_positions(displacement_field, direction):
	
	lin_growth_factor = Dt(A_INIT, [OMEGA_M0, OMEGA_LAMBDA0, OMEGA_K0])
	mass_resolution = N_CELLS/N_PARTS

	#Define unperturbed lattice positions
	#Has to be evenly distributed over a periodic box to prevent unwanted perturbations
	#We have included a displacement of 0.5 to reduce shot noise
	x_space = np.linspace(0, N_CELLS-mass_resolution, N_PARTS) + 0.5
	y_space = np.linspace(0, N_CELLS-mass_resolution, N_PARTS) + 0.5
	z_space = np.linspace(0, N_CELLS-mass_resolution, N_PARTS) + 0.5

	positions = np.reshape(np.meshgrid(x_space, y_space, z_space, indexing='ij')[direction], N_PARTS*N_PARTS*N_PARTS)

	#Perturb using Zel'dovich approximation
	#And additionally we add a %N_CELLS to enforce periodic boundaries
	positions += lin_growth_factor*displacement_field

	np.random.seed(RANDOM_SEED)
	for i in range(len(positions)):
		positions[i] += random.uniform(-2.,2.)

	return positions%N_CELLS

def zeldovich_velocities(displacement_field):
	dt_0 = Dt(A_INIT, [OMEGA_M0, OMEGA_LAMBDA0, OMEGA_K0])
	h_0 = H(A_INIT, H0, [OMEGA_M0, OMEGA_LAMBDA0, OMEGA_K0])
	f_0 = f(A_INIT, [OMEGA_M0, OMEGA_LAMBDA0, OMEGA_K0])

	return A_INIT*f_0*h_0*dt_0*displacement_field