import numpy as np
import numba as nb

from time import time  

from density import density
from integrate import advance_time
from zeldovich import zeldovich
from fourier_utils import fourier_grid
from gaussian_random_field import gaussian_random_field
from cosmology import *
from save_data import save_file
from plot_helper import plot_step, plot_grf, plot_projection

from configure_me import N_CELLS, N_PARTS, BOX_SIZE, N_CPU, STEPS, N_SAVE_FILES, RANDOM_SEED
from configure_me import POWER, OMEGA_M0, OMEGA_B0, OMEGA_K0, OMEGA_LAMBDA0, H0, A_INIT
from configure_me import SAVE_DATA, PLOT_STEPS, PLOT_PROJECTIONS, PLOT_GRF, PRINT_STATUS

class simbox:
	def __init__(self, Npart, Ngrid, Lx, n_cpu, seed, n_save_files, steps):
		self.Npart = Npart
		self.Ngrid = Ngrid
		self.Lx = Lx
		self.force_resolution = Lx/Ngrid
		self.seed = seed
		self.n_cpu = n_cpu
		self.steps = steps
		self.savesteps = n_save_files
		self.mass = (Ngrid/Npart)**3

def simulator():

	print('Starting the simulation for {}^3 particles'.format(N_PARTS), 'with {}^3 grid cells'.format(N_CELLS))

	#Physical particle mass in solar masses
	particle_mass = 1.32*10**5*(OMEGA_M0*H0**2)*(BOX_SIZE/(N_PARTS/128))**3
	print('Particle mass in solar mass for this configuration: {:.3E}'.format(particle_mass))

	#Since the universe must have a certain average density, the particle mass
	#in code units depends on the number of grid cells to particles cubed.
	dens_contrast = (N_CELLS/N_PARTS)**3
	print('Particle mass in code units: {:.3e}'.format(dens_contrast))
	
	da = (1.0 - A_INIT)/STEPS
	da_save = (1.0 - A_INIT)/N_SAVE_FILES
	a_current = A_INIT
	n_file = 0
	
	#Creating initial conditions with GRF and Zeldovich displacement
	rho = gaussian_random_field()
	positions, velocities = zeldovich(rho)
	ksq_inverse = fourier_grid()
	
	if SAVE_DATA:
		save_file(rho, positions, velocities, 0, a_current)
		n_file += 1
	if PLOT_GRF:
		plot_grf(rho)  
 
	print('Starting the integrations...')
	for s in range(STEPS):   

		start_time = time()

		rho = density(positions, dens_contrast)
		positions, velocities = advance_time(rho, positions, velocities, ksq_inverse, a_current, da)

		a_current += da
		
		if a_current >= n_file*da_save:
			if SAVE_DATA:
				save_file(rho, positions, velocities, n_file, a_current)
			if PLOT_STEPS:
				plot_step(rho, n_file)
			if PLOT_PROJECTIONS:
				plot_projection(rho, n_file, 15)

			n_file += 1

		if PRINT_STATUS:
			print_status(s, start_time)
						
	return

def print_status(s, start_time):

	percentile = 100*s/STEPS
	print('{}%'.format(percentile), "Save step time: --- %.5f seconds ---" % (time() - start_time))

if __name__ == "__main__":

	nb.set_num_threads(N_CPU)
	start_time = time()
	simulator()
	print("Finished in")
	print("--- %.2f seconds ---" % (time() - start_time))
