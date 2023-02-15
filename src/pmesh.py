"""
	The following program executes an N-body simulation using the Particle Mesh (PM) method. 
	It was created for fun over a summer holiday by Richard Kooij and finished by Richard Kooij and Francis Tang
	as a project for the course Modelling and Simulation, semester Ia at the Rijksuniversiteit Groningen.
	Currently uses LCDM cosmology, but can be changed to involve other power spectra/cosmologies.

	The general working of the code is as follows

	- Generate a Gaussian Random Field (GRF)
	- Deposit particles uniformly on a uniform 3D grid
	- Displace particles with Zel'dovich approximation based on GRF
	- Find density on the grid according to particle distribution
	- Calculate the potential from Poisson's equation
	- Integrate using leap frog to find positions, velocities
"""

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
from configure_me import SAVE_DATA, PLOT_STEPS, PLOT_PROJECTIONS, PRINT_STATUS

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

def simulator(box, cos):

	#Unpack some variables
	Npart = box.Npart
	Ngrid = box.Ngrid
	Lx = box.Lx
	steps = box.steps
	a_current = cos.a_init
	H0 = cos.H0
	cosmology = [H0, cos.omega_lambda0, cos.omega_k0]

	print('Starting the simulation for {}^3 particles'.format(Npart), 'with {}^3 grid cells'.format(Ngrid))

	#Physical particle mass in solar masses
	particle_mass = 1.32*10**5*(cos.omega_m0*H0**2)*(Lx/(Npart/128))**3
	print('Particle mass in solar mass for this configuration: {:.3E}'.format(particle_mass))

	unit_conv_pos = 7.8*(Lx/(Ngrid/128))/10**3 #Mpc
	unit_conv_vel = 0.781*Lx*H0/(a_current*Ngrid/128) #km/s

	#Since the universe must have a certain average density, the particle mass
	#in code units depends on the number of grid cells to particles cubed.
	dens_contrast = (Ngrid/Npart)**3
	
	print('Particle mass in code units: {:.3e}'.format(dens_contrast))
	
	#Defining an array of the scale factor that the simulation will loop over
	a = np.linspace(a_current, 1., steps)
	da = a[1] - a_current
	da_save = (a[-1] - a[0])/N_SAVE_FILES
	n_file = 0
	
	#Creating initial conditions with GRF and Zeldovich displacement
	print('Preparing the Gaussian random field...')
	rho = gaussian_random_field(box, cos)

	print('Calculating Zeldovich displacements...')
	positions, velocities = zeldovich(box, cos, rho)

	print("Setting up Fourier grid...")
	ksq_inverse = fourier_grid(box)
	
	#Saving the first step to disk
	if SAVE_DATA:
		save_file(rho, positions, velocities, 0, unit_conv_pos, unit_conv_vel)
		n_file += 1

	if PLOT_STEPS:
		plot_grf(rho, box)  
 
	print('Starting the integrations...')
	for s in range(steps):   

		start_time = time()
		
		fa1 = f(a_current+da, cosmology)

		rho = density(box.Ngrid, positions, dens_contrast)
		positions, velocities = advance_time(box, cos, rho, positions, velocities, ksq_inverse, a_current, fa1, da)

		#Updating the time
		a_current += da
		
		if SAVE_DATA:
			if a_current >= n_file*da_save:
				
				save_file(rho, positions, velocities, n_file, unit_conv_pos, unit_conv_vel)

				if PLOT_STEPS:
					plot_step(rho, box, n_file)
				if PLOT_PROJECTIONS:
					plot_projection(rho, box.Lx, box.mass, n_file, 15)

				n_file += 1

		if PRINT_STATUS:
			print_status(s, steps, start_time)
						
	return

def print_status(s, steps, start_time):

	percentile = 100*s/steps
	print('{}%'.format(percentile), "Save step time: --- %.5f seconds ---" % (time() - start_time))

if __name__ == "__main__":

	box = simbox(N_PARTS, N_CELLS, BOX_SIZE, N_CPU, RANDOM_SEED, N_SAVE_FILES, STEPS)
	cos = cosmo(H0, OMEGA_M0, OMEGA_B0, OMEGA_LAMBDA0, OMEGA_K0, POWER, A_INIT)
	nb.set_num_threads(N_CPU)

	#Time how long it runs
	start_time = time()

	#Calling the main function
	simulator(box, cos)

	print("Finished in")
	print("--- %s seconds ---" % (time() - start_time))
