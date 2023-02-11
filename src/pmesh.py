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

from time import time  

from density import density
from integrate import advance_time
from zeldovich import zeldovich
from fourier_utils import fourier_grid
from gaussian_random_field import gaussian_random_field
from cosmology import *
from save_data import save_file
from plot_helper import plot_step, plot_overview, plot_grf, plot_projection


class simbox:
	def __init__(self, Npart, Ngrid, Lx, n_cpu, seed, force_resolution, stepspersavestep, savesteps):
		self.Npart = Npart
		self.Ngrid = Ngrid
		self.Lx = Lx
		self.force_resolution = force_resolution
		self.seed = seed
		self.n_cpu = n_cpu
		self.stepspersavestep = stepspersavestep
		self.savesteps = savesteps
		self.mass = (Ngrid/Npart)**3

def simulator(box, cos):

	#Unpack some variables
	Npart = box.Npart
	Ngrid = box.Ngrid
	Lx = box.Lx
	steps = box.stepspersavestep
	savesteps = box.savesteps
	a_current = cos.a_init
	H0 = cos.H0
	cosmology = [H0, cos.omega_lambda0, cos.omega_k0]

	print('Starting the simulation for {}^3 particles'.format(Npart), 'with {}^3 grid cells'.format(Ngrid))

	#Physical particle mass in solar masses
	particle_mass = 1.32*10**5*(cos.omega_m0*H0**2)*(Lx/(Npart/128))**3
	print('Particle mass in solar mass for this configuration: {:.3E}'.format(particle_mass))

	unit_conv_pos = 7.8*(Lx/(Ngrid/128))/10**3 #Mpc
	unit_conv_vel = 0.781*Lx*H0/(a_init*Ngrid/128) #km/s

	#Since the universe must have a certain average density, the particle mass
	#in code units depends on the number of grid cells to particles cubed.
	dens_contrast = (Ngrid/Npart)**3
	
	print('Particle mass in code units: {:.3e}'.format(dens_contrast))
	
	#Defining an array of the scale factor that the simulation will loop over
	a = np.linspace(a_init, 1, steps*savesteps)
	da = a[1] - a_init
	
	#Creating initial conditions with GRF and Zeldovich displacement
	print('Preparing the Gaussian random field...')
	rho = gaussian_random_field(box, cos)

	print('Calculating Zeldovich displacements...')
	positions, velocities = zeldovich(box, cos, rho)

	print("Setting up Fourier grid...")
	ksq_inverse = fourier_grid(box)
	
	#Saving the first step to disk
	save_file([rho, positions[0], positions[1], positions[2], velocities[0], velocities[1], velocities[2]], 0, unit_conv_pos, unit_conv_vel)  
	plot_grf(box, 0)  
 
	#We then loop over the amount of savesteps we want
	print('Starting the integrations...')
	for s in range(savesteps):    
		start_time = time()
		#And loop over the number of steps per savestep 
		for t in range(steps):

			fa1 = f(a_current+da, cosmology)

			rho = density(box.Ngrid, positions, dens_contrast)
			positions, velocities = advance_time(box, cos, rho, positions, velocities, ksq_inverse, a_current, fa1, da)

			#Updating the time
			a_current += da
			
			#Saving the data after savesteps
			if t == steps-2:
				
				if SAVE_DATA:
					save_file([rho, positions[0], positions[1], positions[2], velocities[0], velocities[1], velocities[2]], s+1, unit_conv_pos, unit_conv_vel)
				if PLOT_STEPS:
					plot_step(box, s+1)
				if PLOT_PROJECTIONS:
					plot_projection(box.Lx, box.mass, s+1, 15)
				if PRINT_STATUS:
					print_status(s, t, steps, start_time)
						
	return

def print_status(s, t, steps, start_time):
	a_ind =  s*steps + t
	percentile = 100*a_ind/(steps*savesteps) + 0.2
	print('{}%'.format(percentile), "Save step time: --- %.5f seconds ---" % (time() - start_time))

if __name__ == "__main__":

	#Defining Simulation properties

	#Harrison Zeldovich spectrum has n~1
	power = 1.

	Npart = 256 #number of particles
	Ngrid = 512 #number of grid cells
	Length_x = 100 #Mpc
	force_resolution = Length_x/Ngrid

	# number of cpu's in the machine the simulation will run on (used for faster fft)
	n_cpu = 16
	# random seed to use for the Gaussian Random Field.
	seed = 38

	stepspersavestep = 10
	savesteps = 100

	#Lambda CDM
	omega_m0 = 0.31
	omega_k0 = 0. 
	omega_lambda0 = 0.69
	omega_b = 0.04
	H0_LCDM = 0.68

	#Initial value of the scale factor
	a_init = 0.01

	box = simbox(Npart, Ngrid, Length_x, n_cpu, seed, force_resolution, stepspersavestep, savesteps)
	cos = cosmo(H0_LCDM, omega_b, omega_m0, omega_lambda0, omega_k0, power, a_init)

	PLOT_STEPS = True
	PLOT_PROJECTIONS = True
	SAVE_DATA = True
	PRINT_STATUS = True

	#Time how long it runs
	start_time = time()

	#Calling the main function
	simulator(box, cos)

	print("Finished in")
	print("--- %s seconds ---" % (time() - start_time))

	print('Started plotting density snapshots for z = [99, 9, 1, 0]...')
	plot_overview(box, cos)

