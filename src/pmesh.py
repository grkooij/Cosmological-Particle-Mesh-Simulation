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
import sys

from IPython.display import clear_output

from density import density
from potential import potential
from integrate import integrate
from zeldovich import zeldovich
from gaussian_random_field import gaussian_random_field
from cosmology import *
from save_data import save_file
from plot_helper import plot_step, plot_overview, plot_grf


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
	#This is the main function - Call this to create initial conditions, 
	#and solve the simulation for the given parameters

	#Unpack some variables
	Npart = box.Npart
	Ngrid = box.Ngrid
	Lx = box.Lx
	steps = box.stepspersavestep
	savesteps = box.savesteps
	a_init = cos.a_init
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
	
	#The stepsize of the scale factor
	da = a[1] - a_init
	
	#Creating initial conditions with GRF and Zeldovich displacement
	#First step is the Gaussian Random Field
	print('Preparing the Gaussian random field...')
	rho = gaussian_random_field(box, cos)
	
	#Calculating Zel'dovich displacements
	print('Calculating Zeldovich displacements...')
	positions, velocities = zeldovich(box, cos, rho)
	
	#Saving the first step to disk
	save_file([rho, positions[0], positions[1], positions[2], velocities[0], velocities[1], velocities[2]], 0, unit_conv_pos, unit_conv_vel)  
	plot_grf(box, 0)  
 
	#We then loop over the amount of savesteps we want
	print('Starting the integrations...')
	for s in range(savesteps):    
		start_time = time()
		#And loop over the number of steps per savestep 
		for t in range(steps):
			
			#Here we calculate the densities, potentials, forces and the velocities for each step
			
			#Calculating the cosmology variables required
			facto = f(a_init, cosmology)
			factoplus1 = f(a_init+da, cosmology)
			a_val = a_init
			
			rho = density(box.Ngrid, positions, dens_contrast) 

			#Solve Poissons equation to obtain the potential       
			#Integrating
			
			positions, velocities = integrate(box, positions, velocities, a_val, facto, factoplus1, da, potential(box, cos, rho, a_val))
			
			#Updating the time
			a_init += da
			
			#Saving the data after savesteps
			if t == steps-2:
				
				#Save results to disk
				save_file([rho, positions[0], positions[1], positions[2], velocities[0], velocities[1], velocities[2]], s+1, unit_conv_pos, unit_conv_vel)
				plot_step(box, s+1)
				print_status(s, t, steps, start_time)
						
	return

def print_status(s, t, steps, start_time):
	a_ind =  s*steps + t
	percentile = 100*a_ind/(steps*savesteps) + 0.2
	print('{}%'.format(percentile), "Save step time: --- %s seconds ---" % (time() - start_time))

if __name__ == "__main__":

	#Defining Simulation properties

	#Harrison Zeldovich spectrum has n~1
	power = 1.

	Npart = 128 #number of particles
	Ngrid = 256 #number of grid cells
	Length_x = 150 #Mpc
	force_resolution = Length_x/Ngrid

	# number of cpu's in the machine the simulation will run on (used for faster fft)
	n_cpu = 4
	# random seed to use for the Gaussian Random Field (our Decennium 1 run used seed=38).
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

	#Time how long it runs
	start_time = time()

	#Calling the main function
	simulator(box, cos)

	print("Finished in")
	print("--- %s seconds ---" % (time() - start_time))

	print('Started plotting density snapshots for z = [99, 9, 1, 0]...')
	plot_overview(box, cos)

