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
from math import floor
import pyfftw
from time import time   
import matplotlib.pyplot as plt
from IPython.display import clear_output
import sys
import os
import math
import h5py

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
		
class cosmo:
	def __init__(self, H0, omega_b, omega_m0, omega_lambda0, omega_k0, power, a_init):
		self.H0 = H0
		self.omega_b = omega_b
		self.powspec = power
		self.omega_m0 = omega_m0
		self.omega_lambda0 = omega_lambda0
		self.omega_k0 = omega_k0
		self.a_init = a_init
		self.Dt = Dt(a_init, [omega_m0, omega_lambda0, omega_k0])
		self.H01 = H(a_init, H0, [omega_m0, omega_lambda0, omega_k0])
		self.f0 = f(a_init, [omega_m0, omega_lambda0, omega_k0])
		
  
  
def Gaussian_Random_Field():
	#This function generates a Gaussian Random Field for a LCDM universe at a
	#given epoch through Dt

	#Unpack variables
	Npart = box.Npart
	Ngrid = box.Ngrid
	Lx = box.Lx
	n_cpu = box.n_cpu
	seed = box.seed

	n = cos.powspec
	h = cos.H0
	Omega_0 = cos.omega_m0
	Omega_b = cos.omega_b
	Dt = cos.Dt

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

	#Calculating the power spectrum
	if n >= 0.:

		#Sum of power spectrum
		summ = np.sum(np.power(kgrid, n, where=kgrid!=0)*LCDM)
		
		#Amplitude of fluctuations
		A = sigma2fluxt * Npix**2 / summ

		#Power spectrum
		P = A * (kgrid)**n*LCDM
	
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
	#print('GRF density contrasts:', density_real)
	return density_real

def Zeldovich(density_real):

	#Unpack variables
	Npart = box.Npart
	Ngrid = box.Ngrid
	Lx = box.Lx
	n_cpu = box.n_cpu

	a_init = cos.a_init
	f0 = cos.f0
	H0 = cos.H01
	Dt = cos.Dt

	#Reobtaining density field in Fourier space
	density_k = np.fft.fftn(density_real)
	
	#Creating the Fourier axes 
	scale = 2*np.pi*Npart/Lx
	lxaxis = scale*np.fft.fftfreq(Npart)
	lyaxis = scale*np.fft.fftfreq(Npart)
	lzaxis = scale*np.fft.fftfreq(Npart)

	#3D Fourier axes
	lz, ly, lx = np.meshgrid(lzaxis, lyaxis, lxaxis, indexing='ij')   
	kgrid = np.sqrt(lx**2 + ly**2 + lz**2)
	
	#k squared operator
	del_sq = -kgrid**2
	
	#Calculating potential and correcting for scale with mass resolution
	potential = np.divide(density_k, del_sq, where=del_sq!=0)
	
	#Defining gradient operators in Fourier space
	grad_x_operator = -1.j *lx
	grad_y_operator = -1.j *ly
	grad_z_operator = -1.j *lz

	periodic_space = Ngrid/Npart
	#Computing the displacement field in Fourier space
	ZAx = grad_x_operator*potential*periodic_space
	ZAy = grad_y_operator*potential*periodic_space
	ZAz = grad_z_operator*potential*periodic_space
	
	#As part of the PyFFTW module, we must create an auxiliary FFT grid 
	#Then, we call the FFTW module which creates an object with so called wisdom of how to 
	#most efficiently compute the FFT
	#Calling this object performs the FFT

	#For x-axis
	fft_grid = np.zeros([Npart, Npart, Npart], dtype='cfloat')
	fft_ZAx_obj = pyfftw.FFTW(ZAx.astype('cfloat'), fft_grid, direction = 'FFTW_BACKWARD', axes=(0,1,2), threads = n_cpu)
	ZAx = fft_ZAx_obj()

	#For y-axis
	fft_grid = np.zeros([Npart, Npart, Npart], dtype='cfloat')
	fft_ZAy_obj = pyfftw.FFTW(ZAy.astype('cfloat'), fft_grid, direction = 'FFTW_BACKWARD', axes=(0,1,2), threads = n_cpu)
	ZAy = fft_ZAy_obj()

	#For z-axis
	fft_grid = np.zeros([Npart, Npart, Npart], dtype='cfloat')
	fft_ZAz_obj = pyfftw.FFTW(ZAz.astype('cfloat'), fft_grid, direction = 'FFTW_BACKWARD', axes=(0,1,2), threads = n_cpu)
	ZAz = fft_ZAz_obj()

	#Reshaping and correcting for scale with force resolution
	DFx = np.reshape(ZAx, (Npart**3)).real
	DFy = np.reshape(ZAy, (Npart**3)).real
	DFz = np.reshape(ZAz, (Npart**3)).real

	#print('Displacement field:', DFx)

	#Define unperturbed lattice positions
	#Has to be evenly distributed over a periodic box to prevent unwanted perturbations
	#We have included a displacement of 0.5 to reduce shot noise
	periodic_space = Ngrid/Npart
	x_space = np.linspace(0, Ngrid-periodic_space, Npart) + 0.5
	y_space = np.linspace(0, Ngrid-periodic_space, Npart) + 0.5
	z_space = np.linspace(0, Ngrid-periodic_space, Npart) + 0.5

	z_unpert, y_unpert, x_unpert = np.meshgrid(z_space, y_space, x_space, indexing='ij')

	#And we perturb using Zel'dovich approximation, and correct for the scale of the force grid
	#And additionally we add a %Ngrid to enforce periodic boundaries
	x_dat = (np.reshape(x_unpert, Npart*Npart*Npart) + Dt*DFx) % Ngrid
	y_dat = (np.reshape(y_unpert, Npart*Npart*Npart) + Dt*DFy) % Ngrid
	z_dat = (np.reshape(z_unpert, Npart*Npart*Npart) + Dt*DFz) % Ngrid
	
	#And finally calculate the Zel'dovich velocities that are also scaled to the new grid
	vx_dat = a_init*f0*H0*Dt*DFx
	vy_dat = a_init*f0*H0*Dt*DFy
	vz_dat = a_init*f0*H0*Dt*DFz

	#print('Zeldovich velocities: ', vx_dat)
	
	return x_dat,y_dat,z_dat,vx_dat,vy_dat,vz_dat   


def density(x_dat, y_dat, z_dat, mass):
	

	#Unpack variables
	Ngrid = box.Ngrid

	#Create a new grid which will contain the densities
	grid = np.zeros([Ngrid, Ngrid, Ngrid], dtype='float64')

	#Find cell center coordinates
	x_c = np.floor(x_dat).astype(int)
	y_c = np.floor(y_dat).astype(int)
	z_c = np.floor(z_dat).astype(int)
		
	#Calculating contributions for the CIC interpolation
	d_x = x_dat - x_c
	d_y = y_dat - y_c
	d_z = z_dat - z_c
	
	t_x = 1 - d_x
	t_y = 1 - d_y
	t_z = 1 - d_z
			   
	#Enforce periodicity for cell center coordinates + 1                
	X = (x_c+1)%Ngrid
	Y = (y_c+1)%Ngrid
	Z = (z_c+1)%Ngrid
					
	#Populate the density grid according to the CIC scheme
	grid[z_c,y_c,x_c] += mass*t_x*t_y*t_z
				   
	grid[z_c,y_c,X] += mass*d_x*t_y*t_z
	grid[z_c,Y,x_c] += mass*t_x*d_y*t_z
	grid[Z,y_c,x_c] += mass*t_x*t_y*d_z
		
	grid[z_c,Y,X] += mass*d_x*d_y*t_z
	grid[Z,Y,x_c] += mass*t_x*d_y*d_z
	grid[Z,y_c,X] += mass*d_x*t_y*d_z
		
	grid[Z,Y,X] += mass*d_x*d_y*d_z

	return grid
	
def potential(density, a):
	
	#Unpack variables
	Ngrid = box.Ngrid
	Lx = box.Lx
	n_cpu = box.n_cpu
	Omega_0 = cos.omega_m0

	#Create an auxiliary FFT required for PyFFTW module - creating wisdom in the fft_object 
	#and calling the object thereby performing the FFT of the density to Fourier space
	fft_grid = np.zeros([Ngrid, Ngrid, Ngrid], dtype='cfloat')
	fft_object = pyfftw.FFTW(density.astype('cfloat'), fft_grid, direction = 'FFTW_FORWARD', axes=(0,1,2), threads = n_cpu)
	fft_density = fft_object()
	
	#Creating a Fourier grid
	scale = 2*np.pi
	k_x = scale*np.fft.fftfreq(Ngrid)
	k_y = scale*np.fft.fftfreq(Ngrid)
	k_z = scale*np.fft.fftfreq(Ngrid)

	#Transforming the axes to 3D grids
	ky, kz, kx = np.meshgrid(k_z, k_y, k_x)
	k_squared = np.sin(kz/2)**2 + np.sin(ky/2)**2 + np.sin(kx/2)**2
	
	#Defining the resolution of the force grid in Mpc/cell
	force_resolution = Lx/Ngrid
	
	#Defining Green's function
	Greens_operator = -3*Omega_0/8/a*np.divide(1, k_squared, where=k_squared!=0)*force_resolution**2
	
	#Convolving Fourier densities with Green's function to obtain the potential field in Fourier space
	grid = Greens_operator*fft_density
	
	#Performing the inverse Fourier transform to obtain potential in real space
	fft_grid = np.zeros([Ngrid, Ngrid, Ngrid], dtype='cfloat')
	ifft_object = pyfftw.FFTW(grid, fft_grid, direction = 'FFTW_BACKWARD',axes=(0,1,2), threads = n_cpu)
	grid = (ifft_object().real).astype('float')
	
	#Return the real space potential
	return grid.real


def multi_particle_vector(x_dat, y_dat, z_dat, vx_dat, vy_dat, vz_dat, a_val, f_a, f_a1, da, potentials):
	#This function calculates the momenta from the potential field
	#and updates particle position and momentum

	#Unpack variables
	Ngrid = box.Ngrid
	Lx = box.Lx
	force_resolution = Lx/Ngrid

	#We again define the cell center coordinates for CIC interpolation
	x = np.floor(x_dat).astype(int)
	y = np.floor(y_dat).astype(int)
	z = np.floor(z_dat).astype(int)
	
	d_x = x_dat - x
	d_y = y_dat - y
	d_z = z_dat - z

	t_x = 1 - d_x
	t_y = 1 - d_y
	t_z = 1 - d_z

	#And enforce periodicity for cell center +1/ +2
	X = (x+1)%Ngrid
	Y = (y+1)%Ngrid
	Z = (z+1)%Ngrid
	X2 = (x+2)%Ngrid
	Y2 = (y+2)%Ngrid
	Z2 = (z+2)%Ngrid

	#Interpolating for the CIC implementation to obtain particle momenta

	#Interpolating for the momenta in x
	gx = (-potentials[z,y,X] + potentials[z,y,x-1])/2.
	gx_x = (-potentials[z,y,X2] + potentials[z,y,x])/2.
	gx_y = (-potentials[z,Y,X] + potentials[z,Y,x-1])/2.
	gx_z = (-potentials[Z,y,X] + potentials[Z,y,x-1])/2.
	gx_xy = (-potentials[z,Y,X2] + potentials[z,Y,x])/2.
	gx_xz = (-potentials[Z,y,X2] + potentials[Z,y,x])/2.
	gx_yz = (-potentials[Z,Y,X] + potentials[Z,Y,x-1])/2.
	gx_xyz = (-potentials[Z,Y,X2] + potentials[Z,Y,x])/2.

	#Interpolating for the momenta in y
	gy = (-potentials[z,Y,x] + potentials[z,y-1,x])/2.
	gy_x = (-potentials[z,Y,X] + potentials[z,y-1,X])/2.
	gy_y = (-potentials[z,Y2,x] + potentials[z,y,x])/2.
	gy_z = (-potentials[Z,Y,x] + potentials[Z,y-1,x])/2.
	gy_xy = (-potentials[z,Y2,X] + potentials[z,y,X])/2.
	gy_xz = (-potentials[Z,Y,X] + potentials[Z,y-1,X])/2.
	gy_yz = (-potentials[Z,Y2,x] + potentials[Z,y,x])/2.
	gy_xyz = (-potentials[Z,Y2,X] + potentials[Z,y,X])/2.

	#Interpolating for the momenta in z
	gz = (-potentials[Z,y,x] + potentials[z-1,y,x])/2.
	gz_x = (-potentials[Z,y,X] + potentials[z-1,y,X])/2.
	gz_y = (-potentials[Z,Y,x] + potentials[z-1,Y,x])/2.
	gz_z = (-potentials[Z2,y,x] + potentials[z,y,x])/2.
	gz_xy = (-potentials[Z,Y,X] + potentials[z-1,Y,X])/2.
	gz_xz = (-potentials[Z2,y,X] + potentials[z,y,X])/2.
	gz_yz = (-potentials[Z2,Y,x] + potentials[z,Y,x])/2.
	gz_xyz = (-potentials[Z2,Y,X] + potentials[z,Y,X])/2.

	#Calculating contribution for each momentum found above
	t1 = t_x*t_y*t_z
	t2 = d_x*t_y*t_z
	t3 = t_x*d_y*t_z
	t4 = t_x*t_y*d_z
	t5 = d_x*d_y*t_z
	t6 = d_x*t_y*d_z
	t7 = t_x*d_y*d_z
	t8 = d_x*d_y*d_z
	
	#We calculate the parent momentum using these contributions
	g_p_x = gx*t1 + gx_x*t2 + gx_y*t3 + gx_z*t4 + gx_xy*t5 + gx_xz*t6 + gx_yz*t7 + gx_xyz*t8
	g_p_y = gy*t1 + gy_x*t2 + gy_y*t3 + gy_z*t4 + gy_xy*t5 + gy_xz*t6 + gy_yz*t7 + gy_xyz*t8
	g_p_z = gz*t1 + gz_x*t2 + gz_y*t3 + gz_z*t4 + gz_xy*t5 + gz_xz*t6 + gz_yz*t7 + gz_xyz*t8
	
	#Calculate the new velocity at a+da
	vx_datnew = vx_dat + da*f_a1*g_p_x/force_resolution
	vy_datnew = vy_dat + da*f_a1*g_p_y/force_resolution
	vz_datnew = vz_dat + da*f_a1*g_p_z/force_resolution
	
	#Drift for 0.5da - then kick - drift for 0.5da
	x_dat = (x_dat + .5*da*vx_dat/a_val**2*f_a + .5*da*vx_datnew/(a_val+da)**2*f_a1) % Ngrid 
	y_dat = (y_dat + .5*da*vy_dat/a_val**2*f_a + .5*da*vy_datnew/(a_val+da)**2*f_a1) % Ngrid
	z_dat = (z_dat + .5*da*vz_dat/a_val**2*f_a + .5*da*vz_datnew/(a_val+da)**2*f_a1) % Ngrid
	
	return x_dat, y_dat, z_dat, vx_datnew, vy_datnew, vz_datnew

def H(a, H0, cosmology):
	#This function calculates the Hubble constant

	omegaM = cosmology[0]
	omegaL = cosmology[1]
	omegaK = cosmology[2]
	
	return np.sqrt(H0**2*(omegaM/a**3 + omega_k0/a**2 + omega_lambda0))

def f(a, cosmology):
	#This function calculates the reciprocal of the time derivative of a time H0

	omegaM = cosmology[0]
	omegaL = cosmology[1]
	omegaK = cosmology[2]
	
	return 1/np.sqrt((omegaM + omegaK*a + omegaL*a**3)/a)

def Dt(a, cosmology):
	#This function calculates the growing mode linear growth factor Dt

	omegaM = cosmology[0]
	omegaL = cosmology[1]
	omegaK = cosmology[2]
	
	return 5/2/omegaM/(omegaM**(4/7) - omegaL + (1+omegaM/2)*(1+omegaL/70))*a

def save_file(data, step, conv_pos, conv_vel):
	#This function saves the data to disk

	import os
	filename = 'Data/'
	os.makedirs(os.path.dirname(filename), exist_ok=True)
		
	hf = h5py.File('Data/data.{}.hdf5'.format(step), 'w')

	hf.create_dataset('density', data=data[0])
	hf.create_dataset('x1', data=data[1]*conv_pos)
	hf.create_dataset('x2', data=data[2]*conv_pos)
	hf.create_dataset('x3', data=data[3]*conv_pos)
	hf.create_dataset('vx1', data=data[4]*conv_vel)
	hf.create_dataset('vx2', data=data[5]*conv_vel)
	hf.create_dataset('vx3', data=data[6]*conv_vel)
	
	hf.close()
	
	return

def simulator():
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

	print('Starting the simulation for {}^3 particles'.format(Npart))
	print('With {}^3 grid cells'.format(Ngrid))

	#Physical particle mass in solar masses
	particle_mass = 1.32*10**5*(cos.omega_m0*H0**2)*(Lx/(Npart/128))**3
	print('Particle mass in solar mass for this configuration: {}'.format(particle_mass))

	unit_conv_pos = 7.8*(Lx/(Ngrid/128))/10**3 #Mpc
	unit_conv_vel = 0.781*Lx*H0/(a_init*Ngrid/128) #km/s

	#Since the universe must have a certain average density, the particle mass
	#in code units depends on the number of grid cells to particles cubed.
	dens_contrast = (Ngrid/Npart)**3
	print('Particle mass in code units:', dens_contrast)
	
	#Resolution of mpc/cell
	force_resolution = Lx/Ngrid
	
	#Defining an array of the scale factor that the simulation will loop over
	a = np.linspace(a_init, 1, steps*savesteps)
	
	#The stepsize of the scale factor
	da = a[1] - a_init

	#Cosmology used in GRF and Zel'dovich approximation
	H01 = H(a_init,H0, cosmology)
	f0 = f(a_init, cosmology)
	Dt0 = Dt(a_init, cosmology)
	
	#Creating initial conditions with GRF and Zeldovich displacement
	#First step is the Gaussian Random Field
	print('Preparing the Gaussian random field...')
	density_real = Gaussian_Random_Field()
	
	#Calculating Zel'dovich displacements
	print('Calculating Zeldovich displacements...')
	x_dat,y_dat,z_dat,vx_dat,vy_dat,vz_dat = Zeldovich(density_real)
	
	#Saving the first step to disk
	save_file([density_real, x_dat,y_dat,z_dat,vx_dat,vy_dat,vz_dat], 0, unit_conv_pos, unit_conv_vel)    
 
	#We then loop over the amount of savesteps we want
	print('Starting the integrations...')
	for s in range(savesteps):    
		
		#And loop over the number of steps per savestep 
		for t in range(steps-1):
			
			#Here we calculate the densities, potentials, forces and the velocities for each step

			#Calculating the rough percentage finished
			a_ind =  s*steps + t
			percentile = 100*a_ind/(steps*savesteps) + 0.2
			
			#Calculating the cosmology variables required
			facto = f(a_init, cosmology)
			factoplus1 = f(a_init+da, cosmology)
			a_val = a_init
			
			#Find the densities
			rho = density(x_dat, y_dat, z_dat, dens_contrast) 

			#Solve Poissons equation to obtain the potential       
			potentials = potential(rho, a_val) 

			#Integrating
			x_dat, y_dat, z_dat, vx_dat, vy_dat, vz_dat = multi_particle_vector(x_dat, y_dat, z_dat, vx_dat, vy_dat, vz_dat, a_val, facto, factoplus1, da, potentials)
			
			#Updating the time
			a_init += da
			
			#Saving the data after savesteps
			if t == steps-2:

				#Save results to disk
				save_file([rho, x_dat, y_dat, z_dat, vx_dat, vy_dat, vz_dat], s+1, unit_conv_pos, unit_conv_vel)
				
				#Print percentage finished
				clear_output(wait=True)
				print('{}%'.format(percentile))
				sys.stdout.flush()
	return



#Defining Simulation properties

#Harrison Zeldovich spectrum has n~1
power = 1.


Npart = 100 #number of particles
Ngrid = 100 #number of grid cells
 #number of grid cells
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
simulator()

print("Finished in")
print("--- %s seconds ---" % (time() - start_time))

print('Started plotting density snapshots for z = [99, 9, 1, 0]...')

hf0 = h5py.File('Data/data.1.hdf5', 'r')
hf1 = h5py.File('Data/data.9.hdf5', 'r')
hf2 = h5py.File('Data/data.49.hdf5', 'r')
hf3 = h5py.File('Data/data.99.hdf5', 'r')

rho0 = np.array(hf0.get('density'))[int(Ngrid/2),:,:]
rho1 = np.array(hf1.get('density'))[int(Ngrid/2),:,:]
rho2 = np.array(hf2.get('density'))[int(Ngrid/2),:,:]
rho3 = np.array(hf3.get('density'))[int(Ngrid/2),:,:]

z = 1/np.linspace(a_init, 1, 100) - 1

f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2)

ax1.imshow(rho0, extent = (0,Length_x,0,Length_x))
title = ax1.text(0.5,0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
				transform=ax1.transAxes, ha="center")
title.set_text('z = {:.2f}'.format(z[0]))

ax2.imshow(rho1, extent = (0,Length_x,0,Length_x))
title = ax2.text(0.5,0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
				transform=ax2.transAxes, ha="center")
title.set_text('z = {:.2f}'.format(z[9]))

ax3.imshow(rho2, extent = (0,Length_x,0,Length_x))
title = ax3.text(0.5,0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
				transform=ax3.transAxes, ha="center")
title.set_text('z = {:.2f}'.format(z[49]))

ax4.imshow(rho3, extent = (0,Length_x,0,Length_x))
title = ax4.text(0.5,0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
				transform=ax4.transAxes, ha="center")
title.set_text('z = {:.2f}'.format(z[-1]))

# save figures in the Data directory
plt.savefig('Data/snapshots_density.png')


print('Finished')
plt.show()
