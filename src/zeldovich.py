import numpy as np
import random
import pyfftw

def zeldovich(box, cosm, density):
	directions = [0, 1, 2]

	density = np.fft.fftn(density)
    #Split up to lower RAM usage
	x_dat, vx_dat = zeldovich_one_direction(box, cosm, potential_k(box, density), directions[0])
	y_dat, vy_dat = zeldovich_one_direction(box, cosm, potential_k(box, density), directions[1])
	z_dat, vz_dat = zeldovich_one_direction(box, cosm, potential_k(box, density), directions[2])

	return x_dat, y_dat, z_dat, vx_dat, vy_dat, vz_dat

def potential_k(box, density_k):
	Npart = box.Npart
	Lx = box.Lx

	scale = 2*np.pi*Npart/Lx
	lxaxis = scale*np.fft.fftfreq(Npart)
	lyaxis = scale*np.fft.fftfreq(Npart)
	lzaxis = scale*np.fft.fftfreq(Npart)

	#3D Fourier axes
	lz, ly, lx = np.meshgrid(lzaxis, lyaxis, lxaxis, indexing='ij')   

	#-k squared operator where k = sqrt(lx**2 + ly**2 + lz**2)
	del_sq = -(lx**2 + ly**2 + lz**2)
	
	#Calculating potential and correcting for scale with mass resolution
	potential = np.divide(density_k, del_sq, where=del_sq!=0)
	return potential

def zeldovich_one_direction(box, cosm, potential_k, direction):
	displacement_field = displacement_field_one_direction(box, potential_k, direction)

	return zeldovich_positions(box, cosm, displacement_field, direction), zeldovich_velocities(cosm, displacement_field)

def displacement_field_one_direction(box, potential_k, direction):
	Npart = box.Npart
	Ngrid = box.Ngrid
	Lx = box.Lx
	n_cpu = box.n_cpu

	force_resolution = Ngrid/Lx

	#FFT to obtain the real displacement field
	fft_grid = np.zeros([Npart, Npart, Npart], dtype='cfloat')
	fft_ZAx_obj = pyfftw.FFTW(displacement_field_k(box, potential_k, direction).astype('cfloat'), fft_grid, direction = 'FFTW_BACKWARD', axes=(0,1,2), threads = n_cpu)

	displacement_field = np.reshape(fft_ZAx_obj(), (Npart**3)).real*force_resolution
	return displacement_field

def displacement_field_k(box, potential_k, direction):
	Npart = box.Npart
	Ngrid = box.Ngrid
	Lx = box.Lx
	resolution = Ngrid/Npart

	#Creating the Fourier axes 
	scale = 2*np.pi*Npart/Lx
	lxaxis = scale*np.fft.fftfreq(Npart)
	lyaxis = scale*np.fft.fftfreq(Npart)
	lzaxis = scale*np.fft.fftfreq(Npart)

	#3D Fourier axes for direction
	l_direction = np.meshgrid(lxaxis, lyaxis, lzaxis, indexing='ij')[direction]
	
	return -1.j*l_direction*potential_k*resolution

def zeldovich_positions(box, cosm, displacement_field, direction):
	Npart = box.Npart
	Ngrid = box.Ngrid
	Dt = cosm.Dt

	mass_resolution = Ngrid/Npart

	#Define unperturbed lattice positions
	#Has to be evenly distributed over a periodic box to prevent unwanted perturbations
	#We have included a displacement of 0.5 to reduce shot noise
	x_space = np.linspace(0, Ngrid-mass_resolution, Npart) + 0.5
	y_space = np.linspace(0, Ngrid-mass_resolution, Npart) + 0.5
	z_space = np.linspace(0, Ngrid-mass_resolution, Npart) + 0.5

	positions = np.reshape(np.meshgrid(x_space, y_space, z_space, indexing='ij')[direction], Npart*Npart*Npart)

	#Perturb using Zel'dovich approximation
	#And additionally we add a %Ngrid to enforce periodic boundaries
	positions += Dt*displacement_field

	for i in range(len(positions)):
		positions[i] += random.uniform(-.75,.75)

	return positions%Ngrid

def zeldovich_velocities(cosm, displacement_field):
	a_init = cosm.a_init
	f0 = cosm.f0
	H0 = cosm.H01
	Dt = cosm.Dt

	return a_init*f0*H0*Dt*displacement_field