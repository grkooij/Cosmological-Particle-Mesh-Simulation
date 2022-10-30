import numpy as np
import pyfftw

def potential(box, cosm, density, a):
	#Return the real space potential
	return potential_real(box, potential_k(box, cosm, density_k(box, density), a))

def potential_k(box, cosm, density_k, a):
	Omega_0 = cosm.omega_m0

	k_squared = fourier_grid(box)

	#Convolving Fourier densities with Green's function to obtain the potential field in Fourier space
	return -3*Omega_0/8/a*np.divide(1, k_squared, where=k_squared!=0)*density_k

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

def fourier_grid(box):
	Ngrid = box.Ngrid
	#Creating a Fourier grid
	scale = 2*np.pi
	k_x = np.array(scale*np.fft.fftfreq(Ngrid), dtype='float32')
	k_y = np.array(scale*np.fft.fftfreq(Ngrid), dtype='float32')
	k_z = np.array(scale*np.fft.fftfreq(Ngrid), dtype='float32')

	#Transforming the axes to 3D grids
	ky, kz, kx = np.meshgrid(k_z, k_y, k_x)
	return np.sin(kz/2)**2 + np.sin(ky/2)**2 + np.sin(kx/2)**2