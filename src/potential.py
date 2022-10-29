import numpy as np
import pyfftw

def potential(box, cosm, density, a):
	
	#Unpack variables
	Ngrid = box.Ngrid
	Lx = box.Lx
	n_cpu = box.n_cpu
	Omega_0 = cosm.omega_m0

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
	
	#Defining Green's function
	Greens_operator = -3*Omega_0/8/a*np.divide(1, k_squared, where=k_squared!=0)
	
	#Convolving Fourier densities with Green's function to obtain the potential field in Fourier space
	grid = Greens_operator*fft_density
	
	#Performing the inverse Fourier transform to obtain potential in real space
	fft_grid = np.zeros([Ngrid, Ngrid, Ngrid], dtype='cfloat')
	ifft_object = pyfftw.FFTW(grid, fft_grid, direction = 'FFTW_BACKWARD',axes=(0,1,2), threads = n_cpu)
	grid = (ifft_object().real).astype('float')
	
	#Return the real space potential
	return grid.real