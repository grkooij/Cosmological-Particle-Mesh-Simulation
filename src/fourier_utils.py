import numpy as np

def fourier_grid(box):
	Ngrid = box.Ngrid
	#Creating a Fourier grid
	scale = 2*np.pi
	k_x = np.array(scale*np.fft.fftfreq(Ngrid), dtype='float32')
	k_y = np.array(scale*np.fft.fftfreq(Ngrid), dtype='float32')
	k_z = np.array(scale*np.fft.fftfreq(Ngrid), dtype='float32')

	#Transforming the axes to 3D grids
	ky, kz, kx = np.meshgrid(k_z, k_y, k_x)
	k_squared = np.sin(kz/2)**2 + np.sin(ky/2)**2 + np.sin(kx/2)**2
	return np.divide(1, k_squared, where=k_squared!=0)