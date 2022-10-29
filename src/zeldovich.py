import numpy as np
import random
import pyfftw

def zeldovich(box, cosm, density_real):

	#Unpack variables
	Npart = box.Npart
	Ngrid = box.Ngrid
	Lx = box.Lx
	n_cpu = box.n_cpu

	a_init = cosm.a_init
	f0 = cosm.f0
	H0 = cosm.H01
	Dt = cosm.Dt

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
	force_resolution = Ngrid/Lx

	DFx = np.reshape(ZAx, (Npart**3)).real * force_resolution
	DFy = np.reshape(ZAy, (Npart**3)).real * force_resolution
	DFz = np.reshape(ZAz, (Npart**3)).real * force_resolution

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
	x = (np.reshape(x_unpert, Npart*Npart*Npart) + Dt*DFx) 
	y_dat = (np.reshape(y_unpert, Npart*Npart*Npart) + Dt*DFy) 
	z_dat = (np.reshape(z_unpert, Npart*Npart*Npart) + Dt*DFz) 

	for i in range(len(x)):
		x[i] += random.uniform(-0.2,0.2)
		y_dat[i] += random.uniform(-0.2,0.2)
		z_dat[i] += random.uniform(-0.2,0.2)

	x = x % Ngrid
	y_dat = y_dat % Ngrid
	z_dat = z_dat % Ngrid
	
	#And finally calculate the Zel'dovich velocities that are also scaled to the new grid
	vx = a_init*f0*H0*Dt*DFx
	vy_dat = a_init*f0*H0*Dt*DFy
	vz_dat = a_init*f0*H0*Dt*DFz
	
	return x,y_dat,z_dat,vx,vy_dat,vz_dat   