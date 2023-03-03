import numba as nb

from time import time  

from density import density
from integrate import advance_time
from zeldovich import zeldovich
from fourier_utils import fourier_grid
from gaussian_random_field import gaussian_random_field
from cosmology import *
from save_data import save_file, from_file
from plot_helper import plot_step, plot_grf, plot_projection

from configure_me import N_CELLS, N_PARTS, BOX_SIZE, N_CPU, STEPS, N_SAVE_FILES, N_PLOTS, RESTART
from configure_me import OMEGA_M0, H0, A_INIT, A_END, RESTART_FROM_N
from configure_me import SAVE_DATA, PLOT_STEPS, PLOT_PROJECTIONS, PLOT_GRF, PRINT_STATUS

def simulator():

	print('Starting the simulation for {}^3 particles'.format(N_PARTS), 'with {}^3 grid cells'.format(N_CELLS))

	#Physical particle mass in solar masses
	particle_mass = 1.32*10**5*(OMEGA_M0*H0**2)*(BOX_SIZE/(N_PARTS/128))**3
	print('Particle mass in solar mass for this configuration: {:.3E}'.format(particle_mass))

	#Since the universe must have a certain average density, the particle mass
	#in code units depends on the number of grid cells to particles cubed.
	dens_contrast = (N_CELLS/N_PARTS)**3
	print('Particle mass in code units: {:.3e}'.format(dens_contrast))
	da = (A_END - A_INIT)/STEPS
	da_save = (A_END - A_INIT)/N_SAVE_FILES
	da_plot = (A_END - A_INIT)/N_PLOTS
	a_current = A_INIT
	n_file = 0
	n_plot = 0
	
	#Restarting from file
	if RESTART:
		print("Restarting from file data.{}.hdf5".format(RESTART_FROM_N))
		n_file = RESTART_FROM_N
		n_plot = (RESTART_FROM_N)/N_SAVE_FILES*N_PLOTS
		positions, velocities, a_current = from_file(n_file)

	#Creating initial conditions with GRF and Zeldovich displacement
	else:
		rho = gaussian_random_field()
		positions, velocities = zeldovich(rho)
		if SAVE_DATA:
			save_file(rho, positions, velocities, 0, a_current)
			n_file += 1
		if PLOT_GRF:
			plot_grf(rho) 

	ksq_inverse = fourier_grid()  
	print('Starting the integrations...')
	while a_current<A_END-da:
		
		start_time = time()

		rho = density(positions, dens_contrast)
		positions, velocities = advance_time(rho, positions, velocities, ksq_inverse, a_current, da)

		a_current += da
		
		if a_current >= A_INIT + n_file*da_save:
			if SAVE_DATA:
				save_file(rho, positions, velocities, n_file, a_current)
			n_file += 1
		if a_current >= A_INIT + n_plot*da_plot:	
			if PLOT_STEPS:
				plot_step(rho, n_plot)
			if PLOT_PROJECTIONS:
				plot_projection(rho, n_plot, 15)
			n_plot += 1

		if PRINT_STATUS:
			print_status(a_current, start_time)
						
	return

def print_status(a_current, start_time):

	percentile = 100*(a_current-A_INIT)/(A_END-A_INIT)
	print("%.5f" % percentile,"%", " Save step time: --- %.5f seconds ---" % (time() - start_time))

if __name__ == "__main__":

	nb.set_num_threads(N_CPU)
	start_time = time()

	simulator()
	print("Finished in")
	print("--- %.2f seconds ---" % (time() - start_time))
