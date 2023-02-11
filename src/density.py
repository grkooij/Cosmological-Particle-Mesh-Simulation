import numpy as np
from numba import njit
import numba as nb

@njit(parallel=True)
def density(Ngrid, positions, mass):

	#Create a new grid which will contain the densities
	grid = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float64)
	x_dir = 0
	y_dir = 1
	z_dir = 2

	#Populate the density grid according to the CIC scheme
	for i in nb.prange(len(positions[x_dir])):

		x_c = np.int64(np.floor(positions[x_dir, i]))
		y_c = np.int64(np.floor(positions[y_dir, i]))
		z_c = np.int64(np.floor(positions[z_dir, i]))

		#Calculating contributions for the CIC interpolation
		d_x = positions[x_dir, i] - x_c
		d_y = positions[y_dir, i] - y_c
		d_z = positions[z_dir, i] - z_c

		t_x = 1 - d_x
		t_y = 1 - d_y
		t_z = 1 - d_z
				
		#Enforce periodicity for cell center coordinates + 1                
		X = (x_c+1)%Ngrid
		Y = (y_c+1)%Ngrid
		Z = (z_c+1)%Ngrid

		grid[z_c,y_c,x_c] += mass*t_x*t_y*t_z
					
		grid[z_c,y_c,X] += mass*d_x*t_y*t_z
		grid[z_c,Y,x_c] += mass*t_x*d_y*t_z
		grid[Z,y_c,x_c] += mass*t_x*t_y*d_z
			
		grid[z_c,Y,X] += mass*d_x*d_y*t_z
		grid[Z,Y,x_c] += mass*t_x*d_y*d_z
		grid[Z,y_c,X] += mass*d_x*t_y*d_z
			
		grid[Z,Y,X] += mass*d_x*d_y*d_z
	return grid