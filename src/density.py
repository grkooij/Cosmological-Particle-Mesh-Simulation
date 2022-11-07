import numpy as np
from numba import njit
import numba as nb

@njit(parallel=True, fastmath=True)
def density(Ngrid, positions, mass):

	#Create a new grid which will contain the densities
	grid = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float64)
	x_dir = 0
	y_dir = 1
	z_dir = 2

	x_c = np.floor(positions[x_dir]).astype(np.int64)
	y_c = np.floor(positions[y_dir]).astype(np.int64)
	z_c = np.floor(positions[z_dir]).astype(np.int64)

	#Calculating contributions for the CIC interpolation
	d_x = positions[x_dir] - x_c
	d_y = positions[y_dir] - y_c
	d_z = positions[z_dir] - z_c

	t_x = 1 - d_x
	t_y = 1 - d_y
	t_z = 1 - d_z
			
	#Enforce periodicity for cell center coordinates + 1                
	X = (x_c+1)%Ngrid
	Y = (y_c+1)%Ngrid
	Z = (z_c+1)%Ngrid

	#Populate the density grid according to the CIC scheme
	for i in nb.prange(len(positions[x_dir])):

		grid[z_c[i],y_c[i],x_c[i]] += mass*t_x[i]*t_y[i]*t_z[i]
					
		grid[z_c[i],y_c[i],X[i]] += mass*d_x[i]*t_y[i]*t_z[i]
		grid[z_c[i],Y[i],x_c[i]] += mass*t_x[i]*d_y[i]*t_z[i]
		grid[Z[i],y_c[i],x_c[i]] += mass*t_x[i]*t_y[i]*d_z[i]
			
		grid[z_c[i],Y[i],X[i]] += mass*d_x[i]*d_y[i]*t_z[i]
		grid[Z[i],Y[i],x_c[i]] += mass*t_x[i]*d_y[i]*d_z[i]
		grid[Z[i],y_c[i],X[i]] += mass*d_x[i]*t_y[i]*d_z[i]
			
		grid[Z[i],Y[i],X[i]] += mass*d_x[i]*d_y[i]*d_z[i]
	return grid