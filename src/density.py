import numpy as np

def density(box, x_dat, y_dat, z_dat, mass):
	
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