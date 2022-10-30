import numpy as np

def integrate(box, x_dat, y_dat, z_dat, vx_dat, vy_dat, vz_dat, a_val, f_a, f_a1, da, potentials):
	#This function calculates the momenta from the potential field
	#and updates particle position and momentum
	directions = [0,1,2]

	x_dat, vx_dat = sweep_one_direction(box, potentials, x_dat, y_dat, z_dat, vx_dat, vy_dat, vz_dat, a_val, f_a1, da, directions[0])
	y_dat, vy_dat = sweep_one_direction(box, potentials, x_dat, y_dat, z_dat, vx_dat, vy_dat, vz_dat, a_val, f_a1, da, directions[1])
	z_dat, vz_dat = sweep_one_direction(box, potentials, x_dat, y_dat, z_dat, vx_dat, vy_dat, vz_dat, a_val, f_a1, da, directions[2])

	return x_dat, y_dat, z_dat, vx_dat, vy_dat, vz_dat

def weights(x, y, z, x_dat, y_dat, z_dat):
	d_x = x_dat - x
	d_y = y_dat - y
	d_z = z_dat - z

	t_x = 1 - d_x
	t_y = 1 - d_y
	t_z = 1 - d_z
	return np.array([t_x*t_y*t_z, d_x*t_y*t_z, t_x*d_y*t_z, t_x*t_y*d_z, d_x*d_y*t_z, d_x*t_y*d_z, t_x*d_y*d_z, d_x*d_y*d_z])
	
def sweep_one_direction(box, potentials, x_dat, y_dat, z_dat, vx, vy, vz, a_val, f_a1, da, direction):
	Ngrid = box.Ngrid

	#We again define the cell center coordinates for CIC interpolation
	x = np.floor(x_dat).astype(int)
	y = np.floor(y_dat).astype(int)
	z = np.floor(z_dat).astype(int)
	t = weights(x, y, z, x_dat, y_dat, z_dat)

	x2 = x; y2 = y; z2 = z

	if direction==0:
		x = (x+1)%Ngrid
		x2 = x-1
		velocities = vx
		positions = x_dat
	elif direction==1:
		y = (y+1)%Ngrid
		y2 = y-1
		velocities = vy
		positions = y_dat
	else:
		z = (z+1)%Ngrid
		z2 = z-1
		velocities = vz
		positions = z_dat

	X = (x+1)%Ngrid
	Y = (y+1)%Ngrid
	Z = (z+1)%Ngrid
	X2 = (x2+1)%Ngrid
	Y2 = (y2+1)%Ngrid
	Z2 = (z2+1)%Ngrid

	g = (-potentials[z,y,x] + potentials[z2,y2,x2])
	g_x = (-potentials[z,y,X] + potentials[z2,y2,X2])
	g_y = (-potentials[z,Y,x] + potentials[z2,Y2,x2])
	g_z = (-potentials[Z,y,x] + potentials[Z2,y2,x2])
	g_xy = (-potentials[z,Y,X] + potentials[z2,Y2,X2])
	g_xz = (-potentials[Z,y,X] + potentials[Z2,y2,X2])
	g_yz = (-potentials[Z,Y,x] + potentials[Z2,Y2,x2])
	g_xyz = (-potentials[Z,Y,X] + potentials[Z2,Y2,X2])
	
	#We calculate the parent momentum using these contributions
	g_p = (g*t[0] + g_x*t[1] + g_y*t[2] + g_z*t[3] + g_xy*t[4] + g_xz*t[5] + g_yz*t[6] + g_xyz*t[7])/2.

	#Calculate the new velocity at a+da
	velocities += da*f_a1*g_p

	positions = (positions + da*velocities/(a_val+da)**2*f_a1)%Ngrid 

	return positions, velocities