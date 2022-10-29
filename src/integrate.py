import numpy as np

def integrate(box, x_dat, y_dat, z_dat, vx_dat, vy_dat, vz_dat, a_val, f_a, f_a1, da, potentials):
	#This function calculates the momenta from the potential field
	#and updates particle position and momentum

	#Unpack variables
	Ngrid = box.Ngrid
	Lx = box.Lx

	#We again define the cell center coordinates for CIC interpolation
	x = np.floor(x_dat).astype(int)
	y = np.floor(y_dat).astype(int)
	z = np.floor(z_dat).astype(int)
	
	d_x = x_dat - x
	d_y = y_dat - y
	d_z = z_dat - z

	t_x = 1 - d_x
	t_y = 1 - d_y
	t_z = 1 - d_z

	#And enforce periodicity for cell center +1/ +2
	X = (x+1)%Ngrid
	Y = (y+1)%Ngrid
	Z = (z+1)%Ngrid
	X2 = (x+2)%Ngrid
	Y2 = (y+2)%Ngrid
	Z2 = (z+2)%Ngrid

	#Interpolating for the CIC implementation to obtain particle momenta

	#Interpolating for the momenta in x
	gx = (-potentials[z,y,X] + potentials[z,y,x-1])/2.
	gx_x = (-potentials[z,y,X2] + potentials[z,y,x])/2.
	gx_y = (-potentials[z,Y,X] + potentials[z,Y,x-1])/2.
	gx_z = (-potentials[Z,y,X] + potentials[Z,y,x-1])/2.
	gx_xy = (-potentials[z,Y,X2] + potentials[z,Y,x])/2.
	gx_xz = (-potentials[Z,y,X2] + potentials[Z,y,x])/2.
	gx_yz = (-potentials[Z,Y,X] + potentials[Z,Y,x-1])/2.
	gx_xyz = (-potentials[Z,Y,X2] + potentials[Z,Y,x])/2.

	#Interpolating for the momenta in y
	gy = (-potentials[z,Y,x] + potentials[z,y-1,x])/2.
	gy_x = (-potentials[z,Y,X] + potentials[z,y-1,X])/2.
	gy_y = (-potentials[z,Y2,x] + potentials[z,y,x])/2.
	gy_z = (-potentials[Z,Y,x] + potentials[Z,y-1,x])/2.
	gy_xy = (-potentials[z,Y2,X] + potentials[z,y,X])/2.
	gy_xz = (-potentials[Z,Y,X] + potentials[Z,y-1,X])/2.
	gy_yz = (-potentials[Z,Y2,x] + potentials[Z,y,x])/2.
	gy_xyz = (-potentials[Z,Y2,X] + potentials[Z,y,X])/2.

	#Interpolating for the momenta in z
	gz = (-potentials[Z,y,x] + potentials[z-1,y,x])/2.
	gz_x = (-potentials[Z,y,X] + potentials[z-1,y,X])/2.
	gz_y = (-potentials[Z,Y,x] + potentials[z-1,Y,x])/2.
	gz_z = (-potentials[Z2,y,x] + potentials[z,y,x])/2.
	gz_xy = (-potentials[Z,Y,X] + potentials[z-1,Y,X])/2.
	gz_xz = (-potentials[Z2,y,X] + potentials[z,y,X])/2.
	gz_yz = (-potentials[Z2,Y,x] + potentials[z,Y,x])/2.
	gz_xyz = (-potentials[Z2,Y,X] + potentials[z,Y,X])/2.

	#Calculating contribution for each momentum found above
	t1 = t_x*t_y*t_z
	t2 = d_x*t_y*t_z
	t3 = t_x*d_y*t_z
	t4 = t_x*t_y*d_z
	t5 = d_x*d_y*t_z
	t6 = d_x*t_y*d_z
	t7 = t_x*d_y*d_z
	t8 = d_x*d_y*d_z
	
	#We calculate the parent momentum using these contributions
	g_p_x = gx*t1 + gx_x*t2 + gx_y*t3 + gx_z*t4 + gx_xy*t5 + gx_xz*t6 + gx_yz*t7 + gx_xyz*t8
	g_p_y = gy*t1 + gy_x*t2 + gy_y*t3 + gy_z*t4 + gy_xy*t5 + gy_xz*t6 + gy_yz*t7 + gy_xyz*t8
	g_p_z = gz*t1 + gz_x*t2 + gz_y*t3 + gz_z*t4 + gz_xy*t5 + gz_xz*t6 + gz_yz*t7 + gz_xyz*t8
	
	#Calculate the new velocity at a+da
	vx_datnew = vx_dat + da*f_a1*g_p_x
	vy_datnew = vy_dat + da*f_a1*g_p_y
	vz_datnew = vz_dat + da*f_a1*g_p_z

	x_dat = (x_dat + da*vx_datnew/(a_val+da)**2*f_a1) % Ngrid 
	y_dat = (y_dat + da*vy_datnew/(a_val+da)**2*f_a1) % Ngrid
	z_dat = (z_dat + da*vz_datnew/(a_val+da)**2*f_a1) % Ngrid
	
	return x_dat, y_dat, z_dat, vx_datnew, vy_datnew, vz_datnew