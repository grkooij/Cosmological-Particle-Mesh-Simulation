import numba as nb
import numpy as np
from numba import njit
from potential import potential

def advance_time(box, cosm, density, positions, velocities, fgrid, a, fa1, da):

	potentials = potential(box, cosm, density, fgrid, a)
	
	return integrate(positions, velocities, box.Ngrid, a, fa1, da, potentials)

@njit(parallel=True)
def integrate(positions, velocities, Ngrid, a_val, f_a1, da, potentials):

	directions = [0,1,2]

	cell_centers = np.floor(positions).astype(np.int64)	
	t = weights(cell_centers, positions)

	for direction in directions:
		positions[direction], velocities[direction] = sweep_one_direction(cell_centers, positions[direction], velocities[direction], t, potentials, da, f_a1, a_val, Ngrid, direction)
	return positions, velocities

@njit('(int64[:,:], float64[:,:])', parallel=True)
def weights(cell_centers, positions):
	
	x_dir = 0
	y_dir = 1
	z_dir = 2
	t = np.zeros((8, len(positions[x_dir])), dtype=np.float64)

	for i in nb.prange(len(positions[x_dir])):
		d_x = positions[x_dir, i] - cell_centers[x_dir, i]
		d_y = positions[y_dir, i] - cell_centers[y_dir, i]
		d_z = positions[z_dir, i] - cell_centers[z_dir, i]

		t_x = 1 - d_x
		t_y = 1 - d_y
		t_z = 1 - d_z

		t[0, i] = t_x*t_y*t_z
		t[1, i] = d_x*t_y*t_z
		t[2, i] = t_x*d_y*t_z
		t[3, i] = t_x*t_y*d_z
		t[4, i] = d_x*d_y*t_z
		t[5, i] = d_x*t_y*d_z
		t[6, i] = t_x*d_y*d_z
		t[7, i] = d_x*d_y*d_z
	return t

@njit('(int64[:,:], float64[:], float64[:], float64[:,:], float32[:,:,:], float64, float64, float64, int64, int64)', parallel=True)
def sweep_one_direction(cell_centers, positions, velocities, t, potentials, da, f_a1, a_val, Ngrid, direction):

	dir_x = 0
	dir_y = 1
	dir_z = 2
	g_p = np.zeros(len(positions), dtype=np.float64)
	cc_n = cell_centers[:].copy()
	cc_p = cell_centers[:].copy()

	cc_n[direction] = cell_centers[direction]-1
	cc_p[direction] = (cell_centers[direction]+1)%Ngrid
	
	for i in nb.prange(len(positions)):

		x = cc_p[dir_x, i]
		y = cc_p[dir_y, i]
		z = cc_p[dir_z, i]

		x2 = cc_n[dir_x, i]
		y2 = cc_n[dir_y, i]
		z2 = cc_n[dir_z, i]

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
		g_p[i] = (g*t[0,i] + g_x*t[1,i] + g_y*t[2,i] + g_z*t[3,i] + g_xy*t[4,i] + g_xz*t[5,i] + g_yz*t[6,i] + g_xyz*t[7,i])/2.

	velocities += da*f_a1*g_p
	positions = (positions + da*velocities/(a_val+da)**2*f_a1)%Ngrid 

	return positions, velocities
