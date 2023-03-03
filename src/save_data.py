import h5py
import os
import numpy as np

from configure_me import SAVE_DENSITY, BOX_SIZE, N_CELLS, N_PARTS, H0

def save_file(rho, positions, velocities, step, a):
	print("Writing to disk: data.{}.hdf5".format(step))

	unit_conv_pos = 7.8*(BOX_SIZE/(N_CELLS/128))/10**3 #Mpc
	unit_conv_vel = 0.781*BOX_SIZE*H0/(a*N_CELLS/128) #km/s

	filename = 'Data/'
	os.makedirs(os.path.dirname(filename), exist_ok=True)
		
	hf = h5py.File('Data/data.{}.hdf5'.format(step), 'w')
	if SAVE_DENSITY:
		hf.create_dataset('density', data=rho)
	hf.create_dataset('x1', data=positions[0]*unit_conv_pos)
	hf.create_dataset('x2', data=positions[1]*unit_conv_pos)
	hf.create_dataset('x3', data=positions[2]*unit_conv_pos)
	hf.create_dataset('vx1', data=velocities[0]*unit_conv_vel)
	hf.create_dataset('vx2', data=velocities[1]*unit_conv_vel)
	hf.create_dataset('vx3', data=velocities[2]*unit_conv_vel)
	hf.create_dataset('a', data=a)
	
	hf.close()
	
	return

def from_file(step):

	positions = np.zeros((3, N_PARTS*N_PARTS*N_PARTS), dtype=np.float32)
	velocities = np.zeros((3, N_PARTS*N_PARTS*N_PARTS), dtype=np.float32)

	hf = h5py.File('Data/data.{}.hdf5'.format(step), 'r')

	a = np.float32(hf.get('a'))

	unit_conv_pos = 7.8*(BOX_SIZE/(N_CELLS/128))/10**3 #Mpc
	unit_conv_vel = 0.781*BOX_SIZE*H0/(a*N_CELLS/128) #km/s

	positions[0] = np.array(hf.get('x1'))/unit_conv_pos
	positions[1] = np.array(hf.get('x2'))/unit_conv_pos
	positions[2] = np.array(hf.get('x3'))/unit_conv_pos
	velocities[0] = np.array(hf.get('vx1'))/unit_conv_vel
	velocities[1] = np.array(hf.get('vx2'))/unit_conv_vel
	velocities[2] = np.array(hf.get('vx3'))/unit_conv_vel

	return positions, velocities, a