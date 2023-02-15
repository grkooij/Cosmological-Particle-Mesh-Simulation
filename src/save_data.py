import h5py
import os

from configure_me import SAVE_DENSITY

def save_file(rho, positions, velocities, step, conv_pos, conv_vel):

	filename = 'Data/'
	os.makedirs(os.path.dirname(filename), exist_ok=True)
		
	hf = h5py.File('Data/data.{}.hdf5'.format(step), 'w')
	if SAVE_DENSITY:
		hf.create_dataset('density', data=rho)
	hf.create_dataset('x1', data=positions[0]*conv_pos)
	hf.create_dataset('x2', data=positions[1]*conv_pos)
	hf.create_dataset('x3', data=positions[2]*conv_pos)
	hf.create_dataset('vx1', data=velocities[0]*conv_vel)
	hf.create_dataset('vx2', data=velocities[1]*conv_vel)
	hf.create_dataset('vx3', data=velocities[2]*conv_vel)
	
	hf.close()
	
	return