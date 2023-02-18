import h5py
import os

from configure_me import SAVE_DENSITY, BOX_SIZE, N_CELLS, H0

def save_file(rho, positions, velocities, step, a):

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
	
	hf.close()
	
	return