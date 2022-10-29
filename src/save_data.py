import h5py
import os

def save_file(data, step, conv_pos, conv_vel):

	filename = 'Data/'
	os.makedirs(os.path.dirname(filename), exist_ok=True)
		
	hf = h5py.File('Data/data.{}.hdf5'.format(step), 'w')

	hf.create_dataset('density', data=data[0])
	hf.create_dataset('x1', data=data[1]*conv_pos)
	hf.create_dataset('x2', data=data[2]*conv_pos)
	hf.create_dataset('x3', data=data[3]*conv_pos)
	hf.create_dataset('vx1', data=data[4]*conv_vel)
	hf.create_dataset('vx2', data=data[5]*conv_vel)
	hf.create_dataset('vx3', data=data[6]*conv_vel)
	
	hf.close()
	
	return