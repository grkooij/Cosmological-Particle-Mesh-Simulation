import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py


def plot_step(box, savestep):
	Ngrid = box.Ngrid
	length_x = box.Lx
	hf = h5py.File('Data/data.{}.hdf5'.format(savestep), 'r')

	rho = np.array(hf.get('density'))[int(Ngrid/2),:,:]

	fig, ax = plt.subplots()
	ax.imshow(rho, extent = (0,length_x,0,length_x))
	plt.savefig('Data/snapshots_density{}.png'.format(savestep))
	plt.close()

def plot_overview(box, cosm):
	n_grid = box.Ngrid
	length_x = box.Lx
	a_init = cosm.a_init

	hf0 = h5py.File('Data/data.1.hdf5', 'r')
	hf1 = h5py.File('Data/data.9.hdf5', 'r')
	hf2 = h5py.File('Data/data.49.hdf5', 'r')
	hf3 = h5py.File('Data/data.99.hdf5', 'r')

	rho0 = np.array(hf0.get('density'))[int(n_grid/2),:,:]
	rho1 = np.array(hf1.get('density'))[int(n_grid/2),:,:]
	rho2 = np.array(hf2.get('density'))[int(n_grid/2),:,:]
	rho3 = np.array(hf3.get('density'))[int(n_grid/2),:,:]

	z = 1/np.linspace(a_init, 1, 100) - 1

	f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2)

	ax1.imshow(rho0, extent = (0,length_x,0,length_x))
	title = ax1.text(0.5,0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
					transform=ax1.transAxes, ha="center")
	title.set_text('z = {:.2f}'.format(z[0]))

	ax2.imshow(rho1, extent = (0,length_x,0,length_x))
	title = ax2.text(0.5,0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
					transform=ax2.transAxes, ha="center")
	title.set_text('z = {:.2f}'.format(z[9]))

	ax3.imshow(rho2, extent = (0,length_x,0,length_x))
	title = ax3.text(0.5,0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
					transform=ax3.transAxes, ha="center")
	title.set_text('z = {:.2f}'.format(z[49]))

	ax4.imshow(rho3, extent = (0,length_x,0,length_x))
	title = ax4.text(0.5,0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
					transform=ax4.transAxes, ha="center")
	title.set_text('z = {:.2f}'.format(z[-1]))

	# save figures in the Data directory
	plt.savefig('Data/snapshots_density.png')

	print('Finished')
	plt.show()