import numpy as np
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from numba import njit
import matplotlib.pyplot as plt
import h5py


def plot_step(rho, box, savestep):
	length_x = box.Lx

	fig, ax = plt.subplots()
	ax.imshow(rho[0,:,:], extent = (0,length_x,0,length_x), vmin=0., vmax=box.mass*3, cmap='viridis')
	ax.set_xlabel("Mpc/h")
	ax.set_ylabel("Mpc/h")
	plt.savefig('Data/snapshots_density{}.png'.format(savestep), dpi=600, bbox_inches='tight')
	plt.close()

def plot_grf(rho, box):

	length_x = box.Lx
	rho = rho[0,:,:]

	fig, ax = plt.subplots()
	ax.imshow(rho, extent = (0,length_x,0,length_x), vmin=-1., vmax=np.max(rho), cmap='viridis')
	plt.savefig('Data/snapshot_grf.png', dpi=600, bbox_inches='tight')
	plt.close()

def plot_projection(rho, boxsize, mass, savestep, depth):
	cmap = colors.LinearSegmentedColormap.from_list("", ["black", "steelblue", "white", "yellow", "orange", "darkred"])
	cmap.set_bad((0,0,0))

	n_slices = np.int32(boxsize/depth)

	length_x = boxsize

	projection = project(rho, n_slices)
	fig, ax = plt.subplots()
	ax.imshow(projection/n_slices, extent = (0,length_x,0,length_x), norm=LogNorm(vmin=0.2, vmax=mass*10), cmap=cmap)
	ax.set_xlabel("Mpc/h")
	ax.set_ylabel("Mpc/h")

	plt.savefig('Data/projection_density{}.png'.format(savestep), dpi=600, bbox_inches='tight')
	plt.close()

@njit
def project(rho, n_slices):
	projection = np.zeros(np.shape(rho[0,:,:]), dtype=np.float64)

	for i in range(n_slices):
		projection += rho[i,:,:]
	
	return projection