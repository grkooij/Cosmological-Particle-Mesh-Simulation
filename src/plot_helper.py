import numpy as np
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from numba import njit
import matplotlib.pyplot as plt

from configure_me import BOX_SIZE, N_CELLS, N_PARTS


def plot_step(rho, savestep):

	mass = (N_CELLS/N_PARTS)**3
	fig, ax = plt.subplots()
	ax.imshow(rho[0,:,:], extent = (0,BOX_SIZE,0,BOX_SIZE), vmin=0., vmax=mass*3, cmap='viridis')
	ax.set_xlabel("Mpc/h")
	ax.set_ylabel("Mpc/h")
	plt.savefig('Data/snapshots_density{}.png'.format(savestep), dpi=1200, bbox_inches='tight')
	plt.close()

def plot_grf(rho):

	rho = rho[0,:,:]

	fig, ax = plt.subplots()
	ax.imshow(rho, extent = (0,BOX_SIZE,0,BOX_SIZE), vmin=-1., vmax=np.max(rho), cmap='viridis')
	plt.savefig('Data/snapshot_grf.png', dpi=1200, bbox_inches='tight')
	plt.close()

def plot_projection(rho, savestep, depth):
	cmap = colors.LinearSegmentedColormap.from_list("", ["black", "steelblue", "white", "yellow", "orange", "darkred"])
	cmap.set_bad((0,0,0))

	mass = (N_CELLS/N_PARTS)**3

	n_slices = np.int32(BOX_SIZE/depth)

	projection = project(rho, n_slices)
	fig, ax = plt.subplots()
	ax.imshow(projection/n_slices, extent = (0,BOX_SIZE,0,BOX_SIZE), norm=LogNorm(vmin=0.2, vmax=mass*10), cmap=cmap)
	ax.set_xlabel("Mpc/h")
	ax.set_ylabel("Mpc/h")

	plt.savefig('Data/projection_density{}.png'.format(savestep), dpi=1200, bbox_inches='tight')
	plt.close()

@njit
def project(rho, n_slices):
	projection = np.zeros(np.shape(rho[0,:,:]), dtype=np.float64)

	for i in range(n_slices):
		projection += rho[i,:,:]
	
	return projection