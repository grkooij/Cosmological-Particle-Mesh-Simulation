import numpy as np
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from numba import njit
import matplotlib.pyplot as plt
import mpl_scatter_density

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
	print("Plotting projection projection_density{}.png".format(np.int32(savestep)))
	cmap = colors.LinearSegmentedColormap.from_list("", ["black", "steelblue", "white", "yellow", "orange", "darkred"])
	cmap.set_bad((0,0,0))

	mass = (N_CELLS/N_PARTS)**3

	n_slices = np.int32(BOX_SIZE/depth)

	projection = project(rho, n_slices)
	fig, ax = plt.subplots()
	density = ax.imshow(projection/n_slices, extent = (0,BOX_SIZE,0,BOX_SIZE), norm=LogNorm(vmin=0.2, vmax=mass*25), cmap=cmap)
	cb = fig.colorbar(density, label='Density')
	cb.outline.set_edgecolor('white')
	cb.set_label('Density', color="white")
	cb.ax.yaxis.set_tick_params(color="white", which="both")

	cb.outline.set_edgecolor("white")
	cb.ax.minorticks_on()
	plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="white")

	ax.set_xlabel("Mpc/h")
	ax.set_ylabel("Mpc/h")
	ax.xaxis.label.set_color("white")
	ax.yaxis.label.set_color("white")
	ax.tick_params(axis='x', colors='white', which='both')
	ax.tick_params(axis='y', colors='white', which='both')
	plt.setp(ax.spines.values(), color="white")
	plt.setp([ax.get_xticklines(), ax.get_yticklines()], color="white")

	fig.patch.set_facecolor('xkcd:black')
	plt.style.context('dark_background')
	plt.savefig('Data/projection_density{}.png'.format(np.int32(savestep)), dpi=1200, bbox_inches='tight')
	plt.close()

# def plot_projection(rho, positions, savestep, depth):
# 	print("Plotting projection projection_density{}.png".format(savestep))
# 	cmap = colors.LinearSegmentedColormap.from_list("", ["black", "steelblue", "white", "yellow", "orange", "darkred"])
# 	cmap.set_bad((0,0,0))

# 	n_slices = np.int32(BOX_SIZE/depth)

# 	projection = project(rho, n_slices)
# 	particles, z = project2(projection, positions, n_slices)



# 	fig, ax = plt.subplots()
	
# 	# ax.imshow(projection/n_slices, extent = (0,BOX_SIZE,0,BOX_SIZE), norm=LogNorm(vmin=0.2, vmax=mass*10), cmap=cmap)
# 	# ax.scatter(particles[0], particles[1], c=projection, s=100)
# 	plt.scatter(x=particles[0][:], y=particles[1][:], marker="s", s=0.05)
# 	ax.set_xlabel("Mpc/h")
# 	ax.set_ylabel("Mpc/h")
# 	ax.xaxis.label.set_color("white")
# 	ax.yaxis.label.set_color("white")
# 	ax.tick_params(axis='x', colors='white', which='both')
# 	ax.tick_params(axis='y', colors='white', which='both')
# 	plt.setp(ax.spines.values(), color="white")
# 	plt.setp([ax.get_xticklines(), ax.get_yticklines()], color="white")
	
# 	fig.patch.set_facecolor('xkcd:black')
# 	plt.style.context('dark_background')
# 	plt.savefig('Data/projection_density{}.png'.format(savestep), dpi=1200, bbox_inches='tight')
# 	plt.close()

@njit
def project(rho, n_slices):
	projection = np.zeros(np.shape(rho[0,:,:]), dtype=np.float64)

	for i in range(n_slices):
		projection += rho[i,:,:]
	
	return projection

@njit
def project2(projection, positions, n_slices):

	for i in range(len(positions)):
		if not positions[2][i] > 0. and not positions[2][i] < n_slices:
			positions[2] = np.delete(positions[2], i)
			positions[1] = np.delete(positions[1], i)
			positions[0] = np.delete(positions[0], i)
	z = np.zeros(len(positions[0][:]))

	for i in range(len(positions)):
		x = np.int32(np.floor(positions[0][i]))
		y = np.int32(np.floor(positions[1][i]))
		z[i] = projection[x][y]

	return positions, z