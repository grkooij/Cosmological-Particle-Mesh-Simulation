# Cosmological-Particle-Mesh-Simulation
Python code to create a simple cosmological particle mesh simulation for a lambda cold dark-matter universe.

# Steps executed by the code
- Generate a Gaussian Random Field (GRF)
- Deposit particles uniformly on a uniform 3D grid
- Displace particles with Zel'dovich approximation based on the GRF
- Find density on the grid according to particle distribution using the CIC interpolation method
- Calculate the potential from Poisson's equation
- Integrate using leap frog to find positions, velocities
- Repeat steps 4,5,6

# Performance
The code is parallelized using pyFFTW for Fourier transforms, and with numba for integrations.

Runtime for 1000 timesteps with 256^3 particles and 512^3 grid cells is ~1 hour using ~7 GB RAM on a Windows os with a 13th gen 16-core Raptor lake.

# How to run the code
Configure the simulation parameters in configure_me.py and run the simulation with `python pmesh.py`.

# Gallery
Example PNG for a 15 Mpc/h projection of the density for a simulation including 512^3 particles on a 1024^3 grid:

![256gif](https://github.com/grkooij/Cosmological-Particle-Mesh-Simulation/blob/master/cosmological_simulation.png)
