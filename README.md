# Cosmological-Particle-Mesh-Simulation
Python code to create a simple cosmological particle mesh simulation for a lambda cold dark-matter universe.

Runtime for 1000 timesteps with 256^3 particles and 512^3 grid cells is, on my old i7 laptop, ~5 hours using ~8 GB RAM.

Set variables in pmesh.py and run with `python pmesh.py`.

Example .gif result for a thin projection of the density for a simulation including 256^3 particles on a 512^3 grid:
![256gif](https://github.com/grkooij/Cosmological-Particle-Mesh-Simulation/blob/master/256gif.gif)