import numpy as np
from numpy import inf as inf
from math import floor
from itertools import product
import scipy.integrate as integrate
import pyfftw
from time import time
import pandas as pd    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, clear_output, HTML, Image
import sys
import os
  
def Gaussian_Random_Field(Npart, Ngrid, Lx, n, h, Omega_0, Omega_b, Dt, n_cpu):
    
    #Generate two sets of uniform, independent random values
    seed=38
    np.random.seed(seed)
    u = np.zeros((Npart**3), dtype = 'float')
    v = np.zeros((Npart**3), dtype = 'float')
    
    n_parts = 0
    
    while n_parts < Npart**3:
        u1 = np.random.uniform(-1,1)
        v1 = np.random.uniform(-1,1)
        if 0. < u1**2 + v1**2 < 1.: 
            u[n_parts] = u1
            v[n_parts] = v1
            n_parts += 1
            
    #Transformation from uniform to Gaussian Random numbers
    s = u**2+v**2

    #Reshaping to a grid
    u = np.reshape(u, (Npart, Npart, Npart))
    v = np.reshape(v, (Npart, Npart, Npart))
    s = np.reshape(s, (Npart, Npart, Npart))

    #Polar Box-Muller transform

    f1 = u*(-2*np.log(s)/s)**0.5
    f2 = v*(-2*np.log(s)/s)**0.5

    #Finding the fourier frequency axes 

    Npix = Ngrid**3
    
    scale = 2*np.pi
    lxaxis = scale*np.fft.fftfreq(Npart)
    lyaxis = scale*np.fft.fftfreq(Npart)
    lzaxis = scale*np.fft.fftfreq(Npart)
    
    
    
    lz, ly, lx = np.meshgrid(lzaxis, lyaxis, lxaxis, indexing='ij')   
    
    kgrid = np.sqrt(lx**2 + ly**2 + lz**2)
    sigma2fluxt = 1
    
    if n >= 0.:
        summ = np.sum(np.power(kgrid, n, where=kgrid!=0))
        
        #summ = np.sum(f**n)
        A = sigma2fluxt * Npix**3 / (2*summ)
        
        Gamma = Omega_0 * h * np.exp(-Omega_b - Omega_b/Omega_0)
        q = kgrid / Gamma
        factor1 = np.sqrt(1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)
        factor2 = np.divide(np.log(1 + 2.34*q)**2, (2.34*q)**2, where=q!=0)
        #P= A * kgrid**n
        P = A * kgrid**n*factor2 / factor1
    
    else:
        kgrid_inverse = np.power(kgrid,-n,where=kgrid!=0)
        div_kgrid = np.divide(1, kgrid_inverse, where=kgrid_inverse!=0)
        summ = np.sum(div_kgrid)
        A = sigma2fluxt * Npix**3 / (2*summ)
        Gamma = Omega_0 * h * np.exp(-Omega_b - Omega_b/Omega_0)
        q = kgrid/Gamma
        factor1 = np.sqrt(1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)
        factor2 = np.divide(np.log(1 + 2.34*q)**2, (2.34*q)**2, where=q!=0)
        P = A * div_kgrid * factor2 / factor1


    #Multiplying by the sqrt of the power spectrum to solve the variance problem

    f1 = np.sqrt(P*Dt**2) * f1
    f2 = np.sqrt(P*Dt**2) * f2
    
    density_k = f1 + 1j*f2
      
    density_real = np.fft.ifftn(density_k).real
    
    return density_real
    
    
def density_potential(x_dat, y_dat, z_dat, mass, Npart, Ngrid, Lx, a, Omega_0, steps, t, s, n_cpu):
 
    #Create a new grid
    grid = np.zeros([Ngrid, Ngrid, Ngrid], dtype='float')
        
    #cell center coordinates
    x_c = np.floor(x_dat).astype(int)
    y_c = np.floor(y_dat).astype(int)
    z_c = np.floor(z_dat).astype(int)
        
    #Calculating contributions
    
    d_x = x_dat - x_c
    d_y = y_dat - y_c
    d_z = z_dat - z_c
    
    
    t_x = 1 - d_x
    t_y = 1 - d_y
    t_z = 1 - d_z
    
                
        #Create periodic indices
                    
    X = (x_c+1)%Ngrid
    Y = (y_c+1)%Ngrid
    Z = (z_c+1)%Ngrid
                    
        #Update the grid
    grid[z_c,y_c,x_c] += mass*t_x*t_y*t_z
    #grid[z_c,y_c,x_c] += mass          
        #CIC implementation
        
                        
    grid[z_c,y_c,X] += mass*d_x*t_y*t_z
    grid[z_c,Y,x_c] += mass*t_x*d_y*t_z
    grid[Z,y_c,x_c] += mass*t_x*t_y*d_z
        
    grid[z_c,Y,X] += mass*d_x*d_y*t_z
    grid[Z,Y,x_c] += mass*t_x*d_y*d_z
    grid[Z,y_c,X] += mass*d_x*t_y*d_z
        
    grid[Z,Y,X] += mass*d_x*d_y*d_z
    
    if s == 0:
        import pandas as pd
        
        df1 = pd.DataFrame([np.reshape(grid[:,:,25], Ngrid*Ngrid).real])          
        df1.to_csv('Data/NbodyDATA_density.csv', mode='w', index = False, header=False, float_format = '%.3f')
    
    if t == steps - 2 and s !=0:
        import pandas as pd
        
        df2 = pd.DataFrame([np.reshape(grid[:,:,25], Ngrid*Ngrid).real])          
        df2.to_csv('Data/NbodyDATA_density.csv', mode='a', index = False, header=False, float_format = '%.3f')
    
    fft_grid = np.zeros([Ngrid, Ngrid, Ngrid], dtype='cfloat')
    
    fft_object = pyfftw.FFTW(grid.astype('cfloat'), fft_grid, direction = 'FFTW_FORWARD', axes=(0,1,2), threads = n_cpu)
    fft_density = fft_object()
    
    scale = 2*np.pi
    
    k_x = scale*np.fft.fftfreq(Ngrid)
    k_y = scale*np.fft.fftfreq(Ngrid)
    k_z = scale*np.fft.fftfreq(Ngrid)
    
    force_resolution = Lx/Ngrid
    
    ky, kz, kx = np.meshgrid(k_z, k_y, k_x)

    k_squared = np.sin(kz/2)**2 + np.sin(ky/2)**2 + np.sin(kx/2)**2

    
    Greens_operator = -3*Omega_0/8/a*np.divide(1, k_squared, where=k_squared!=0)*force_resolution**2
    
    grid = Greens_operator*fft_density
    
    fft_grid = np.zeros([Ngrid, Ngrid, Ngrid], dtype='cfloat')
    ifft_object = pyfftw.FFTW(grid, fft_grid, direction = 'FFTW_BACKWARD',axes=(0,1,2), threads = n_cpu)
    grid = (ifft_object().real).astype('float')
    
    #Hier moet die potential op sla gedeelte
    if s == 0:
        import pandas as pd
        
        df3 = pd.DataFrame([np.reshape(grid[:,:,25], Ngrid*Ngrid).real])          
        df3.to_csv('Data/NbodyDATA_potential.csv', mode='w', index = False, header=False, float_format = '%.3f')
    
    if t == steps - 2 and s !=0:
        import pandas as pd
        
        df4 = pd.DataFrame([np.reshape(grid[:,:,25], Ngrid*Ngrid).real])          
        df4.to_csv('Data/NbodyDATA_potential.csv', mode='a', index = False, header=False, float_format = '%.3f')
          
    return grid.real


def multi_particle_vector(x_dat, y_dat, z_dat, vx_dat, vy_dat, vz_dat, a_val, f_a, f_a1, da, Ngrid, potentials,force_resolution):
    #Kick step

    #Defining cell coordinates from position and velocity data
    
    x = np.floor(x_dat).astype(int)
    y = np.floor(y_dat).astype(int)
    z = np.floor(z_dat).astype(int)
    
    d_x = x_dat - x
    d_y = y_dat - y
    d_z = z_dat - z

    t_x = 1 - d_x
    t_y = 1 - d_y
    t_z = 1 - d_z

    #Now we make them periodic
    
    X = (x+1)%Ngrid
    Y = (y+1)%Ngrid
    Z = (z+1)%Ngrid
    X2 = (x+2)%Ngrid
    Y2 = (y+2)%Ngrid
    Z2 = (z+2)%Ngrid

    #Interpolating for the CIC implementation

    #Interpolating for the accelerations in x
    gx = (-potentials[z,y,X] + potentials[z,y,x-1])/2.
    gx_x = (-potentials[z,y,X2] + potentials[z,y,x])/2.
    gx_y = (-potentials[z,Y,X] + potentials[z,Y,x-1])/2.
    gx_z = (-potentials[Z,y,X] + potentials[Z,y,x-1])/2.
    gx_xy = (-potentials[z,Y,X2] + potentials[z,Y,x])/2.
    gx_xz = (-potentials[Z,y,X2] + potentials[Z,y,x])/2.
    gx_yz = (-potentials[Z,Y,X] + potentials[Z,Y,x-1])/2.
    gx_xyz = (-potentials[Z,Y,X2] + potentials[Z,Y,x])/2.

    #Interpolating for the accelerations in y
    gy = (-potentials[z,Y,x] + potentials[z,y-1,x])/2.
    gy_x = (-potentials[z,Y,X] + potentials[z,y-1,X])/2.
    gy_y = (-potentials[z,Y2,x] + potentials[z,y,x])/2.
    gy_z = (-potentials[Z,Y,x] + potentials[Z,y-1,x])/2.
    gy_xy = (-potentials[z,Y2,X] + potentials[z,y,X])/2.
    gy_xz = (-potentials[Z,Y,X] + potentials[Z,y-1,X])/2.
    gy_yz = (-potentials[Z,Y2,x] + potentials[Z,y,x])/2.
    gy_xyz = (-potentials[Z,Y2,X] + potentials[Z,y,X])/2.

    #Interpolating for the accelerations in z
    gz = (-potentials[Z,y,x] + potentials[z-1,y,x])/2.
    gz_x = (-potentials[Z,y,X] + potentials[z-1,y,X])/2.
    gz_y = (-potentials[Z,Y,x] + potentials[z-1,Y,x])/2.
    gz_z = (-potentials[Z2,y,x] + potentials[z,y,x])/2.
    gz_xy = (-potentials[Z,Y,X] + potentials[z-1,Y,X])/2.
    gz_xz = (-potentials[Z2,y,X] + potentials[z,y,X])/2.
    gz_yz = (-potentials[Z2,Y,x] + potentials[z,Y,x])/2.
    gz_xyz = (-potentials[Z2,Y,X] + potentials[z,Y,X])/2.

    t1 = t_x*t_y*t_z
    t2 = d_x*t_y*t_z
    t3 = t_x*d_y*t_z
    t4 = t_x*t_y*d_z
    t5 = d_x*d_y*t_z
    t6 = d_x*t_y*d_z
    t7 = t_x*d_y*d_z
    t8 = d_x*d_y*d_z
    
    #Parent acceleration
    g_p_x = gx*t1 + gx_x*t2 + gx_y*t3 + gx_z*t4 + gx_xy*t5 + gx_xz*t6 + gx_yz*t7 + gx_xyz*t8
    g_p_y = gy*t1 + gy_x*t2 + gy_y*t3 + gy_z*t4 + gy_xy*t5 + gy_xz*t6 + gy_yz*t7 + gy_xyz*t8
    g_p_z = gz*t1 + gz_x*t2 + gz_y*t3 + gz_z*t4 + gz_xy*t5 + gz_xz*t6 + gz_yz*t7 + gz_xyz*t8
    
    
    #Leapfrog
    vx_datnew = vx_dat + da*f_a1*g_p_x/force_resolution
    vy_datnew = vy_dat + da*f_a1*g_p_y/force_resolution
    vz_datnew = vz_dat + da*f_a1*g_p_z/force_resolution
        
    x_dat = (x_dat + .5*da*vx_dat/a_val**2*f_a + .5*da*vx_datnew/(a_val+da)**2*f_a1) % Ngrid 
    y_dat = (y_dat + .5*da*vy_dat/a_val**2*f_a + .5*da*vy_datnew/(a_val+da)**2*f_a1) % Ngrid
    z_dat = (z_dat + .5*da*vz_dat/a_val**2*f_a + .5*da*vz_datnew/(a_val+da)**2*f_a1) % Ngrid
    
    return x_dat, y_dat, z_dat, vx_datnew, vy_datnew, vz_datnew

def H(a, H0, cosmology):
    
    omegaM = cosmology[0]
    omegaL = cosmology[1]
    omegaK = cosmology[2]
    
    return np.sqrt(H0**2*(omegaM/a**3 + omega_k0/a**2 + omega_lambda0))

def f(a, cosmology):
    
    omegaM = cosmology[0]
    omegaL = cosmology[1]
    omegaK = cosmology[2]
    
    return 1/np.sqrt((omegaM + omegaK*a + omegaL*a**3)/a)

def Dt(a, cosmology):
    #The linear growth factor Dt can be approximated with     
    #Slide 80 http://popia.ft.uam.es/aknebe/page3/files/ComputationalCosmology/05ICs.pdf
    
    omegaM = cosmology[0]
    omegaL = cosmology[1]
    omegaK = cosmology[2]
    
    return 5/2/omegaM/(omegaM**(4/7) - omegaL + (1+omegaM/2)*(1+omegaL/70))*a

def Zeldovich(density_real, Lx, Npart, Ngrid, H0, f0, Dt, a_init):
    
    density_k = np.fft.fftn(density_real)
    
    #Creating the Fourier axes 
    scale = 2*np.pi
    lxaxis = scale*np.fft.fftfreq(Npart)
    lyaxis = scale*np.fft.fftfreq(Npart)
    lzaxis = scale*np.fft.fftfreq(Npart)
    
    lz, ly, lx = np.meshgrid(lzaxis, lyaxis, lxaxis, indexing='ij')   
    
    kgrid = np.sqrt(lx**2 + ly**2 + lz**2)
    
    #k squared operator
    del_sq = -kgrid**2
    
    #mpc per grid cell for a smaller grid of Npart
    mass_resolution = Lx/Npart
    
    #Calculating potential
    potential = np.divide(density_k, del_sq, where=del_sq!=0)*mass_resolution**2
    
    grad_x_operator = -1.j *lx
    grad_y_operator = -1.j *ly
    grad_z_operator = -1.j *lz
    
    #Computing the displacement field in Fourier space
    
    ZAx = grad_x_operator*potential
    ZAy = grad_y_operator*potential
    ZAz = grad_z_operator*potential
    
    #Auxiliary grid required for PyFFTW
    fft_grid = np.zeros([Npart, Npart, Npart], dtype='cfloat')
    
    #Creating FFT objects to obtain wisdom 
    fft_ZAx_obj = pyfftw.FFTW(ZAx.astype('cfloat'), fft_grid, direction = 'FFTW_BACKWARD', axes=(0,1,2), threads = n_cpu)
    fft_ZAy_obj = pyfftw.FFTW(ZAy.astype('cfloat'), fft_grid, direction = 'FFTW_BACKWARD', axes=(0,1,2), threads = n_cpu)
    fft_ZAz_obj = pyfftw.FFTW(ZAz.astype('cfloat'), fft_grid, direction = 'FFTW_BACKWARD', axes=(0,1,2), threads = n_cpu)
    
    #Calling the objects to start the FFT's
    ZAx = fft_ZAx_obj()
    ZAy = fft_ZAy_obj()
    ZAz = fft_ZAz_obj()
    
    DFx = np.reshape(ZAx/mass_resolution, (Npart**3)).real
    DFy = np.reshape(ZAy/mass_resolution, (Npart**3)).real
    DFz = np.reshape(ZAz/mass_resolution, (Npart**3)).real
    
    #Define unperturbed lattice positions
    #Has to be evenly distributed
    periodic_space = Ngrid/Npart
    
    x_space = np.linspace(0, Ngrid-periodic_space, Npart) + 0.5
    y_space = np.linspace(0, Ngrid-periodic_space, Npart) + 0.5
    z_space = np.linspace(0, Ngrid-periodic_space, Npart) + 0.5

    z_unpert, y_unpert, x_unpert = np.meshgrid(z_space, y_space, x_space, indexing='ij')
    
    #And we perturb using Zel'dovich approximation
    
    force_resolution = Lx/Ngrid
    x_dat = (np.reshape(x_unpert, Npart*Npart*Npart) - Dt*DFx/force_resolution) % Ngrid
    y_dat = (np.reshape(y_unpert, Npart*Npart*Npart) - Dt*DFy/force_resolution) % Ngrid
    z_dat = (np.reshape(z_unpert, Npart*Npart*Npart) - Dt*DFz/force_resolution) % Ngrid
    
    
    vx_dat = -a_init*f0*H0*Dt*DFx/force_resolution
    vy_dat = -a_init*f0*H0*Dt*DFy/force_resolution
    vz_dat = -a_init*f0*H0*Dt*DFz/force_resolution
    

    return x_dat,y_dat,z_dat,vx_dat,vy_dat,vz_dat

def save_file(data, step):
    import os
    filename = 'Data/'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df1 = pd.DataFrame(data[0][:,np.newaxis], dtype='float')
    df2 = pd.DataFrame(data[1][:,np.newaxis], dtype='float')
    df3 = pd.DataFrame(data[2][:,np.newaxis], dtype='float')
    df4 = pd.DataFrame(data[3][:,np.newaxis], dtype='float')
    df5 = pd.DataFrame(data[4][:,np.newaxis], dtype='float')
    df6 = pd.DataFrame(data[5][:,np.newaxis], dtype='float')
    
    df1.to_csv('Data/NbodyDATA_x{}.csv'.format(step), mode='w', index=False, header=False, float_format = '%.3f')
    df2.to_csv('Data/NbodyDATA_y{}.csv'.format(step), mode='w', index=False, header=False, float_format = '%.3f')
    df3.to_csv('Data/NbodyDATA_z{}.csv'.format(step), mode='w', index=False, header=False, float_format = '%.3f')
    df4.to_csv('Data/NbodyDATA_vx{}.csv'.format(step), mode='w', index=False, header=False, float_format = '%.3f')
    df5.to_csv('Data/NbodyDATA_vy{}.csv'.format(step), mode='w', index=False, header=False, float_format = '%.3f')
    df6.to_csv('Data/NbodyDATA_vz{}.csv'.format(step), mode='w', index=False, header=False, float_format = '%.3f')
    return

def simulator(power, Npart, Ngrid, Lx, steps, savesteps, cosmology, h, a_init, Omega_b, n_cpu):
    H0 = h
    #Lx = Lx/h
    particle_mass = 1.32*10**5*(cosmology[0]*h**2)*(Lx/(Npart/128))**3
    print('Particle mass [M_odot] for this configuration:', particle_mass)

    dens_contrast = (Ngrid/Npart)**3
    print('Particle mass in code units:', dens_contrast)
    
    #Resolution of mpc/cell
    force_resolution = Lx/Ngrid
    
    #Initial conditions 
    a = np.linspace(a_init, 1, steps*savesteps)
    da = a[1] - a_init

    #Cosmology
    H01 = H(a_init,H0, cosmology)
    f0 = f(a_init, cosmology)
    Dt0 = Dt(a_init, cosmology)
    
    #First step is the Gaussian Random Field
    print('Preparing the Gaussian random field...')
    density_real = Gaussian_Random_Field(Npart, Ngrid, Lx, power, H0, cosmology[0], Omega_b, Dt0, n_cpu)
    
    #Zeldovich displacement
    x_dat,y_dat,z_dat,vx_dat,vy_dat,vz_dat = Zeldovich(density_real, Lx, Npart, Ngrid, H01, f0, Dt0, a_init)
    print('Zeldovich velocities:', vx_dat)
    
    save_file([x_dat,y_dat,z_dat,vx_dat,vy_dat,vz_dat], 0)    
 
    #We then loop over the amount of save steps we want
    print('Starting the integrations...')
    
    for s in range(savesteps):    
        
        #Looping over the number of steps per save step 
        for t in range(steps-1):
            
            a_ind =  s*steps + t
            
            #Percentage finished
            percentile = 100*a_ind/(steps*savesteps) + 0.2

            #Here we calculate the densities, potentials, forces and the velocities for each step
            
            facto = f(a_init, cosmology)
            factoplus1 = f(a_init+da, cosmology)
            a_val = a_init
            
            #Solve the Poisson-Vlasov equation        
            potentials = density_potential(x_dat, y_dat, z_dat, dens_contrast, Npart, Ngrid, Lx, a_val, cosmology[0], steps, t, s, n_cpu) 

            #Integrate 
            x_dat, y_dat, z_dat, vx_dat, vy_dat, vz_dat = multi_particle_vector(x_dat, y_dat, z_dat, vx_dat, vy_dat, vz_dat, a_val, facto, factoplus1, da, Ngrid, potentials, force_resolution)
            
            #Updating the time
            a_init += da
            
            #Saving the data after savesteps
            
            if t == steps-2:
                
                #Unit conversion to physical
                unit_conv_pos = 7.8*(Lx/(Ngrid/128))/10**3 #Mpc
                unit_conv_vel = 0.781*Lx*h/(a_init*Ngrid/128) #km/s

                save_file([unit_conv_pos*x_dat,unit_conv_pos*y_dat,unit_conv_pos*z_dat,unit_conv_vel*vx_dat,unit_conv_vel*vy_dat,unit_conv_vel*vz_dat], s+1)
                
                
                clear_output(wait=True)
                print('{}%'.format(percentile))
                sys.stdout.flush()
    return


from time import time
import pandas as pd

#Harrison Zeldovich spectrum has p=1
power = 1.


#The size per grid cell is determined in the GRF, since there we have Lx/Npart. This has to be 
#transformed. A probable fix is that we should calculate the full displacement field using Ngrid
#And either interpolate, or require Ngrid to be a multiple of Npart.

Npart = 128 #number of particles
Ngrid = 256 #number of grid cells
Length_x = 100 #Mpc

# number of cpu's in the machine the simulation will run on (used for faster fft)
n_cpu = 16
    
stepspersavestep = 10
savesteps = 100
#Lambda CDM

omega_m0 = 0.31
omega_k0 = 0. 
omega_lambda0 = 0.69
Omega_b = 0.04

H0_LCDM = 0.68
H0_EdS = 0.7

a_init = 0.01

LCDM = [omega_m0, omega_lambda0, omega_k0]

#In modern parlance, the Einstein–de Sitter universe can be described as 
#a cosmological model for a flat matter-only
#Friedmann–Lemaître–Robertson–Walker metric (FLRW) universe.
EdS = [1.0, 0., 0.]

Empty = [0., 0., 1.0]

deSitter = [0., 1.0, 0.]

Closed = [6., 0., -5.]

import math
import threading
import time

start_time = time.time()

simulator(power, Npart, Ngrid, Length_x, stepspersavestep, savesteps, LCDM, H0_LCDM, a_init, Omega_b, n_cpu)

print("Finished in")
print("--- %s seconds ---" % (time.time() - start_time))
