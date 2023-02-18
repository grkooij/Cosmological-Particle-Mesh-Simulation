#Defining Simulation properties

###################################################
# General simulation settings
###################################################

N_PARTS          = 256 # Number of particles in one dimension
N_CELLS          = 512 # Number of cells in one dimension
BOX_SIZE         = 100 # Box size in Mpc/h in one dimension

N_CPU            = 16 # Number of cpu's used
RANDOM_SEED      = 38 # random seed to use for the Gaussian Random Field.

STEPS            = 1000 # Number of timesteps 
N_SAVE_FILES     = 100 # Number of save files/plotsteps/projections

PLOT_STEPS       = False
PLOT_PROJECTIONS = False
PLOT_GRF         = False
SAVE_DATA        = True
SAVE_DENSITY     = False
PRINT_STATUS     = True

###################################################
# Cosmology settings
###################################################
POWER                  = 1.00 # Harrison-Zeldovich spectrum has n~1
LCDM_TRANSFER_FUNCTION = True # Specifies whether to use a LCDM transfer function for the GRF, should only be used for POWER=1.0

OMEGA_M0               = 0.31 # Mass density
OMEGA_B0               = 0.04 # Baryon density
OMEGA_K0               = 0.00 # Curvature density
OMEGA_LAMBDA0          = 0.69 # Dark energy density
H0                     = 0.68 # Hubble
A_INIT                 = 0.01 # Initial scale factor
