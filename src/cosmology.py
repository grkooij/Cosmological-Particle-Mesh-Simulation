import numpy as np

from configure_me import H0, OMEGA_K0, OMEGA_LAMBDA0, OMEGA_M0, A_INIT

class cosmo:
	def __init__(self):
		self.Dt = Dt(A_INIT, [OMEGA_M0, OMEGA_LAMBDA0, OMEGA_K0])
		self.H01 = H(A_INIT, H0, [OMEGA_M0, OMEGA_LAMBDA0, OMEGA_K0])
		self.f0 = f(A_INIT, [OMEGA_M0, OMEGA_LAMBDA0, OMEGA_K0])

def H(a, H0, cosmology):
	#This function calculates the Hubble constant

	omega_m0 = cosmology[0]
	omega_l0 = cosmology[1]
	omega_k0 = cosmology[2]
	
	return np.sqrt(H0**2*(omega_m0/a**3 + omega_k0/a**2 + omega_l0))

def f(a, cosmology):
	#This function calculates the reciprocal of the time derivative of a time H0

	omegaM = cosmology[0]
	omegaL = cosmology[1]
	omegaK = cosmology[2]
	
	return 1/np.sqrt((omegaM + omegaK*a + omegaL*a**3)/a)

def Dt(a, cosmology):
	#This function calculates the growing mode linear growth factor Dt as an approximation

	omegaM = cosmology[0]
	omegaL = cosmology[1]
	omegaK = cosmology[2]
	
	return 5/2/omegaM/(omegaM**(4/7) - omegaL + (1+omegaM/2)*(1+omegaL/70))*a