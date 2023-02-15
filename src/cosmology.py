import numpy as np

class cosmo:
	def __init__(self, H0, omega_m0, omega_b, omega_lambda0, omega_k0, power, a_init):
		self.H0 = H0
		self.powspec = power
		self.omega_m0 = omega_m0
		self.omega_b = omega_b
		self.omega_lambda0 = omega_lambda0
		self.omega_k0 = omega_k0
		self.a_init = a_init
		self.Dt = Dt(a_init, [omega_m0, omega_lambda0, omega_k0])
		self.H01 = H(a_init, H0, [omega_m0, omega_lambda0, omega_k0])
		self.f0 = f(a_init, [omega_m0, omega_lambda0, omega_k0])

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
	#This function calculates the growing mode linear growth factor Dt

	omegaM = cosmology[0]
	omegaL = cosmology[1]
	omegaK = cosmology[2]
	
	return 5/2/omegaM/(omegaM**(4/7) - omegaL + (1+omegaM/2)*(1+omegaL/70))*a