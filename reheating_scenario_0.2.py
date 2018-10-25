#!/usr/bin/python
# -*- coding: latin-1 -*-


import math
import numpy as np
from scipy import special
import scipy.integrate as integrate
from scipy.optimize import fsolve, minimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time


#Initialize all constant values for simulation
t_0 = math.pow(10, 11)
t, m, l, b, xi = (t_0*1.1, math.pow(10,-15), math.pow(10,-20), 1.0, 0.0)
#G_N = 6.7071186*math.pow(10.0, -39.0)
#t_0 = 1.52*math.pow(10.0, -8.0)
G_N = 1.0
#t_0 = math.pow(10.0,-32.0)


#Integration settings
#Limit for scipy.quad
lmt = 1000000
dvm = 1000000000
tolerance=1.48e-35
rtolerance=1.48e-35
#Default tolerances
#tol=1.48e-08 
#rtol=1.48e-08

#Resolution for figures
resolution = 50


#Timing decorator
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap


#creation rate for massive particles
def Gamma_phi(t):
	airy_param = -math.pow(3.0*m*t/2.0, 2.0/3.0)
	ai, aip, bi, bip = special.airy(airy_param)	
	return 3*math.pow(m*b, 16.0/3.0)*t*(math.pow(ai, 2) + math.pow(bi, 2))/(32.0*b)



#print Gamma_phi(m,b,time_in_gev(1))


#Differential decay rate for massless particles
def Gamma_psi(t, a):
	g_p = math.pow(l, 2.0)*t*(math.pow(special.jv(a, m*t), 2.0) + math.pow(special.yv(a, m*t), 2.0))/(32.0*m)
	return g_p



#Index for Hankel functions
def alpha(n):
	a = math.sqrt(1.0 - n*(n - 2.0)*(6.0*xi - 1.0))/(2.0 + n)
	#Return only real part for now
	return a

#print Gamma_psi(10.0,10,10, alpha(4,0.0))


def f(t, t_0, n):
	a = alpha(n)
	r_p = integrate.romberg(lambda x: Gamma_psi(x, a), t_0, t, divmax=dvm, tol=tolerance, rtol=rtolerance)
	return r_p

def bessel(t, a):
	b = math.pow(l*t, 2.0)*(math.pow(special.jv(a, m*t), 2.0) - special.jv(a-1, m*t)*special.jv(a+1, m*t) - special.yv(a+1, m*t)*special.yv(a-1, m*t) + math.pow(special.yv(a, m*t), 2.0))/64.0 
	#print b
	return b 

@timing
def f2(t, t_0, n):
	a = alpha(n)
	return bessel(t, a)- bessel(t_0, a)

print f(1, 0.1, 1)
print f2(1, 0.1, 1)


def scale_factor(t):
	return math.pow(t, 1.0/3.0)

#print f(1.0,0.001, 4, 0.01,0.01)

#Energy density for massive particles
def rho_phi(t, t_0, n):
	r_p = math.pow(scale_factor(t), -3.0)*math.exp(-f(t, t_0, n))*integrate.romberg(lambda x: math.pow(scale_factor(x), 3.0)*Gamma_phi(x)*math.exp(f(x, t_0, n)), t_0, t, divmax=dvm, tol=tolerance, rtol=rtolerance)
	return r_p



#Energy density for massless particles
def rho_psi(t, t_0, n):
	a = alpha(n)
	r_p = math.pow(scale_factor(t), -4.0)*integrate.romberg(lambda x: Gamma_psi(x, a)*rho_phi(x, t_0, n)*math.pow(scale_factor(x),4.0), t_0, t, divmax=dvm, tol=tolerance, rtol=rtolerance)
	return r_p


#Energy density of stiff matter
def rho_stiff(t):
	return 1.0 / (24.0*math.pi*G_N*math.pow(t, 2.0))
#print "G_N: " + str(G_N)
#print rho_stiff(0.1, G_N)



################################################# AGORITHM ################################################# 


################################################# UINVERSE ENDS UP IN MATTER DOMINATED ERA ################################################# 

def matter_minus_stiff(t, t_0, n):
	return abs(rho_phi(t, t_0, n) - rho_stiff(t))

def rad_minus_stiff(t, t_0, n):
	return abs(rho_psi(t, t_0, n) - rho_stiff(t))



def eq_time(t, t_0, n):
	data = (t_0, n)
	t_eq = fsolve(matter_minus_stiff, t, args=data)
	r_p = rho_phi(t_eq, t_0, n)
	r_s = rho_stiff(t_eq)
	print("Absolute error: ") + str(r_p - r_s)
	return t_eq[0]

def eq_time2(t, t_0, n):
	data = (t_0, n)
	t_eq = fsolve(rad_minus_stiff, t, args=data)
	r_p = rho_psi(t_eq, t_0, n)
	r_s = rho_stiff(t_eq)
	print("Absolute error: ") + str(r_p - r_s)
	return t_eq[0]







#The constant n: 1 for stiff matter, 2 for radiation, 4 for matter




############### MATTER DOMINATED PHASE ##################
#Energy density of matter
def rho_phi_mat(t, n, eq_time, rho_eq):
	return math.pow(math.pow(eq_time/t, 2.0/3.0), 3.0)*(math.exp(-f(t, eq_time, n)) * rho_eq)



#Energy density of radiation
def rho_psi_mat(t, n, eq_time, rho_psi_eq, rho_eq):
	a = alpha(n)
	return math.pow(1.0 / math.pow(t, 2.0/3.0), 4.0)*integrate.romberg(lambda x: Gamma_psi(x, a)*rho_phi_mat(x, n, eq_time, rho_eq)*math.pow(math.pow(x, 2.0/3.0), 4.0), eq_time, t, divmax=dvm, tol=tolerance, rtol=rtolerance) + rho_psi_eq*math.pow(math.pow(eq_time/t, 2.0/3.0), 4.0)


#print rho_psi_mat(t, t_0, n, m, l, b, xi, eq_time, rho_psi_eq, rho_eq)



def matter_minus_rad(t, n, eq_time, rho_psi_eq, rho_eq):
	return abs(rho_phi_mat(t, n, eq_time, rho_eq) - rho_psi_mat(t, n, eq_time, rho_psi_eq, rho_eq))


#Solve when radiation dominated era begins
def eq_tau(t, n, eq_time, rho_psi_eq, rho_eq):
	data = (n, eq_time, rho_psi_eq, rho_eq)
	t_eq = fsolve(matter_minus_rad, t, args=data)
	r_p = rho_phi_mat(t_eq, n, eq_time, rho_eq)
	r_r = rho_psi_mat(t_eq, n, eq_time, rho_psi_eq, rho_eq)
	print("Absolute error: ") + str(r_p - r_r)
	return t_eq[0]



###################### Radiation dominated era #####################

#Energy density of radiation dominated era


def rho_phi_rad(t, n, eq_tau, rho_tau):
	return math.pow(math.pow(eq_tau/t, 1.0/2.0), 3.0)*(math.exp(-f(t, eq_tau, n)) * rho_tau)

#print rho_phi_rad(1500, 0.001, 2, 3, 1.1, 0.01)



def rho_psi_rad(t, n, eq_tau, rho_tau, psi_tau):
	a = alpha(n)
	return math.pow(math.pow(1.0/t, 1.0/2.0), 4.0)*integrate.romberg(lambda x: Gamma_psi(x, a)*rho_phi_rad(x, n, eq_tau, rho_tau)*math.pow(math.pow(x, 1.0/2.0), 4.0), eq_tau, t, divmax=dvm, tol=tolerance, rtol=rtolerance) + psi_tau*math.pow(math.pow(eq_tau/t, 1.0/2.0), 4.0)


def d_rho_psi_rad(t, n, eq_tau, rho_tau, psi_tau):
	a = alpha(n)
	return -2.0*integrate.romberg(lambda x: Gamma_psi(x, a)*rho_phi_rad(x, n, eq_tau, rho_tau)*math.pow(math.pow(x, 1.0/2.0), 4.0), eq_tau, t, divmax=dvm, tol=tolerance, rtol=rtolerance)/math.pow(t, 3.0) + Gamma_psi(t, a)*rho_phi_rad(t, n, eq_tau, rho_tau) - 2*psi_tau*math.pow(eq_tau, 2.0)*math.pow(1.0/t, 3.0)
'''
def d_rho_psi_rad(t, n, eq_tau, rho_tau, psi_tau):
	a = alpha(n)
	delta = 100000
	return (rho_psi_rad(t + delta, n, eq_tau, rho_tau, psi_tau)- rho_psi_rad(t, n, eq_tau, rho_tau, psi_tau))
'''
def neg_rho_psi_rad(t, n, eq_tau, rho_tau, psi_tau):
	return -1*rho_psi_rad(t, n, eq_tau, rho_tau, psi_tau)


def reheating_temperature(t, n, eq_tau, rho_tau, psi_tau):
	data = (n, eq_tau, rho_tau, psi_tau)
	cons = {'type': 'ineq', 'fun': lambda x: x - eq_tau}
	rh_time = minimize(neg_rho_psi_rad, t, args=data, jac=d_rho_psi_rad, constraints=cons, method='SLSQP')
	print rh_time
	return rh_time




######################## MAIN #####################

def reheating_time(t, t_0):
	global n
	#First calculate time when density of radiation equals that of stiff matter
	#n equals 1 at this time
	n = 1
	etime_rad = eq_time2(t, t_0, n)
	print etime_rad
	#Density of radiation at etime_rad
	phi_1 =  rho_phi(etime_rad, t_0, n)
	print "phi_1: " + str(phi_1)
	#Density of matter at etime_rad
	psi_1 = rho_psi(etime_rad, t_0, n)
	#If density of radiation is greater, we go straight to radiation dominated era
	print "psi_1: " + str(psi_1)
	if psi_1 > phi_1:
		print "Transition from stiff to radiation dominated era"
		#n eqauals 2 at this time
		n = 2
		psi_rad_init = psi_1
		phi_rad_init = phi_1
		print rho_phi_rad(etime_rad, n, etime_rad, phi_rad_init)
		print rho_psi_rad(etime_rad, n, etime_rad, phi_rad_init, psi_rad_init)

		plt.figure("Stiff matter dominated era")
		stiff_era = np.linspace(t_0, etime_rad, resolution) # 100 linearly spaced numbers
		mat = np.array([rho_phi(z, t_0, 1) for z in stiff_era])
		rad = np.array([rho_psi(z, t_0, 1) for z in stiff_era])
		stiff = np.array([rho_stiff(z) for z in stiff_era])
		plt.ylim(0, rad[-1]*1.1)
		plt.plot(stiff_era, mat, 'b-')
		plt.plot(stiff_era, rad, 'y-')
		plt.plot(stiff_era, stiff, 'r-')


		#Radiation dominated era begins, n equals 2
		n = 2
		
		
		#Sove for the reheating temperature
		reh_time = reheating_temperature(etime_rad*1.1, n, etime_rad, phi_rad_init, psi_rad_init)
		print -1*reh_time['fun']
		print rho_psi_rad(reh_time['x'], n, etime_rad, phi_rad_init, psi_rad_init)
		#Transfer reheating time to temperature
		print "Reheating temperature: "
		print math.pow(rho_psi_rad(reh_time['x'], n, etime_rad, phi_rad_init, psi_rad_init),1.0/4.0)


		print "RADIATION INTERSECTS STIFF: " + str(etime_rad)
		print "REHEATING TIME: " + str(reh_time['x'])
		print "REHEATING TIME LARGER THAN FIRST INTERSECTION (NEEDS TO BE TRUE): {}".format(str(float(reh_time['x']) > float(etime_rad)))

		plt.figure("Radiation dominated era")
		rad_era = np.linspace(etime_rad, reh_time['x']*1.1, resolution) # 100 linearly spaced numbers
		mat = np.array([rho_phi_rad(z, n, etime_rad, phi_rad_init) for z in rad_era])
		rad = np.array([rho_psi_rad(z, n, etime_rad, phi_rad_init, psi_rad_init) for z in rad_era])
		stiff = np.array([rho_stiff(z) for z in rad_era])
		plt.ylim(0, rad[-1]*1.1)
		plt.plot(rad_era, mat, 'b-')
		plt.plot(rad_era, rad, 'y-')
		plt.plot(rad_era, stiff, 'r-')

		plt.show()

	else:
		print "Transition from stiff to matter dominated era"
		#n equals 4 at this era
		#Calculate intersection of stiff and matter
		n = 1
		etime_mat = eq_time(t, t_0, n)
	
		print etime_mat
		plt.figure("Stiff matter dominated era")
		stiff_era = np.linspace(t_0, etime_mat*1.1, resolution) # 100 linearly spaced numbers
		mat = np.array([rho_phi(z, t_0, 1) for z in stiff_era])
		rad = np.array([rho_psi(z, t_0, 1) for z in stiff_era])
		stiff = np.array([rho_stiff(z) for z in stiff_era])
		plt.ylim(0, mat[-1]*1.1)
		plt.plot(stiff_era, mat, 'b-')
		plt.plot(stiff_era, rad, 'y-')
		plt.plot(stiff_era, stiff, 'r-')

		
		phi_1 =  rho_phi(etime_mat, t_0, n)
		psi_1 = rho_psi(etime_mat, t_0, n)
		stiff_1 = rho_stiff(etime_mat)
		print etime_mat
		print phi_1
		print psi_1
		print stiff_1

		#Matter dominated era begins, and n is now 4
		n = 4

		etime_rad = eq_tau(etime_mat*1.1, n, etime_mat, psi_1, phi_1)
		print etime_rad
		phi_2 = rho_phi_mat(etime_rad, n, etime_mat, phi_1)
		psi_2 = rho_psi_mat(etime_rad, n, etime_mat, psi_1, phi_1)
		print phi_2
		print psi_2

		plt.figure("Matter dominated era")
		mat_era = np.linspace(etime_mat, etime_rad*1.1, resolution) # 100 linearly spaced numbers
		mat = np.array([rho_phi_mat(z, n, etime_mat, phi_1) for z in mat_era])
		rad = np.array([rho_psi_mat(z, n, etime_mat, psi_1, phi_1) for z in mat_era])
		stiff = np.array([rho_stiff(z) for z in mat_era])
		plt.ylim(0, max(psi_1, psi_2)*1.1)
		plt.plot(mat_era, mat, 'b-')
		plt.plot(mat_era, rad, 'y-')
		plt.plot(mat_era, stiff, 'r-')
		
		#Radiation dominated era begins, n equals 2
		n = 2
		
		
		#Sove for the reheating temperature
		reh_time = reheating_temperature(etime_rad*1.1, n, etime_rad, phi_2, psi_2)
		print -1*reh_time['fun']
		print rho_psi_rad(reh_time['x'], n, etime_rad, phi_2, psi_2)
		#Transfer reheating time to temperature
		print "Reheating temperature: "
		print math.pow(rho_psi_rad(reh_time['x'], n, etime_rad, phi_2, psi_2),1.0/4.0)


		plt.figure("Radiation dominated era")
		rad_era = np.linspace(etime_rad, reh_time['x']*1.1, resolution) # 100 linearly spaced numbers
		mat = np.array([rho_phi_rad(z, n, etime_rad, phi_2) for z in rad_era])
		rad = np.array([rho_psi_rad(z, n, etime_rad, phi_2, psi_2) for z in rad_era])
		stiff = np.array([rho_stiff(z) for z in rad_era])
		plt.ylim(0, max(rad)*1.1)
		plt.plot(rad_era, mat, 'b-')
		plt.plot(rad_era, rad, 'y-')
		plt.plot(rad_era, stiff, 'r-')

		plt.show()



#reheating_time(t, t_0)
#n=1
#print eq_time(t, t_0, n)