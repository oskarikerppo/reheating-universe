#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division
import numpy as np
from scipy import special
from scipy.special import hankel1 as h1
from scipy.special import hankel2 as h2
import scipy.integrate as integrate
from scipy.optimize import fsolve, minimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import pickle as pickle
import sys


#The constant n: 1 for stiff matter, 2 for radiation, 4 for matter



#Integration settings
#Limit for scipy.quad
lmt = 1000000
dvm = 50
tolerance=1.48e-08
rtolerance=1.48e-08
#Default tolerances
#tol=1.48e-08 
#rtol=1.48e-08

#Tolerances for quad
eabs=1.49e-08
erel=1.49e-08
lmt=10000


#Resolution for figures
resolution = 50


#Timing decorator
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0))
        return ret
    return wrap

#creation rate for massive particles
def Gamma_phi(t, m, b):
	airy_param = -((3*m*t/2.0)**(2/3.0))
	ai, aip, bi, bip = special.airy(airy_param)	
	return 3*((m*b)**(13/3.0))*t*(ai**2 + bi**2)/(32.0*b)

#Differential decay rate for massless particles
def Gamma_chi(t, m, a, l, n):
	mt = m*t
	return np.real((l**2)*t*h1(a, mt)*h2(a, mt)/32)

def Gamma_psi(t, m, a, h, n):
	mt = m*t
	return np.real((h**2)*t*(h1(a - 1, mt) - h1(a + 1, mt) + ((1 - n)/t)*h1(a, mt))*(h2(a - 1, mt) - h2(a + 1, mt)+((1 - n)/t)*h2(a, mt))/128)





#Index for Hankel functions
def alpha(n, xi):
        #minkowskian return 1/2.0
	return ((1 - n*(n - 2)*(6*xi - 1))**(1/2.0))/(2.0 + n)

def bessel(t, m, a, l):
	return ((l*t)**2.0)*((special.jv(a, m*t)**2) - special.jv(a-1, m*t)*special.jv(a+1, m*t) - special.yv(a+1, m*t)*special.yv(a-1, m*t) + (special.yv(a, m*t)**2))/64.0
 
def f(t, t_0, n, xi, m, l, h):
	a = alpha(n, xi)
	return integrate.quad(lambda x: Gamma_chi(x, m, a, l, n) + Gamma_psi(x, m, a, h, n), t_0, t, epsabs=eabs, epsrel=erel, limit=lmt)[0]

def scale_factor(t):
	return t**(1/3.0)


#Energy density for massive particles
def rho_phi(t, t_0, n, xi, m, l, h, b):
	return (scale_factor(t)**(-3.0))*np.exp(-f(t, t_0, n, xi, m, l, h))*integrate.quad(lambda x: (scale_factor(x)**3.0)*Gamma_phi(x, m, b)*np.exp(f(x, t_0, n, xi, m, l, h)), t_0, t, epsabs=eabs, epsrel=erel, limit=lmt)[0]

#Energy density for massless particles
def rho_psi(t, t_0, n, xi, m, l, h, b):
	a = alpha(n, xi)
	return (scale_factor(t)**(-4.0))*integrate.quad(lambda x: (Gamma_psi(x, m, a, h, n) + Gamma_chi(x, m, a, l, n))*rho_phi(x, t_0, n, xi, m, l, h, b)*(scale_factor(x)**4.0), t_0, t, epsabs=eabs, epsrel=erel, limit=lmt)[0]

#Energy density of stiff matter
def rho_stiff(t, G_N):
	return 1 / (24*np.pi*G_N*(t**2))

def rho_stiff_mat(t, t_0, G_N):
	return (t_0**4) / (24*np.pi*G_N*(t_0**2)*(t**4))

def rho_stiff_rad(t, t_0, t_1, G_N):
	return (t_0**4) * (t_1**3) / (24*np.pi*G_N*(t_0**2)*(t_1**4)*(t**3))

def rho_stiff_rad_no_mat(t, t_0, G_N):
	return (t_0**3) / (24*np.pi*G_N*(t_0**2)*(t**3))



################################################# AGORITHM ################################################# 


################################################# UINVERSE ENDS UP IN MATTER DOMINATED ERA ################################################# 

def matter_minus_stiff(t, t_0, n, xi, m, l, h, b, G_N):
	return float(rho_stiff(t, G_N) - rho_phi(t, t_0, n, xi, m, l, h, b))

def rad_minus_stiff(t, t_0, n, xi, m, l, h, b, G_N):
	return float(rho_stiff(t, G_N) - rho_psi(t, t_0, n, xi, m, l, h, b))

def eq_time(t, t_0, n, xi, m, l, h, b, G_N):
	'''
	order = 0
	t = t_0*10**order
	difference = matter_minus_stiff(t, t_0, n, xi, m, l, b, G_N)
	while difference < 0:
		order += 0.1
		t = t_0*10**order
		difference = matter_minus_stiff(t, t_0, n, xi, m, l, b, G_N)
	order -= 0.1
	'''
	data = (t_0, n, xi, m, l, h, b, G_N)
	t_eq = fsolve(matter_minus_stiff, t, args=data)
	r_p = rho_phi(t_eq, t_0, n, xi, m, l, h, b)
	r_s = rho_stiff(t_eq, G_N)
	#print("Absolute error: ") + str(r_p - r_s)
	return t_eq[0]

def eq_time2(t, t_0, n, xi, m, l, h, b, G_N):
	'''
	order = 1
	t = t_0*10**order
	difference = rad_minus_stiff(t, t_0, n, xi, m, l, b, G_N)
	while difference < 0:
		order += 0.1
		t = t_0*10**order
		difference = rad_minus_stiff(t, t_0, n, xi, m, l, b, G_N)
	order -= 0.1
	'''
	data = (t_0, n, xi, m, l, h, b, G_N)
	t_eq = fsolve(rad_minus_stiff, t, args=data)
	r_p = rho_psi(t_eq, t_0, n, xi, m, l, h, b)
	r_s = rho_stiff(t_eq, G_N)
	#print("Absolute error: ") + str(r_p - r_s)
	return t_eq[0]



############### MATTER DOMINATED PHASE ##################
#Energy density of matter
def rho_phi_mat(t, n, xi, m, l, h, eq_time, rho_eq):
	return ((eq_time/t)**2.0)*(np.exp(-f(t, eq_time, n, xi, m, l, h)) * rho_eq)

#Energy density of radiation
def rho_psi_mat(t, n, xi, m, l, h, eq_time, rho_psi_eq, rho_eq):
	a = alpha(n, xi)
	return (1/(t**(8/3.0)))*integrate.quad(lambda x: (Gamma_psi(x, m, a, h, n) + Gamma_chi(x, m, a, l, n))*rho_phi_mat(x, n, xi, m, l, h, eq_time, rho_eq)*(x**(8/3.0)), eq_time, t, epsabs=eabs, epsrel=erel, limit=lmt)[0] + rho_psi_eq*(eq_time/t)**(8/3.0)

def matter_minus_rad(t, n, xi, m, l, h, eq_time, rho_psi_eq, rho_eq):
	return rho_phi_mat(t, n, xi, m, l, h, eq_time, rho_eq) - rho_psi_mat(t, n, xi, m, l, h, eq_time, rho_psi_eq, rho_eq)

#Solve when radiation dominated era begins
def eq_tau(t, n, xi, m, l, h, eq_time, rho_psi_eq, rho_eq):
	'''
	order = 1
	t = eq_time*10**order
	difference = matter_minus_rad(t, n, xi, m, l, eq_time, rho_psi_eq, rho_eq)
	while difference > 0:
		order += 0.1
		t = eq_time*10**order
		difference = matter_minus_rad(t, n, xi, m, l, eq_time, rho_psi_eq, rho_eq)
	order -= 0.1
	'''
	data = (n, xi, m, l, h, eq_time, rho_psi_eq, rho_eq)
	t_eq = fsolve(matter_minus_rad, t, args=data)
	r_p = rho_phi_mat(t_eq, n, xi, m, l, h, eq_time, rho_eq)
	r_r = rho_psi_mat(t_eq, n, xi, m, l, h, eq_time, rho_psi_eq, rho_eq)
	#print("Absolute error: ") + str(r_p - r_r)
	return t_eq[0]



###################### Radiation dominated era #####################

#Energy density of radiation dominated era
def rho_phi_rad(t, n, xi, m, l, h, eq_tau, rho_tau):
	return ((eq_tau/t)**(3/2))*(np.exp(-f(t, eq_tau, n, xi, m, l, h)) * rho_tau)

def rho_psi_rad(t, n, xi, m, l, h, eq_tau, rho_tau, psi_tau):
	a = alpha(n, xi)
	return ((1/t)**2)*integrate.quad(lambda x: (Gamma_chi(x, m, a, l, n) + Gamma_psi(x, m, a, h, n))*rho_phi_rad(x, n, xi, m, l, h, eq_tau, rho_tau)*(x**2), eq_tau, t, epsabs=eabs, epsrel=erel, limit=lmt)[0] + psi_tau*((eq_tau/t)**2)

def d_rho_psi_rad(t, n, xi, m, l, h, eq_tau, rho_tau, psi_tau):
	a = alpha(n, xi)
	return -2*integrate.quad(lambda x: (Gamma_chi(x, m, a, l, n) + Gamma_psi(x, m, a, h, n))*rho_phi_rad(x, n, xi, m, l, h, eq_tau, rho_tau)*(x**2), eq_tau, t, epsabs=eabs, epsrel=erel, limit=lmt)[0]/(t**3) + Gamma_psi(t, m, a, h, n)*rho_phi_rad(t, n, xi, m, l, h, eq_tau, rho_tau) - 2*psi_tau*(eq_tau**2)/(t**3)

def r_time(t, n, xi, m, l, h, eq_tau, rho_tau, psi_tau):
	#First calculate derivative of radiation density at equal time
	d_1 = d_rho_psi_rad(eq_tau, n, xi, m, l, h, eq_tau, rho_tau, psi_tau)
	if d_1 < 0:
		#The maximum value is the initial value
		return eq_tau
	data = (n, xi, m, l, h, eq_tau, rho_tau, psi_tau)
	rt = fsolve(d_rho_psi_rad, t, data)
	#print rt
	return rt[0]





######################## MAIN #####################

def reheating_time(t, t_0, m, l, h, b, xi, G_N, plot):
	#First calculate time when density of radiation equals that of stiff matter
	#n equals 1 at this time
	n = 1
	#Must use function eq_time with new Gamma
	etime_rad = eq_time(t, t_0, n, xi, m, l, h, b, G_N)
	#print etime_rad
	#Density of radiation at etime_rad
	phi_1 =  rho_phi(etime_rad, t_0, n, xi, m, l, h, b)
	#print "phi_1: " + str(phi_1)
	#Density of matter at etime_rad
	psi_1 = rho_psi(etime_rad, t_0, n, xi, m, l, h, b)
	#If density of radiation is greater, we go straight to radiation dominated era
	#print "psi_1: " + str(psi_1)
	#print "stiff_1: " + str(rho_stiff(etime_rad, G_N)) 
	if psi_1 > phi_1:
		etime_rad = eq_time2(t, t_0, n, xi, m, l, h, b, G_N)
		phi_1 =  rho_phi(etime_rad, t_0, n, xi, m, l, h, b)
		psi_1 = rho_psi(etime_rad, t_0, n, xi, m, l, h, b)
		#print "Transition from stiff to radiation dominated era"
		#n eqauals 2 at this time
		n = 2
		psi_rad_init = psi_1
		phi_rad_init = phi_1
		#print rho_phi_rad(etime_rad, n, xi, m, l, etime_rad, phi_rad_init)
		#print rho_psi_rad(etime_rad, n, xi, m, l, etime_rad, phi_rad_init, psi_rad_init)
		if plot:
			plt.figure("Stiff matter dominated era")
			stiff_era = np.linspace(t_0, etime_rad*1.5, resolution) # 100 linearly spaced numbers
			mat = np.array([rho_phi(z, t_0, 1, xi, m, l, h, b) for z in stiff_era])
			rad = np.array([rho_psi(z, t_0, 1, xi, m, l, h, b) for z in stiff_era])
			stiff = np.array([rho_stiff(z, G_N) for z in stiff_era])
			plt.ylim(0, max(mat[-1], rad[-1])*1.5)
			plt.plot(stiff_era, mat, 'b-')
			plt.plot(stiff_era, rad, 'y-')
			plt.plot(stiff_era, stiff, 'r-')


		#Radiation dominated era begins, n equals 2
		n = 2
		
		#Sove for the reheating temperature
		reh_time = r_time(etime_rad, n, xi, m, l, h, etime_rad, phi_rad_init, psi_rad_init)
		#print reh_time
		max_density = rho_psi_rad(reh_time, n, xi, m, l, h, etime_rad, phi_rad_init, psi_rad_init)
		#print max_density
		#Transfer reheating time to temperature
		#print "Reheating temperature: "
		#print math.pow(max_density,1.0/4.0)


		#print "RADIATION INTERSECTS STIFF: " + str(etime_rad)
		#print "REHEATING TIME: " + str(reh_time)
		#print "REHEATING TIME LARGER THAN FIRST INTERSECTION (NEEDS TO BE TRUE): {}".format(str(float(reh_time) > float(etime_rad)))

		if plot:
			plt.figure("Radiation dominated era")
			rad_era = np.linspace(etime_rad, reh_time*1.05, resolution) # 100 linearly spaced numbers
			mat = np.array([rho_phi_rad(z, n, xi, m, l, h, etime_rad, phi_rad_init) for z in rad_era])
			rad = np.array([rho_psi_rad(z, n, xi, m, l, h, etime_rad, phi_rad_init, psi_rad_init) for z in rad_era])
			d_rad = np.array([d_rho_psi_rad(z, n, xi, m, l, h, etime_rad, phi_rad_init, psi_rad_init) for z in rad_era])
			stiff = np.array([rho_stiff_rad_no_mat(z, etime_rad, G_N) for z in rad_era])
			plt.ylim(0, max(rad)*1.1)
			plt.plot(rad_era, mat, 'b-')
			plt.plot(rad_era, rad, 'y-')
			plt.plot(rad_era, d_rad, 'g-')
			plt.plot(rad_era, stiff, 'r-')
			#print(rad)
			#print(reh_time)
			#print(d_rho_psi_rad(etime_rad, n, xi, m, l, etime_rad, phi_rad_init, psi_rad_init))
			#print(d_rho_psi_rad(reh_time, n, xi, m, l, etime_rad, phi_rad_init, psi_rad_init))
			#print(d_rad)
			plt.show()
		return [max_density**(1/4) ,(max_density, reh_time), (psi_rad_init, etime_rad), False]

	else:
		#print "Transition from stiff to matter dominated era"
		#n equals 4 at this era
		#Calculate intersection of stiff and matter
		n = 1
		etime_mat = eq_time(t, t_0, n, xi, m, l, h, b, G_N)
	
		#print etime_mat
		if plot:
			plt.figure("Stiff matter dominated era")
			stiff_era = np.linspace(t_0, etime_mat, resolution) # 100 linearly spaced numbers
			mat = np.array([rho_phi(z, t_0, 1, xi, m, l, h, b) for z in stiff_era])
			rad = np.array([rho_psi(z, t_0, 1, xi, m, l, h, b) for z in stiff_era])
			stiff = np.array([rho_stiff(z, G_N) for z in stiff_era])
			plt.ylim(0, mat[-1]*1.1)
			plt.plot(stiff_era, mat, 'b-')
			plt.plot(stiff_era, rad, 'y-')
			plt.plot(stiff_era, stiff, 'r-')

		
		phi_1 =  rho_phi(etime_mat, t_0, n, xi, m, l, h, b)
		psi_1 = rho_psi(etime_mat, t_0, n, xi, m, l, h, b)
		stiff_1 = rho_stiff(etime_mat, G_N)
		#print etime_mat
		#print phi_1
		#print psi_1
		#print stiff_1

		#Matter dominated era begins, and n is now 4
		n = 4

		etime_rad = eq_tau(etime_mat, n, xi, m, l, h, etime_mat, psi_1, phi_1)
		#print etime_rad
		phi_2 = rho_phi_mat(etime_rad, n, xi, m, l, h, etime_mat, phi_1)
		psi_2 = rho_psi_mat(etime_rad, n, xi, m, l, h, etime_mat, psi_1, phi_1)
		#print phi_2
		#print psi_2
		if plot:
			plt.figure("Matter dominated era")
			mat_era = np.linspace(etime_mat, etime_rad*1, resolution) # 100 linearly spaced numbers
			mat = np.array([rho_phi_mat(z, n, xi, m, l, h, etime_mat, phi_1) for z in mat_era])
			rad = np.array([rho_psi_mat(z, n, xi, m, l, h, etime_mat, psi_1, phi_1) for z in mat_era])
			stiff = np.array([rho_stiff_mat(z, etime_mat, G_N) for z in mat_era])
			plt.ylim(0, max(phi_1, phi_2)*1.1)
			plt.plot(mat_era, mat, 'b-')
			plt.plot(mat_era, rad, 'y-')
			plt.plot(mat_era, stiff, 'r-')
		
		#Radiation dominated era begins, n equals 2
		n = 2
		
		
		#Sove for the reheating temperature
		reh_time = r_time(etime_rad, n, xi, m, l, h, etime_rad, phi_2, psi_2)
		#print(reh_time)
		#print reh_time
		max_density = rho_psi_rad(reh_time, n, xi, m, l, h, etime_rad, phi_2, psi_2)
		#print max_density
		#Transfer reheating time to temperature
		#print "Reheating temperature: "
		#print math.pow(max_density, 1.0/4.0)

		if plot:
			plt.figure("Radiation dominated era")
			rad_era = np.linspace(etime_rad, reh_time*1.05, resolution) # 100 linearly spaced numbers
			mat = np.array([rho_phi_rad(z, n, xi, m, l, h, etime_rad, phi_2) for z in rad_era])
			rad = np.array([rho_psi_rad(z, n, xi, m, l, h, etime_rad, phi_2, psi_2) for z in rad_era])
			d_rad = np.array([d_rho_psi_rad(z, n, xi, m, l, h, etime_rad, phi_2, psi_2) for z in rad_era])
			#print d_rad
			stiff = np.array([rho_stiff_rad(z, etime_mat, etime_rad, G_N) for z in rad_era])
			plt.ylim(0, max(rad)*1.1)
			plt.plot(rad_era, mat, 'b-')
			plt.plot(rad_era, rad, 'y-')
			plt.plot(rad_era, stiff, 'r-')
			plt.plot(rad_era, d_rad, 'g-')

			plt.show()
		#print max_density
		#print math.pow(max_density, 1.0/4.0)
		return [max_density**(1/4) ,(max_density, reh_time), (psi_2, etime_rad), (phi_1 ,etime_mat), True]

def reheating_time_star(args):
	return [args, reheating_time(*args)]


def generate_datapoints():
	data = []

	t_0 = 10**11
	G_N = 1.0
	plot = False

	max_mass = -7
	min_mass = -12
	mass_points = np.logspace(min_mass, max_mass, 100, endpoint=True, base=10)

	#lam = 10**min_mass*0.1

	min_b = -1
	max_b = 1
	b_points = np.logspace(min_b, max_b, 100, endpoint=True, base=10)

	minimal_xi = 0.0
	conformal_xi = 1.0/6
	other = 1.0/8

	h = 10**-12

	for m in mass_points:
		for b in b_points:
			for l in [10**-1, 10**-2, 10**-3]:
				for xi in [conformal_xi]:
					data.append((1.1*t_0, t_0, m, m*l, h, b, xi, G_N, plot))
	return data

np.seterr(all='raise')



if __name__ == "__main__":
	start = time.time()
	plot = False
	results = []
	p = Pool()
	data = generate_datapoints() 
	for i, x in enumerate(p.imap_unordered(reheating_time_star, data, 1)):
		results.append(x)
		sys.stderr.write('\rdone {0:%}'.format(float(i)/len(data)))
	p.terminate()
	with open('results.pkl', 'wb') as f:
		pickle.dump(results, f)
	sys.exit()

'''
data = generate_datapoints()
print(data[0])
s = []
for p in range(4, 16):
	try:
		print("Trying {}".format(p))
		h = 10**-p
		m = 10**-10
		t, t_0, m, l, h, b, xi, G_N, plot = (1.1*10**11, 10**11, m, (10**-3) * m, h, 10**0, 1/6, 1.0, True)
		print(reheating_time_star((t, t_0, m, l, h, b, xi, G_N, plot)))
		s.append(p)
	except:
		print("Failed {}".format(p))
print(s)





t_0 = 10**11
xi = 1/6
m=10**-17
l = m*10**-3
h = 10**-20
b=1
G_N=1
n = 1
plt.figure("Stiff matter dominated era")
stiff_era = np.linspace(t_0, t_0*2, resolution) # 100 linearly spaced numbers
mat = np.array([rho_phi(z, t_0, 1, xi, m, l, h, b) for z in stiff_era])
rad = np.array([rho_psi(z, t_0, 1, xi, m, l, h, b) for z in stiff_era])
stiff = np.array([rho_stiff(z, G_N) for z in stiff_era])
plt.ylim(0, max(mat[-1], rad[-1])*1.5)
plt.plot(stiff_era, mat, 'b-')
plt.plot(stiff_era, rad, 'y-')
plt.plot(stiff_era, stiff, 'r-')
plt.show()


etime_rad = eq_time(t_0*1.1, t_0, n, xi, m, l, h, b, G_N)
print(etime_rad)

phi_1 =  rho_phi(etime_rad, t_0, n, xi, m, l, h, b)
#print "phi_1: " + str(phi_1)
#Density of matter at etime_rad
psi_1 = rho_psi(etime_rad, t_0, n, xi, m, l, h, b)

print(phi_1 > psi_1)
plt.show()

print(Gamma_psi(10**18,10**-17, 1, 10**-20, 1))
print(Gamma_chi(10**18,10**-17, 1, 10**-18, 1))
'''
