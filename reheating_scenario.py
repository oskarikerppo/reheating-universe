#!/usr/bin/python
# -*- coding: latin-1 -*-


import math
import numpy as np
from scipy import special
import scipy.integrate as integrate
from scipy.optimize import fsolve, minimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt


#Initialize all constant values for simulation
t_0 = 1.5e+6
t, m, l, b, xi = (t_0*1.1, 10.0, 0.1, 1.0, 0.0)
#G_N = 6.7071186*math.pow(10.0, -39.0)
#t_0 = 1.52*math.pow(10.0, -8.0)
G_N = 1.0
#t_0 = math.pow(10.0,-32.0)



#creation rate for massive particles
def Gamma_phi(m,b,t):
	airy_param = -math.pow(3.0*m*t/2.0, 2.0/3.0)
	ai, aip, bi, bip = special.airy(airy_param)	
	g_p = 3*math.pow(m*b, 16.0/3.0)*t*(math.pow(ai, 2) + math.pow(bi, 2))/(32.0*b)
	return g_p



#print Gamma_phi(m,b,time_in_gev(1))


#Differential decay rate for massless particles
def Gamma_psi(m, l, t, a):
	g_p = math.pow(l, 2.0)*t*special.hankel1(a, m*t)*special.hankel2(a, m*t)/(32.0*m)
	#return real part for now
	return g_p.real



#Index for Hankel functions
def alpha(n, xi):
	a = math.sqrt(1.0 - n*(n - 2.0)*(6.0*xi - 1.0))/(2.0 + n)
	#Return only real part for now
	return a.real 

#print Gamma_psi(10.0,10,10, alpha(4,0.0))

def f(t, t_0, n, m, l, xi):
	a = alpha(n, xi)
	r_p = integrate.quad(lambda x: Gamma_psi(m, l, x, a), t_0, t)
	return r_p[0]


def scale_factor(t):
	return math.pow(t, 1.0/3.0)

#print f(1.0,0.001, 4, 0.01,0.01)

#Energy density for massive particles
def rho_phi(t, t_0, n, m, l, b, xi):
	r_p = math.pow(scale_factor(t), -3.0)*math.exp(-f(t, t_0, n, m, l, xi))*integrate.quad(lambda t: math.pow(scale_factor(t), 3.0)*Gamma_phi(m, b, t)*math.exp(f(t, t_0, n, m, l, xi)), t_0, t)[0]
	return r_p



#Energy density for massless particles
def rho_psi(t, t_0, n, m, l, b, xi):
	a = alpha(n, xi)
	r_p = math.pow(scale_factor(t), -4.0)*integrate.quad(lambda t: Gamma_psi(m, l, t, a)*rho_phi(t, t_0, n, m, l, b, xi)*math.pow(scale_factor(t),4), t_0, t)[0]
	return r_p


#Energy density of stiff matter
def rho_stiff(t, G_N):
	return 1.0 / (24.0*math.pi*G_N*math.pow(t, 2.0))
#print "G_N: " + str(G_N)
#print rho_stiff(0.1, G_N)


################################################# AGORITHM ################################################# 


################################################# UINVERSE ENDS UP IN MATTER DOMINATED ERA ################################################# 

def matter_minus_stiff(t, t_0, n, m, l, b, xi, G_N):
	return rho_phi(t, t_0, n, m, l, b, xi) - rho_stiff(t, G_N)

def rad_minus_stiff(t, t_0, n, m, l, b, xi, G_N):
	return rho_psi(t, t_0, n, m, l, b, xi) - rho_stiff(t, G_N)



def eq_time(t, t_0, n, m, l, b, xi, G_N):
	data = (t_0, n, m, l, b, xi, G_N)
	t_eq = fsolve(matter_minus_stiff, t, args=data)
	r_p = rho_phi(t_eq, t_0, n, m, l, b, xi)
	r_s = rho_stiff(t_eq, G_N)
	print("Absolute error: ") + str(r_p - r_s)
	return t_eq[0]

def eq_time2(t, t_0, n, m, l, b, xi, G_N):
	data = (t_0, n, m, l, b, xi, G_N)
	t_eq = fsolve(rad_minus_stiff, t, args=data)
	r_p = rho_psi(t_eq, t_0, n, m, l, b, xi)
	r_s = rho_stiff(t_eq, G_N)
	print("Absolute error: ") + str(r_p - r_s)
	return t_eq[0]


#e_time = eq_time(t, t_0, 1, m, l, b, xi, G_N)
#e_time2 = eq_time2(t, t_0, 1, m, l, b, xi, G_N)
#print e_time
#print e_time2


'''
D = np.linspace(t_0, 0.1, 25, endpoint=True)
stiff_data = []
matter_data = []
rad_data = []

for i in D:
	stiff_data.append(rho_stiff(i,G_N))
	matter_data.append(rho_phi(i, t_0, 1, m, l, b, xi))
	rad_data.append(rho_psi(i, t_0, 1, m, l, b, xi))

print matter_data
print rad_data
print stiff_data
plt.ylim(0, matter_data[-1])
plt.plot(D, matter_data)
plt.plot(D, rad_data)
plt.plot(D, stiff_data)
plt.show()
'''



#eqtime2 =  eq_time2(100, t_0, 2, m, l, b, xi, G_N)

#The constant n: 1 for stiff matter, 2 for radiation, 4 for matter !!!!




############### MATTER DOMINATED PHASE ##################
#Energy density of matter
def rho_phi_mat(t, n, m, l, xi, eq_time, rho_eq):
	r_p_m = math.pow(math.pow(eq_time/t, 2.0/3.0), 3)*(math.exp(-f(t, eq_time, n, m, l, xi)) * rho_eq)
	return r_p_m



#Energy density of radiation
def rho_psi_mat(t, n, m, l, b, xi, eq_time, rho_psi_eq, rho_eq):
	a = alpha(n, xi)
	r_p_m = math.pow(1.0 / math.pow(t, 2.0/3.0), 4)*integrate.quad(lambda t: Gamma_psi(m, l, t, a)*rho_phi_mat(t, n, m, l, xi, eq_time, rho_eq)*math.pow(math.pow(t, 2.0/3.0), 4), eq_time, t)[0] + rho_psi_eq*math.pow(math.pow(eq_time/t, 2.0/3.0), 4.0)
	return r_p_m

#print rho_psi_mat(t, t_0, n, m, l, b, xi, eq_time, rho_psi_eq, rho_eq)



def matter_minus_rad(t, n, m, l, b, xi, eq_time, rho_psi_eq, rho_eq):
	return rho_phi_mat(t, n, m, l, xi, eq_time, rho_eq) - rho_psi_mat(t, n, m, l, b, xi, eq_time, rho_psi_eq, rho_eq)


#Solve when radiation dominated era begins
def eq_tau(t, n, m, l, b, xi, eq_time, rho_psi_eq, rho_eq):
	data = (n, m, l, b, xi, eq_time, rho_psi_eq, rho_eq)
	t_eq = fsolve(matter_minus_rad, t, args=data)
	r_p = rho_phi_mat(t_eq, n, m, l, xi, eq_time, rho_eq)
	r_r = rho_psi_mat(t_eq, n, m, l, b, xi, eq_time, rho_psi_eq, rho_eq)
	print("Absolute error: ") + str(r_p - r_r)
	return t_eq[0]



###################### Radiation dominated era #####################

#Energy density of radiation dominated era


def rho_phi_rad(t, n, m, l, xi, eq_tau, rho_tau):
	r_p_r = math.pow(math.pow(eq_tau/t, 1.0/2.0), 3)*(math.exp(-f(t, eq_tau, n, m, l, xi)) * rho_tau)
	return r_p_r

#print rho_phi_rad(1500, 0.001, 2, 3, 1.1, 0.01)



def rho_psi_rad(t, n, m, l, xi, eq_tau, rho_tau, psi_tau):
	a = alpha(n, xi)
	r_p_r = math.pow(math.pow(1.0/t, 1.0/2.0), 4)*integrate.quad(lambda t: Gamma_psi(m, l, t, a)*rho_phi_rad(t, n, m, l, xi, eq_tau, rho_tau)*math.pow(math.pow(t, 1.0/2.0), 4.0), eq_tau, t)[0] + psi_tau*math.pow(math.pow(eq_tau/t, 1.0/2.0), 4.0)
	return r_p_r


def neg_rho_psi_rad(t, n, m, l, xi, eq_tau, rho_tau, psi_tau):
	return -1*rho_psi_rad(t, n, m, l, xi, eq_tau, rho_tau, psi_tau)


def reheating_temperature(t, n, m, l, xi, eq_tau, rho_tau, psi_tau):
	data = (n, m, l, xi, eq_tau, rho_tau, psi_tau)
	rh_time = minimize(neg_rho_psi_rad, t, args=data)
	print rh_time
	return rh_time
#print rho_psi_rad(1000, 0.001, 2, 3, 1.1, 0.01)


#print  eq_time(t, t_0, 2, m, l, b, xi, G_N)


######################## MAIN #####################

def reheating_time(t, t_0, m, l, b, xi, G_N):
	#First calculate time when density of radiation equals that of stiff matter
	#n equals 1 at this time
	n = 1
	etime_rad = eq_time2(t, t_0, n, m, l, b, xi, G_N)
	print etime_rad
	#Density of radiation at etime_rad
	phi_1 =  rho_phi(etime_rad, t_0, n, m, l, b, xi)
	print "phi_1: " + str(phi_1)
	#Density of matter at etime_rad
	psi_1 = rho_psi(etime_rad, t_0, n, m, l, b, xi)
	#If density of radiation is greater, we go straight to radiation dominated era
	print "psi_1: " + str(psi_1)
	if psi_1 > phi_1:
		print "Transition from stiff to radiation dominated era"
		#n eqauals 2 at this time
		n = 2
		psi_rad_init = psi_1
		phi_rad_init = psi_1
		rho_phi_rad(t, n, m, l, xi, etime_rad, phi_rad_init)
		rho_psi_rad(t, n, m, l, xi, etime_rad, phi_rad_init, psi_rad_init)
	else:
		print "Transition from stiff to matter dominated era"
		#n equals 4 at this era
		#Calculate intersection of stiff and matter
		n = 1
		etime_mat = eq_time(t_0, t_0, n, m, l, b, xi, G_N)
	
		stiff_data = []
		rad_data = []
		mat_data = []
		D = []
		
		
		stiff_range = np.linspace(t_0, etime_rad, 50, endpoint=False)
		D = stiff_range
		for i in stiff_range:
			stiff_data.append(rho_stiff(i, G_N))
			rad_data.append(rho_psi(i, t_0, n, m, l, b, xi))
			mat_data.append(rho_phi(i, t_0, n, m, l, b, xi))
		
		
		phi_1 =  rho_phi(etime_mat, t_0, n, m, l, b, xi)
		psi_1 = rho_psi(etime_mat, t_0, n, m, l, b, xi)
		stiff_1 = rho_stiff(etime_mat, G_N)
		print phi_1
		print psi_1
		print stiff_1

		n = 4
		
		etime_rad = eq_tau(etime_mat, n, m, l, b, xi, etime_mat, psi_1, phi_1)
		print etime_rad
		phi_2 = rho_phi_mat(etime_rad, n, m, l, xi, etime_mat, phi_1)
		psi_2 = rho_psi_mat(etime_rad, n, m, l, b, xi, etime_mat, psi_1, phi_1)
		print phi_2
		print psi_2
		'''
		mat_range = np.linspace(etime_mat, etime_rad, 50, endpoint=False)
		D = np.append(D, mat_range)
		for i in mat_range:
			stiff_data.append(rho_stiff(i, G_N))
			rad_data.append(rho_psi_mat(i, n, m, l, b, xi, etime_mat, psi_1, phi_1))
			mat_data.append(rho_phi_mat(i, n, m, l, xi, etime_mat, phi_1))
		'''
		
		
		#Radiation dominated era begins
		n = 2
		
		
		#Sove for the reheating temperature
		reh_time = reheating_temperature(etime_rad*1.1, n, m, l, xi, etime_rad, phi_2, psi_2)
		print -1*reh_time['fun']
		print rho_psi_rad(reh_time['x'], n, m, l, xi, etime_rad, phi_2, psi_2)
		#Transfer reheating time to temperature
		print "Reheating temperature: "
		print math.pow(rho_psi_rad(reh_time['x'], n, m, l, xi, etime_rad, phi_2, psi_2),1.0/4.0)
		'''
		rad_range = np.linspace(etime_rad, etime_rad*3, 50, endpoint=False)
		D = np.append(D, rad_range)
		for i in rad_range:
			stiff_data.append(rho_stiff(i, G_N))
			rad_data.append(rho_psi_rad(i, n, m, l, xi, etime_rad, phi_2, psi_2))
			mat_data.append(rho_phi_rad(i, n, m, l, xi, etime_rad, phi_2))
		'''
		plt.ylim(0, mat_data[-1]*1.2)
		plt.plot(D, stiff_data, 'r-')
		
		plt.plot(D, mat_data, 'b-')
		plt.plot(D, rad_data, 'y-')
		plt.show()

reheating_time(t, t_0, m, l, b, xi, G_N)
'''
eq_time, eq_tau, reheating_t, rho_eq, rho_psi_eq, rho_tau, psi_tau, t_0, m, l, b, xi = reheating_time(t, t_0, m, l, b, xi, G_N)


print("-------------------- PLOTS -------------------")


print("Equal time: " + str(eq_time))
print("Equal tau: " + str(eq_tau))
print("Solution for reheating: ")
print(reheating_t)
'''
'''
######################## PLOTS ######################

print "eq_time: " + str(eq_time)
print rho_stiff(eq_time, G_N)
print rho_phi(eq_time, t_0, 1, m, l, b, xi)
'''
'''
X = np.linspace(t_0, eq_time, 200, endpoint=False)
Y = np.linspace(eq_time, eq_tau, 100, endpoint=False)
Z = np.linspace(eq_tau, eq_tau + 1, 50, endpoint=False)

domain = np.append(X,Y)
domain = np.append(domain, Z)

stiff_points = []
mat_points = []
rad_points = []

for i in domain:
		stiff_points.append(rho_stiff(i, G_N))
		mat_points.append(rho_phi_mat(i, t_0, 4, m, l, xi, eq_time, rho_eq))
		rad_points.append(rho_phi_rad(i, t_0, 2, m, l, xi, eq_tau, rho_tau))

#plt.plot(domain, stiff_points)
plt.plot(domain, mat_points)
plt.plot(domain, rad_points)

#plt.ylim((0,20))
plt.show()

'''


'''


p_f = []
p_s = []
p_p = []
x = []


for i in np.arange(t_0,eq_time,0.02):
	print i
	x.append(i)
	p_f.append(rho_phi(i, t_0, 1, m, l, b, xi))
	p_s.append(rho_stiff(i, 1.0))
	p_p.append(rho_psi(i, t_0, 1, m, l, b, xi))


for i in np.arange(eq_time, eq_tau, 1):
	print i
	x.append(i)
	p_f.append(rho_phi_mat(i, t_0, 4, m, l, xi, eq_time, rho_eq))
	p_s.append(rho_stiff(i, 1.0))
	p_p.append(rho_psi_mat(i, t_0, 4, m, l, b, xi, eq_time, rho_psi_eq, rho_eq))	

for i in np.arange(eq_tau,eq_tau + 100, 1):
	print i
	x.append(i)
	p_f.append(rho_phi_rad(i, t_0, 2, m, l, xi, eq_tau, rho_tau))
	p_s.append(rho_stiff(i, 1.0))
	p_p.append(rho_psi_rad(i, t_0, 2, m, l, xi, eq_tau, rho_tau, psi_tau))	

plt.ylim((0,2))

plt.plot(x, p_f)
plt.plot(x, p_s)
plt.plot(x, p_p)
plt.show()
'''