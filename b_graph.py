import pickle as pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from tqdm import tqdm
import time

matplotlib.rc('text', usetex = True)

with open('results.pkl', 'rb') as f:
	results = pickle.load(f)


temps= []
for x in results:
	try:
		temps.append([x[1][0], x[0][2], x[0][3], x[0][4], x[0][5], x[1][-1], x[1][1][1], x[1][2][1], x[1][3][1]])
	except:
		temps.append([x[1][0], x[0][2], x[0][3], x[0][4], x[0][5], x[1][-1], x[1][1][1], x[1][2][1], np.nan])

print(len(temps))
#print(temps)
mat_temps = []
rad_temps = []

for x in temps:
	if x[-1]:
		mat_temps.append(x)
	else:
		rad_temps.append(x)
'''
print len(rad_temps)
print max(rad_temps)
print min(rad_temps)
print len(mat_temps)
print max(mat_temps)
print min(mat_temps)
'''
mass_points = list(set([x[1] for x in temps]))
b_points = list(set([x[3] for x in temps])) 

mass_points = sorted(mass_points)
b_points = sorted(b_points)

print(len(b_points))
b_p = b_points[-1]

def create_Z(mass_points, l, xi, b):
	data = [h for h in temps if h[4] == xi and h[2] == l*h[1] and h[3] == b]
	points = [(data[k][0], data[k][1]) for k in range(len(data))] # (temp, mass)
	points = sorted(points, key=lambda x: x[1])
	return points

#print(temps[0])

points = create_Z(mass_points, 10**-1, 1/8, b_p)
print(points)
print (b_p)
plt.plot([p[1] for p in points], [p[0] for p in points], "b-")

points = create_Z(mass_points, 10**-2, 1/8, b_p)
print(points)
print (b_p)
plt.plot([p[1] for p in points], [p[0] for p in points], "g--")

points = create_Z(mass_points, 10**-3, 1/8, b_p)
print(points)
print (b_p)
plt.plot([p[1] for p in points], [p[0] for p in points], "r:")

        
plt.xlabel('$m$', fontsize=16)
plt.ylabel('$T_{rh}$', rotation='horizontal', fontsize=16)
plt.xscale("log")
plt.yscale("log")
plt.title("")
plt.show()
