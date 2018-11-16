import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from tqdm import tqdm
import time

with open('results.pkl', 'rb') as f:
	results = pickle.load(f)

temps= []
for x in results:
	temps.append([x[1][0], x[0][2], x[0][3], x[0][4], x[0][5], x[1][-1]])

print len(temps)
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

X, Y = np.meshgrid(mass_points, b_points)



print X.shape
print Y.shape

def create_string_tick(x):
	x = str(x)
	tick = ""
	dot_reached = False
	num_of_decimals = 2
	current_decimals = 0
	e_reached = False
	for c in x:
		if not dot_reached and c != ".":
			tick += c
		elif c == ".":
			tick += "."
			dot_reached = True
		elif dot_reached and current_decimals < num_of_decimals:
			tick += c
			current_decimals += 1
		elif c == "e":
			tick += c
			e_reached = True
		elif e_reached:
			tick += c
	return tick



def create_Z(x, y, l, xi):
	Z = np.zeros((len(x), len(y)))
	M = np.zeros((len(x), len(y)))
	data = [h for h in temps if h[4] == xi and h[2] == l*h[1]]
	#print max(data)
	#print min(data)
	for i in tqdm(range(len(x))):
		for j in tqdm(range(len(y))):
			for k in range(len(data)):
				if data[k][1] == x[i][j] and data[k][3] == y[i][j]:
					Z[i][j] = data[k][0]
					if data[k][-1]:
						M[i][j] = 1
					else:
						M[i][j] = 0
					data.pop(k)
					break
	return Z, M


Z, M = create_Z(X, Y, 10**-3, 1.0/6)

'''
X = X-np.min(X)
X = X/np.max(X)
Y = Y-np.min(Y)
Y = Y/np.max(Y)
'''

#plt.scatter(X,Y)
#plt.show()





fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation='None', cmap='rainbow',
               origin='lower', aspect='auto', extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)]
               ,vmin=np.min(Z), vmax=np.max(Z))
ax.scatter(X, Y)
ticks=np.linspace(np.min(Z), np.max(Z), 6)
ticks_labels = [create_string_tick(x) for x in ticks]
cbar = fig.colorbar(im, ticks=ticks)
cbar.ax.set_yticklabels(ticks_labels)


fig2, ax2 = plt.subplots()
im2 = ax2.imshow(M, interpolation='None', cmap='rainbow',
               origin='lower', aspect='auto', extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)]
               ,vmin=0, vmax=1)
ax2.scatter(X, Y)
ticks2 = [0, 1]
ticks_labels2 = ["Radiation", "Matter"]
cbar2 = fig2.colorbar(im2, ticks=ticks2)
cbar2.ax.set_yticklabels(ticks_labels2)


ax.set_title("Temperature as a function of mass and b")
ax.set_xlabel("mass", fontsize=16)
ax.set_ylabel("b", fontsize=16, rotation='horizontal')
ax.set_xscale('log')
ax.set_yscale('log')
ax2.set_title("Matter and radiation dominated areas")
ax2.set_xlabel("mass", fontsize=16)
ax2.set_ylabel("b", fontsize=16, rotation='horizontal')
ax2.set_xscale('log')
ax2.set_yscale('log')






plt.show()
