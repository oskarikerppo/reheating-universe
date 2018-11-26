import pickle as pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from tqdm import tqdm
import time

with open('results.pkl', 'rb') as f:
	results = pickle.load(f)

temps= []
for x in results:
	temps.append([x[1][0], x[0][2], x[0][3], x[0][4], x[0][5], x[1][-1], x[1][1][1], x[1][2][1], x[1][3][1]])

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

X, Y = np.meshgrid(mass_points, b_points)




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
	T = np.zeros((len(x), len(y)))
	T2 = np.zeros((len(x), len(y)))
	T3 = np.zeros((len(x), len(y)))
	M = np.zeros((len(x), len(y)))
	data = [h for h in temps if h[4] == xi and h[2] == l*h[1]]
	#print max(data)
	#print min(data)
	for i in tqdm(range(len(x))):
		for j in tqdm(range(len(y))):
			for k in range(len(data)):
				if data[k][1] == x[i][j] and data[k][3] == y[i][j]:
					Z[i][j] = data[k][0]
					if data[k][5]:
						M[i][j] = 1
					else:
						M[i][j] = 0
					T[i][j] = data[k][-3]
					T2[i][j] = data[k][-2]
					T3[i][j] = data[k][-1]
					data.pop(k)
					break
	return Z, M, T, T2, T3


Z, M, T, T2, T3 = create_Z(X, Y, 10**-3, 1.0/6)

'''
X = X-np.min(X)
X = X/np.max(X)
Y = Y-np.min(Y)
Y = Y/np.max(Y)
'''

#plt.scatter(X,Y)
#plt.show()





fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation='bilinear', cmap='rainbow',
               origin='lower', aspect='auto', extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)]
               ,vmin=np.min(Z), vmax=np.max(Z))
#ax.scatter(X, Y)
ticks=np.linspace(np.min(Z), np.max(Z), 6)
ticks_labels = [create_string_tick(x) for x in ticks]
cbar = fig.colorbar(im, ticks=ticks)
cbar.ax.set_yticklabels(ticks_labels)


fig2, ax2 = plt.subplots()
im2 = ax2.imshow(M, interpolation='bilinear', cmap='rainbow',
               origin='lower', aspect='auto', extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)]
               ,vmin=0, vmax=1)
#ax2.scatter(X, Y)
ticks2 = [0, 1]
ticks_labels2 = ["Radiation", "Matter"]
cbar2 = fig2.colorbar(im2, ticks=ticks2)
cbar2.ax.set_yticklabels(ticks_labels2)


fig3, ax3 = plt.subplots()
im3 = ax3.imshow(T, interpolation='bilinear', cmap='rainbow',
               origin='lower', aspect='auto', extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)]
               ,vmin=np.min(T), vmax=np.max(T))
#ax.scatter(X, Y)
ticks3=np.linspace(np.min(T), np.max(T), 6)
ticks_labels3 = [create_string_tick(x) for x in ticks3]
cbar3 = fig3.colorbar(im3, ticks=ticks3)
cbar3.ax.set_yticklabels(ticks_labels3)

fig4, ax4 = plt.subplots()
im4 = ax4.imshow(T2, interpolation='bilinear', cmap='rainbow',
               origin='lower', aspect='auto', extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)]
               ,vmin=np.min(T2), vmax=np.max(T2))
#ax.scatter(X, Y)
ticks4=np.linspace(np.min(T2), np.max(T2), 6)
ticks_labels4 = [create_string_tick(x) for x in ticks4]
cbar4 = fig4.colorbar(im4, ticks=ticks4)
cbar4.ax.set_yticklabels(ticks_labels4)

fig5, ax5 = plt.subplots()
im5 = ax5.imshow(T3, interpolation='bilinear', cmap='rainbow',
               origin='lower', aspect='auto', extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)]
               ,vmin=np.min(T3), vmax=np.max(T3))
#ax.scatter(X, Y)
ticks5=np.linspace(np.min(T3), np.max(T3), 6)
ticks_labels5 = [create_string_tick(x) for x in ticks5]
cbar5 = fig5.colorbar(im5, ticks=ticks5)
cbar5.ax.set_yticklabels(ticks_labels5)


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
ax3.set_title("Age of universe at reheating time")
ax3.set_xlabel("mass", fontsize=16)
ax3.set_ylabel("b", fontsize=16, rotation='horizontal')
ax3.set_xscale('log')
ax3.set_yscale('log')

ax4.set_title("Time of transition to radiation dominance")
ax4.set_xlabel("mass", fontsize=16)
ax4.set_ylabel("b", fontsize=16, rotation='horizontal')
ax4.set_xscale('log')
ax4.set_yscale('log')

ax5.set_title("Time of transition to matter dominance")
ax5.set_xlabel("mass", fontsize=16)
ax5.set_ylabel("b", fontsize=16, rotation='horizontal')
ax5.set_xscale('log')
ax5.set_yscale('log')





plt.show()
