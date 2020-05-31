import pickle as pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import numpy as np
from tqdm import tqdm
import time


matplotlib.rc('text', usetex = True)

with open('results.pkl', 'rb') as f:
	results = pickle.load(f)

temps= []
for x in results:
  print(x)
  try:
    temps.append([x[1][0], x[0][2], x[0][3], x[0][5], x[0][6], x[1][-1], x[1][1][1], x[1][2][1], x[1][3][1]])
  except:
    temps.append([x[1][0], x[0][2], x[0][3], x[0][5], x[0][6], x[1][-1], x[1][1][1], x[1][2][1], np.nan])

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


X, Y = np.meshgrid(mass_points, b_points)
#Y = np.transpose(Y)




def create_string_tick(x):
        x = "{:.2e}".format(x)
        #x = str(x)
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
	#print(max(data))
	#print(min(data))
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

print(temps[0])
Z, M, T, T2, T3 = create_Z(X, Y, 10**-3, 1/6)

print(np.max(Z))
print(np.min(Z))


'''
X = X-np.min(X)
X = X/np.max(X)
Y = Y-np.min(Y)
Y = Y/np.max(Y)
'''

#plt.scatter(X,Y)
#plt.show()





fig, ax = plt.subplots()
im = ax.pcolor(X, Y, Z, cmap='plasma', norm=colors.LogNorm(vmin=np.min(Z), vmax=np.max(Z)))
#ax.scatter(X, Y)
ticks=np.logspace(np.log10(np.min(Z)), np.log10(np.max(Z)), 6, endpoint=True, base=10)
ticks_labels = [create_string_tick(x) for x in ticks]
cbar = fig.colorbar(im, ticks=ticks)
cbar.ax.set_yticklabels(ticks_labels)


fig2, ax2 = plt.subplots()
im2 = ax2.pcolor(X, Y, M, cmap='plasma',vmin=0, vmax=1)
#ax2.scatter(X, Y)
ticks2 = [0, 1]
ticks_labels2 = ["Radiation", "Matter"]
cbar2 = fig2.colorbar(im2, ticks=ticks2)
cbar2.ax.set_yticklabels(ticks_labels2)


fig3, ax3 = plt.subplots()
im3 = ax3.pcolor(X, Y, T, cmap='plasma', norm=colors.LogNorm(vmin=np.min(T), vmax=np.max(T)))
#ax.scatter(X, Y)
ticks3=np.logspace(np.log10(np.min(T)), np.log10(np.max(T)), 6, endpoint=True, base=10)
ticks_labels3 = [create_string_tick(x) for x in ticks3]
cbar3 = fig3.colorbar(im3, ticks=ticks3)
cbar3.ax.set_yticklabels(ticks_labels3)

fig4, ax4 = plt.subplots()
im4 = ax4.pcolor(X, Y, T2, cmap='plasma', norm=colors.LogNorm(vmin=np.min(T2), vmax=np.max(T2)))
#ax.scatter(X, Y)
ticks4=np.logspace(np.log10(np.min(T2)), np.log10(np.max(T2)), 6, endpoint=True, base=10)
ticks_labels4 = [create_string_tick(x) for x in ticks4]
cbar4 = fig4.colorbar(im4, ticks=ticks4)
cbar4.ax.set_yticklabels(ticks_labels4)
print("NOW MIN AND MAX")
print(np.nanmin(T3))
print(np.nanmax(T3))

c_cmap =matplotlib.cm.jet
#c_cmap.set_bad('black')
#m_array = np.ma.array(T3, mask=np.isnan(T3))
fig5, ax5 = plt.subplots()
im5 = ax5.pcolor(X, Y, T3, cmap='plasma', norm=colors.LogNorm(vmin=np.nanmin(T3), vmax=np.nanmax(T3)))
#im5 = ax5.scatter(X, Y, c=T3, s=T3 ,cmap=c_cmap, vmin=np.nanmin(T3), vmax=np.nanmax(T3))
#ax5.scatter(X, Y, c=T3, vmin=np.nanmin(T3), vmax=np.nanmax(T3))
ticks5=np.logspace(np.log10(np.nanmin(T3)), np.log10(np.nanmax(T3)), 6, endpoint=True, base=10)
ticks_labels5 = [create_string_tick(x) for x in ticks5]
cbar5 = fig5.colorbar(im5, ticks=ticks5)
cbar5.ax.set_yticklabels(ticks_labels5)

#print(m_array)

#Temperature as a function of mass and b
ax.set_title("")
ax.set_xlabel('$m$', fontsize=16)
ax.set_ylabel('$b$', fontsize=16, rotation='horizontal')
ax.set_xscale('log')
ax.set_yscale('log')

ax2.set_title("")
ax2.set_xlabel('$m$', fontsize=16)
ax2.set_ylabel('$b$', fontsize=16, rotation='horizontal')
ax2.set_xscale('log')
ax2.set_yscale('log')

#Age of universe at reheating time
ax3.set_title("")
ax3.set_xlabel('$m$', fontsize=16)
ax3.set_ylabel('$b$', fontsize=16, rotation='horizontal')
ax3.set_xscale('log')
ax3.set_yscale('log')

ax4.set_title("")
ax4.set_xlabel('$m$', fontsize=16)
ax4.set_ylabel('$b$', fontsize=16, rotation='horizontal')
ax4.set_xscale('log')
ax4.set_yscale('log')

#Time of transition to matter dominance
ax5.set_title("")
ax5.set_xlabel('$m$', fontsize=16)
ax5.set_ylabel('$b$', fontsize=16, rotation='horizontal')
ax5.set_xscale('log')
ax5.set_yscale('log')





plt.show()
