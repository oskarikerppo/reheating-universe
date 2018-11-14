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

print len(rad_temps)
print max(rad_temps)
print min(rad_temps)
print len(mat_temps)
print max(mat_temps)
print min(mat_temps)

mass_points = list(set([x[1] for x in temps]))
b_points = list(set([x[3] for x in temps])) 

X, Y = np.meshgrid(mass_points, b_points)



print X.shape
print Y.shape


def create_Z(x, y, l, xi):
	Z = np.zeros((len(x), len(y)))
	data = [h for h in temps if h[4] == xi and h[2] == l*h[1]]
	for i in tqdm(range(len(x))):
		for j in tqdm(range(len(y))):
			for k in range(len(data)):
				if data[k][1] == x[i][j] and data[k][3] == y[i][j]:
					Z[i][j] = data[k][0]
					data.pop(k)
					break
	return Z


plt.xscale('log')
#plt.yscale('log')
plt.scatter(X,Y)
plt.show()


'''
Z = create_Z(X, Y, 10**-3, 0)

fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn,
               origin='lower', extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)])
plt.xscale('log')
plt.yscale('log')
plt.show()
'''