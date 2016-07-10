import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import csv
import json

data = json.load(open('pruebaV2_1', 'rb'))
# print type(data)
# dict = {}
#
#
# for key, val in csv.reader(open("output.csv")):
#     dict[key] = val

centers = [[1, 1], [-1, -1], [1, -1]]
X = np.array(data['data'])
attributes = np.array(data['attributes'])
X1 = sp.delete(X, 0, 1)
X1 = sp.delete(X1, 0, 1)
X1 = sp.delete(X1, 0, 1)
est = KMeans(n_clusters=2)
fignum = 1

fig = plt.figure(fignum, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
est.fit(X1)
print est.predict(X1[1,:])
labels = est.labels_
print labels.astype(np.float)
cmap = []
for items in labels:
    if items == 0: cmap.append('r')
    if items == 1: cmap.append('b')
    if items == 2: cmap.append('g')

ax.scatter(X[:,0], X[:,1], X[:,4], s = 70, c=cmap, norm = 1)
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel(attributes[1])
ax.set_ylabel(attributes[2])
ax.set_zlabel(attributes[5])
# plt.colorbar()
plt.show()