#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:39:01 2019

@author: sarahli
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

plt.close('all');
n = 4;
Qii = np.array([[0, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 0]]);
Qneighbour = np.zeros((4,4)); Qneighbour[0:2, 0:2] = np.eye(2);

N = 3;
smallWorld = nx.navigable_small_world_graph(N);
circleSmallWorld = nx.circular_layout(smallWorld);
plt.figure();
nx.draw(smallWorld, circleSmallWorld, node_color=range(N*N), node_size=800, cmap=plt.cm.Blues)
nx.draw_networkx_labels(smallWorld,circleSmallWorld,font_size=20,font_family='sans-serif')
plt.show();

adjacency = nx.to_numpy_matrix(smallWorld); # column of adjacency gives who column i receives

N,N = adjacency.shape;

Q = np.zeros((N*n, N*n));
normSum = np.zeros((N));
for i in range(N):
    Q[i*n:(i+1)*n,i*n:(i+1)*n] = np.multiply(np.random.rand(4,4),Qii);
    neighbours, fet = np.where(adjacency[:,i] == 1);
    for neigh in neighbours: 
        Q[i*n:(i+1)*n, neigh*n:(neigh+1)*n] = np.multiply(np.random.rand(4,4),Qneighbour);
        normSum[i] += np.linalg.norm(Q[i*n:(i+1)*n, neigh*n:(neigh+1)*n] , 2);
    if np.linalg.norm(Q[i*n:(i+1)*n,i*n:(i+1)*n]) < normSum[i]:
        Q[i*n:(i+1)*n,i*n:(i+1)*n] = 1.1*normSum[i] / np.linalg.norm(Q[i*n:(i+1)*n,i*n:(i+1)*n], 2)*Q[i*n:(i+1)*n,i*n:(i+1)*n];


Gamma = np.eye(N*n);
for i in range(N):
    gammai =2/(normSum[i] +np.linalg.norm(Q[i*n:(i+1)*n,i*n:(i+1)*n])) ;
    Gamma[i*n: (i+1)*n]  = gammai*Gamma[i*n: (i+1)*n]; 
barA = np.eye(N*n)  - Gamma.dot(Q); 
w, v = np.linalg.eig(barA);
print (w)
print (np.linalg.norm(barA,2))