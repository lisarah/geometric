#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:39:01 2019

@author: sarahli
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import util as ut
import isa as isa
import decouplingController as dd
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
normSum = np.zeros((N)); maxEig = np.zeros((N)); minEig = np.zeros((N))
for i in range(N):
    diagNorm = np.multiply(np.random.rand(4,4),Qii);
    Q[i*n:(i+1)*n, i*n:(i+1)*n] = diagNorm + diagNorm.T;
    
    w,v = np.linalg.eig(Q[i*n:(i+1)*n, i*n:(i+1)*n]);
    Q[i*n:(i+1)*n, i*n:(i+1)*n] = Q[i*n:(i+1)*n, i*n:(i+1)*n]/2./(abs(w[0])+1e-4);
    maxEig[i] = w[0]; minEig[i] = w[n-1];
    
    neighbours, fet = np.where(adjacency[:,i] == 1);
    for neigh in neighbours: 
        Q[i*n:(i+1)*n, neigh*n:(neigh+1)*n] = np.multiply(np.random.rand(4,4),Qneighbour);
        normSum[i] += np.linalg.norm(Q[i*n:(i+1)*n, neigh*n:(neigh+1)*n], 2);
    for neigh in neighbours:
        Q[i*n:(i+1)*n, neigh*n:(neigh+1)*n] =  Q[i*n:(i+1)*n, neigh*n:(neigh+1)*n]/normSum[i]/2.;

Gamma = np.eye(N*n);
# step sizes----uniform Version
S = 0.5*(Q + Q.T);
u, s, vt = np.linalg.svd(S);
ut.truncate(s);
alpha = np.min(s[np.nonzero(s)]); 
beta = np.linalg.norm(Q,2);
gammai = alpha/(beta*beta);
Gamma = gammai*Gamma;
barA = np.eye(N*n)  - Gamma.dot(Q); 
w, v = np.linalg.eig(barA);
print (w)

# simulate the system
T = int(1e3);
x0 = np.random.rand((n*N));
timeLine = np.linspace(1,T ,T)
xTraj = ut.runGradient(0.99*barA, x0,int(T));
plt.figure();
plt.plot(timeLine, xTraj.T);
plt.show();

#------------------ DISTURBANCE -------------------
E  = np.array([1,0,0,0]);
C = np.array([0,0,0,1]);
B = np.array([0,0,1,0]);

barE = np.zeros((n*N, N)); barC = np.zeros((N,n*N)); barB = np.zeros((n*N,N))
for i in range(N):
    barE[n*i:(i+1)*n, i] = E;
    barC[i, n*i:(i+1)*n] = C;
    barB[n*i:(i+1)*n, i] =  B;
invariantSub = isa.ISA(isa.ker(barC), barA, barB)

if (isa.contained(barE, invariantSub)):
    print ("disturbance decouplable");
else:
    print ("not disturbance decouplable");
    
    
# if decouplable we do the next step: 
F = dd.ddController(invariantSub, barA, barB);
print (F)
print (barC.dot(barA+barB.dot(F)).dot(barE))