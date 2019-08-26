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
    diagNorm = np.random.rand(4,4);
#    Q[i*n:(i+1)*n, i*n:(i+1)*n] = diagNorm + diagNorm.T;
    Q[i*n:(i+1)*n, i*n:(i+1)*n]  = diagNorm.T.dot(diagNorm)+ 7*np.eye(n); # 
#    w,v = np.linalg.eig(Q[i*n:(i+1)*n, i*n:(i+1)*n]);
#    Q[i*n:(i+1)*n, i*n:(i+1)*n] = Q[i*n:(i+1)*n, i*n:(i+1)*n]/2./(abs(w[0])+1e-4);
#    maxEig[i] = w[0]; minEig[i] = w[n-1];
    
    neighbours, fet = np.where(adjacency[:,i] == 1);
    for neigh in neighbours: 
        Q[i*n:(i+1)*n, neigh*n:(neigh+1)*n] = np.random.rand(4,4)*0.03 #np.multiply(,Qneighbour);
#        normSum[i] += np.linalg.norm(Q[i*n:(i+1)*n, neigh*n:(neigh+1)*n], 2);
#    for neigh in neighbours:
#        Q[i*n:(i+1)*n, neigh*n:(neigh+1)*n] =  Q[i*n:(i+1)*n, neigh*n:(neigh+1)*n]/100.;

Gamma = np.eye(N*n);
# step sizes----uniform Version
S = 0.5*(Q + Q.T);
eigS, eigVecS = np.linalg.eig(S.T.dot(S));
print (eigS);
alpha = np.min(eigS); 
eigQ, eigVecQ = np.linalg.eig(Q.T.dot(Q));
beta = np.max(eigQ);
gammai = np.sqrt(alpha)/beta;
Gamma = gammai*Gamma;
barA = np.eye(N*n)  - Gamma.dot(Q); 
w, v = np.linalg.eig(barA);
print (w)

# simulate the system
T = int(1e3);
x0 = np.random.rand((n*N));
timeLine = np.linspace(1,T ,T)
xTraj = ut.runGradient(barA, x0 = x0,T=int(T));
plt.figure();
plt.title("no noise, check if these step sizes converges")
plt.plot(timeLine, xTraj.T);
plt.grid();
plt.show();

#------------------ DISTURBANCE -------------------
E = np.zeros((n*N,N)); E[0] = 1.; E[1] = 1.;E[3] = 1.;E[3] = 1.;
C = np.zeros(n*N); C[n*N -1] = 1.;

#B = np.array([0,0,1,0]);
#
barE = np.zeros((n*N, N)); barC = np.zeros((N,n*N)); #barB = np.zeros((n*N,N))
#for i in range(N):
#    barE[n*i:(i+1)*n, i] = E;
#    barC[i, n*i:(i+1)*n] = C;
#    barB[n*i:(i+1)*n, i] =  B;
#invariantSub = isa.ISA(isa.ker(barC), barA, barB)
#
#if (isa.contained(barE, invariantSub)):
#    print ("disturbance decouplable");
#else:
#    print ("not disturbance decouplable");
#    
#    
## if decouplable we do the next step: 
#F = dd.ddController(invariantSub, barA, barB);
#print (F)
#print (barC.dot(barA+barB.dot(F)).dot(barE))

# -------------------simulate the system again-------------------
#T = int(1e3);
##x0 = np.random.rand((n*N));
#timeLine = np.linspace(1,T ,T)
#xTraj = ut.runGradient((0.99*barA + barB.dot(F)), x0, T= int(T));
#plt.figure();
#plt.title("simulated the disturbance decoupled closed loop without noise")
#plt.plot(timeLine, xTraj.T);
#plt.show();
#
#
## if decouplable we do the next step: 
#F = dd.ddController(invariantSub, barA, barB);
#print (F)
#print (barC.dot(barA+barB.dot(F)).dot(barE))

##---------- simulate noisy version -----------------------
#T = int(1e3);
#
#timeLine = np.linspace(1,T ,T);
#w = np.random.rand(9,T);
#
#xTraj = ut.runGradient((0.99*barA + barB.dot(F)), x0, barE.dot(w), int(T));
#plt.figure();
#plt.title("simulate the DD closed loop WIth Noise");
#plt.plot(timeLine, xTraj.T );
#plt.show();

##---------- simulate noisy version without feedback control -----------------------
T = int(1e3);

timeLine = np.linspace(1,T ,T);
w = 0.01*(np.random.rand(N,T)+0.5);

xTraj = ut.runGradient((barA), x0, noise = E.dot(w), T = int(T));
plt.figure();
plt.title("simulate the not DDed with NOISE")
plt.grid();
plt.plot(timeLine, xTraj.T );
plt.show();