# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:10:35 2019

@author: craba
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
Qneighbour = np.zeros((n,n)); Qneighbour[0:2, 0:2] = np.eye(2);

N = 2;
smallWorld = nx.navigable_small_world_graph(N);
circleSmallWorld = nx.circular_layout(smallWorld);
plt.figure();
nx.draw(smallWorld, circleSmallWorld, node_color=range(N*N), node_size=800, cmap=plt.cm.Blues)
nx.draw_networkx_labels(smallWorld,circleSmallWorld,font_size=20,font_family='sans-serif')
plt.show();

adjacency = nx.to_numpy_matrix(smallWorld); # column of adjacency gives who column i receives

N,N = adjacency.shape;

Q = np.zeros((N*n, N*n));
normSum = np.zeros((N)); maxEig = np.zeros((N)); minEig = np.zeros((N));

Gamma = np.eye(N*n);

for i in range(N):
    print ("in agent ", i);
    randMat = np.random.rand(4,4);

    Q[i*n:(i+1)*n, i*n:(i+1)*n] = np.multiply(Qii + Qii.T, np.multiply(randMat,randMat));
    
    w,v = np.linalg.eig(Q[i*n:(i+1)*n, i*n:(i+1)*n]);
    
    maxEig[i] = max(abs(w)); minEig[i] = abs(w[n-1]);
    
    neighbours, fet = np.where(adjacency[:,i] == 1);
    for neigh in neighbours: 
        Q[i*n:(i+1)*n, neigh*n:(neigh+1)*n] = np.multiply(np.random.rand(4,4),Qneighbour);
        print (neigh)
        normSum[i] += np.linalg.norm(Q[i*n:(i+1)*n, neigh*n:(neigh+1)*n], 2);
#    for neigh in neighbours:
#        Q[i*n:(i+1)*n, neigh*n:(neigh+1)*n] =  Q[i*n:(i+1)*n, neigh*n:(neigh+1)*n];

    Gamma[i,i] = min( 0.3/normSum[i], 1.7/(np.linalg.norm(Q[i*n:(i+1)*n, i*n:(i+1)*n],2)));
    
    
# step sizes---- gershgorin circle

#S = 0.5*(Q + Q.T);
#u, s, vt = np.linalg.svd(S);
#ut.truncate(s);
#alpha = np.min(s[np.nonzero(s)]); 
#beta = np.linalg.norm(Q,2);
#gammai = alpha/(beta*beta);
#Gamma = gammai*Gamma;
    
barA = np.eye(N*n)  - Gamma.dot(Q); 
w, v = np.linalg.eig( -Gamma.dot(Q));
print (abs(w))

w, v = np.linalg.eig(barA);
print (abs(w))

# simulate the system
#T = int(1e3);
#x0 = np.random.rand((n*N));
#timeLine = np.linspace(1,T ,T)
#xTraj = ut.runGradient(barA, x0,T=int(T));
#plt.figure();
#plt.title("no noise, check if these step sizes converges")
#plt.plot(timeLine, xTraj.T);
#plt.show();

##------------------ DISTURBANCE -------------------
#E  = np.array([1,0,0,0]);
#C = np.array([0,0,0,1]);
#B = np.array([0,0,1,0]);
#
#barE = np.zeros((n*N, N)); barC = np.zeros((N,n*N)); barB = np.zeros((n*N,N))
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
##print (F)
##print (barC.dot(barA+barB.dot(F)).dot(barE))
#
## -------------------simulate the system again-------------------
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
##print (F)
##print (barC.dot(barA+barB.dot(F)).dot(barE))
#
###---------- simulate noisy version -----------------------
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
#
###---------- simulate noisy version without feedback control -----------------------
#T = int(1e3);
#
#timeLine = np.linspace(1,T ,T);
##w = np.random.rand(9,T);
#
#xTraj = ut.runGradient((0.99*barA), x0, noise = barE.dot(w), T = int(T));
#plt.figure();
#plt.title("simulate the not DDed with NOISE")
#plt.plot(timeLine, xTraj.T );
#plt.show();