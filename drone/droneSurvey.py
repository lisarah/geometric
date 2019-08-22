# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:19:16 2019

@author: craba
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.close('all')

def droneDyn(graph, couplingCoeff, N, n):
    # graph: communication adjacency matrix
    # coupling coefficient: coefficient that the coupling matrices are multiplied by
    # N: number of agents
    # n: state space
    # returns a N*n x N*n matrix DW for quadratic games
    DW = np.eye(n*N);
    A = np.eye(n,n);
    B = couplingCoeff*np.eye(n,n);
    # Complete graph
    for i in range(N):
        DW[n*i:n*(i+1), n*i:n*(i+1)] = (A + A.T); 
        for j in range(N):
            if graph[i,j] != 0:    
                DW[n*i:n*(i+1), n*i:n*(i+1)] += -graph[i,j]*(B + B.T); 
                DW[n*i:n*(i+1), n*j:n*(1+j)] = B+B.T;

    # step sizes
    w, eigVecs = np.linalg.eig(0.25*(DW + DW.T).T.dot(DW + DW.T));
    alpha = np.min(w);
    w2, eigVecs = np.linalg.eig(DW.T.dot(DW));
    beta = np.max(w);
    gamma = np.eye(n*N)*alpha/beta; 
    if alpha == beta:
        gamma = gamma*couplingCoeff;
    L= np.eye(n*N) - gamma.dot(DW);
    wL, eigL = np.linalg.eig(L);
    print (wL);
    return L, gamma;

# drone cost functions
N = 5; # number of drones
n = 2; # state space size of each drone
T = 1000;
noiseBound = 3.;
A = np.eye(n,n); 
B = 0.24*np.eye(n);
ref = np.zeros((n*N,T));
ref = np.random.rand(n*N)*2.;


noise = np.random.rand(n,T) - 0.5;
#---------- game ------------------
completeGraph = np.ones((N,N)) - np.eye(N);
uncoupledGraph = np.zeros((N,N));
k4Graph= np.tril(np.ones((N,N)), k = -1);
lineGraph = np.zeros((N,N)); lineGraph[1:N, 0:N-1] = np.eye(N-1);
L, completeGamma = droneDyn(completeGraph,0.2, N, n);
uncoupledL, uncoupledGamma = droneDyn(uncoupledGraph, 0.04, N,n);
k4L, k4Gamma = droneDyn(k4Graph, 0.33, N,n);
lineL, lineGamma = droneDyn(lineGraph, 0.4, N, n);
#DW = np.eye(n*N);
#uncoupleDW = np.eye(n*N);
## Complete graph
#for i in range(N):
#    DW[n*i:n*(i+1), n*i:n*(i+1)] = (A + A.T); # 
#    uncoupleDW[n*i:n*(i+1), n*i:n*(i+1)] = (A + A.T); # 
##    print DW[i:2+i]
#    for j in range(N):
#        if j != i:            
#            DW[n*i:n*(i+1), n*i:n*(i+1)] += -(B + B.T); 
#            if j < i:
#                DW[n*i:n*(i+1), n*j:n*(1+j)] = B+B.T;
#                DW[n*j:n*(1+j), n*i:n*(i+1)] = (B + B.T);
## step sizes
#w, eigVecs = np.linalg.eig(0.25*(DW + DW.T).T.dot(DW + DW.T));
#alpha = np.min(w);
#w2, eigVecs = np.linalg.eig(DW.T.dot(DW));
#beta = np.max(w);
#gamma = np.eye(n*N)*alpha/beta; 
#
##gamma[0,0] = gamma[0,0] / 2;
##gamma[1,1] = gamma[1,1] / 2;
#L= np.eye(n*N) - gamma.dot(DW);
#uncoupledL = np.eye(n*N) - gamma.dot(uncoupleDW);
#wL, eigL = np.linalg.eig(L);
#print (wL);

diagA = np.zeros((n*N, n*N));
for i in range(N):
    diagA[i*n:(i+1)*n, i*n:(i+1)*n] = A + A.T;

#--------------propogating dynamics -----------------#
subStep  = 1;
x = np.zeros((n*N,T*subStep),dtype=np.complex_);
#noisyx = np.zeros((n*N, T*subStep),dtype=np.complex_);
uncoupledX= np.zeros((n*N, T*subStep), dtype=np.complex_);
k4x =  np.zeros((n*N, T*subStep), dtype=np.complex_);
linex =  np.zeros((n*N, T*subStep), dtype=np.complex_);
for t in range(T*subStep):
    if t == 0:
        x[:,0] = np.random.rand(n*N)*noiseBound;
#        noisyx[:,0] = 1.0*x[:,0];
        uncoupledX[:,0] = 1.0*x[:,0];
        k4x[:,0] = 1.0*x[:,0];
        linex[:, 0] = 1.0*x[:,0];
    else: 
        x[:,t] = L.dot(x[:,t-1]);
        uncoupledX[:,t] = uncoupledL.dot(uncoupledX[:, t-1]);
#        noisyx[:,t] = L.dot(noisyx[:,t-1]);
        k4x[:,t] = k4L.dot(k4x[:,t-1]);
        linex[:,t] = lineL.dot(linex[:,t-1])
        c= completeGamma.dot(diagA).dot(ref);
        x[:,t] += (c);
#        noisyx[:,t] += (c);
        uncoupledX[:,t] += uncoupledGamma.dot(diagA).dot(ref);
        k4x[:,t] += k4Gamma.dot(diagA).dot(ref);
        linex[:,t] += lineGamma.dot(diagA).dot(ref);
    # noise
#    noisyx[4:4+n, t] += noise[:,t];
plt.figure();
ax = plt.axes(projection='3d')
# Data for a three-dimensional line
zline = np.linspace(0, T*subStep, T*subStep, endpoint = False)

for i in range(N):
#    ax.scatter(uncoupledX[i*n,:].real,  uncoupledX[n*i+1,:].real, zline, label = 'uncoupled '+str(i+1));
#    ax.plot3D(k4x[i*n,:].real, k4x[i*n+1,:].real, zline, '--', label = 'k4 '+str(i+1));
    ax.plot3D(x[i*n,:].real,  x[n*i+1,:].real, zline, label = 'complete '+str(i+1));
#    ax.plot3D(linex[i*n,:].real,  linex[n*i+1,:].real, zline, ':',label = 'line '+str(i+1));
#    ax.plot3D(noisyx[i*n,:].real,  noisyx[n*i+1,:].real, zline, '--',label =str(i+1));
#    print(n*i, "   ", n*i+1)
plt.legend();
plt.show();

