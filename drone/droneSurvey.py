# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:19:16 2019

@author: craba
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.close('all')
# drone cost functions
N = 4; # number of drones
n = 2; # state space size of each drone
T = 5;
A = np.eye(n,n); 
B = 0.23*np.eye(n);
ref = np.zeros((n*N,T));
ref = np.random.rand(n*N)*10.;

noise = np.random.rand(n,T) - 0.5;
#---------- game ------------------
subStep  = 1;
x = np.zeros((n*N,T*subStep));
noisyx = np.zeros((n*N, T*subStep));
DW = np.eye(n*N);
# Complete graph
for i in range(N):
    DW[n*i:n*(i+1), n*i:n*(i+1)] = (A + A.T); # 
#    print DW[i:2+i]
    for j in range(N):
        if j != i:            
            DW[n*i:n*(i+1), n*i:n*(i+1)] += -(B + B.T); 
            if j < i:
                DW[n*i:n*(i+1), n*j:n*(1+j)] = B+B.T;
#            DW[2*j:2*(1+j), 2*i:2*(i+1)] = (B + B.T);
# step sizes
w, eigVecs = np.linalg.eig(0.25*(DW + DW.T).T.dot(DW + DW.T));
alpha = np.min(w);
w2, eigVecs = np.linalg.eig(DW.T.dot(DW));
beta = np.max(w);
gamma = np.eye(n*N)*alpha/beta; 
diagA = np.zeros((n*N, n*N));
for i in range(N):
    diagA[i*n:(i+1)*n, i*n:(i+1)*n] = A + A.T;
#gamma[0,0] = gamma[0,0] / 2;
#gamma[1,1] = gamma[1,1] / 2;
L= np.eye(n*N) - gamma.dot(DW);
wL, eigL = np.linalg.eig(L);
print (wL);
#--------------propogating dynamics -----------------#
for t in range(T*subStep):
    if t == 0:
        x[:,0] = np.random.rand(n*N)*3;

    else: 
        x[:,t] = L.dot(noisyx[:,t-1]);
        c= gamma.dot(diagA).dot(ref);
        x[:,t] += c;
        # noise
    noisyx[:,t] = 1.0*x[:,t];
    noisyx[0:n, t] += noise[:,t];
plt.figure();
ax = plt.axes(projection='3d')
# Data for a three-dimensional line
zline = np.linspace(0, T*subStep, T*subStep, endpoint = False)

for i in range(N):
    ax.plot3D(x[i*n,:],  x[n*i+1,:], zline);
    ax.plot3D(noisyx[i*n,:],  noisyx[n*i+1,:], zline, '--');
#    print(n*i, "   ", n*i+1)

plt.show();

