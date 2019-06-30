#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 19:24:28 2019

@author: sarahli
"""
import numpy as np
import matplotlib.pyplot as plt
m = 10;
N = 3;



Q = np.zeros((m, m, N, N));
for i in range(N):
    A = np.random.rand(m,m);
    Qii = A.T.dot(A);
    w,v  = np.linalg.eig(Qii);
    Q[:, :, i, i] =Qii/0.9/np.min(w);

# neighbours
Q[:,:, 0,1] = np.random.rand(m,m);
w01, v01 = np.linalg.eig(Q[:,:, 0, 1]);
#Q[:,:, 1,0] = np.random.rand(m,m);
Q[:,:, 1,0] = Q[:,:, 0,1];
w10, v10 = np.linalg.eig(Q[:,:, 1, 0]);
Q[:,:, 1,2] = np.random.rand(m,m);
w12, v12 = np.linalg.eig(Q[:,:, 1, 2]);
Q[:,:, 2,1] = Q[:,:, 1,2];
#Q[:,:, 2,1] = np.random.rand(m,m);
w21, v21 = np.linalg.eig(Q[:,:, 2, 1]);

# normalization
Q[:,:,0,0] = Q[:,:,0,0]*np.max(np.abs(w01));
Q[:,:,1,1] = Q[:,:,1,1]*np.max(np.sqrt(np.abs(w10)**2 + np.abs(w12)**2));
Q[:,:,2,2] = Q[:,:,2,2]*np.max(np.abs(w21));

#for  i in range(N):
#    w, v = np.linalg.eig(Q[:,:,i,i]);
#    print (np.max(w));
    
T = 10;

time = np.linspace(0,T-1, T);
xInit = np.random.rand(m,N) + 10.;
xTraj = np.zeros((m,N,T));
for t in range(T): 
    for i in range(N):
        if t == 0:
            xTraj[:,i,t] = xInit[:,i];
        else:
            BR = np.zeros(m);
            # best response
            for j in range(N):
                BR  += Q[:,:,i,j].dot(xTraj[:,j,t-1]);
            xTraj[:,i,t] = -np.linalg.inv(Q[:,:,i,i]).dot(BR);
             
plt.figure();
plt.plot(time, xTraj[:,0,:].T, color='b');
plt.plot(time, xTraj[:,1,:].T, color = 'r');
plt.plot(time, xTraj[:,2,:].T, color = 'y');
plt.show();
        
            