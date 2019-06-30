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
    Qii =  A.T.dot(A);
    w,v = np.linalg.eig(Qii);
    Q[:, :, i, i] = Qii / (0.9*np.min(w));
    w, v = np.linalg.eig(Q[:, :, i, i])   ;
    print (w)

# neighbours
Q[:,:, 0,1] = np.random.rand(m,m);
#Q[:,:, 1,0] = np.random.rand(m,m);
Q[:,:, 1,0] = Q[:,:, 0,1];
Q[:,:, 1,2] = np.random.rand(m,m);
Q[:,:, 2,1] = Q[:,:, 1,2];
#Q[:,:, 2,1] = np.random.rand(m,m);

T = 100
time = np.linspace(0,T-1, T);
xInit = np.random.rand(m,N) + 10.;
xTraj = np.zeros((m,N,T));
for t in range(T): 
#    print (t);
    for i in range(N):
#        print (i)
        if t == 0:
            xTraj[:,i,t] = xInit[:,i];
        else:
            BR = np.zeros(m)
            # best response
            for j in range(N):
                BR  += Q[:,:,i,j].dot(xTraj[:,j,t-1]);
            xTraj[:,i,t] = np.linalg.inv(Q[:,:,i,i]).dot(BR);
plt.figure();
plt.plot(time, xTraj[:,0,:].T, color='b');
plt.plot(time, xTraj[:,1,:].T, color = 'r');
plt.plot(time, xTraj[:,2,:].T, color = 'y');
plt.show();
        
            