# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:51:24 2019

@author: craba
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.close('all')
# drone cost functions
N = 2;
T = 20;
A = np.eye(2); B = 0.41*np.eye(2);
ref1 = np.zeros((2,T)); ref2 = np.zeros((2,T));
ref1[0,:] = np.linspace(0, T, T,endpoint = False);
ref2[0,:] = np.linspace(0, T, T, endpoint = False);
ref1[1,:] = np.sin(np.linspace(0,T/10., T)) + 1.;
ref2[1,:] = -np.sin(np.linspace(0,T/10., T)) +2.;

plt.figure();

plt.plot(ref1[0,:], ref1[1,:]);
plt.plot(ref2[0,:], ref2[1,:]);
plt.show();

#---------- game ------------------
subStep  = 10;
x = np.zeros((2*N,T*subStep));
DW = np.eye(2*N);

DW[0:2, 0:2] = (A + A.T- B - B.T); # 
DW[0:2, 2:4] = (B + B.T);
DW[2:4, 0:2] = (B + B.T);
DW[2:4, 2:4] = (A + A.T - B - B.T); # 
# step sizes
w, eigVecs = np.linalg.eig(0.25*(DW + DW.T).T.dot(DW + DW.T));
alpha = np.min(w);
w2, eigVecs = np.linalg.eig(DW.T.dot(DW));
beta = np.max(w);
gamma = np.eye(2*N)*alpha/beta; 
#gamma[0,0] = gamma[0,0] / 2;
#gamma[1,1] = gamma[1,1] / 2;
L= np.eye(2*N) - gamma.dot(DW);
for t in range(T*subStep):
    refInd  = int(t/subStep);
    if t == 0:
        x[:,0] = np.concatenate((ref1[:,0], ref2[:,0]));
    else: 
        x[:,t] = L.dot(x[:,t-1]);
        c = np.concatenate((gamma[0,0]*(A + A.T).dot(ref1[:,refInd]), 
                            gamma[1,1]*(A + A.T).dot(ref2[:,refInd]) ));
        x[:,t] += c;

plt.figure();
ax = plt.axes(projection='3d')
# Data for a three-dimensional line
zline = np.linspace(0, T*subStep, T*subStep, endpoint = False)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(x[0,:],  x[1,:], zline)
ax.plot3D(x[2,:],  x[3,:], zline)
plt.show();

