#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:33:38 2019

@author: sarahli
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
def truncate(mat, is2D = False, eps = 1e-10):
    if is2D:
        m,n = mat.shape;
        for i in range(m):
            for j in range(n):
                if abs(mat[i,j]) <= eps:
                    mat[i,j] = 0;
    else:
        M = mat.shape;
        for i in range(M[0]): 
            if abs(mat[i]) <= eps:
                mat[i] = 0;
    return mat;
def runNonlinearGradient(func, x0, noise = None, eps = 1.0, T = 1000):
#    print ("----------- non linear gradient ---------");
    n = x0.shape;
    xHist = np.zeros((n[0], T));
    for t in range(T):
        if t == 0:
            xHist[:,0] = 1.*x0;
        else:
            xHist[:, t] = func(xHist[:, t-1], eps);

        if noise is not None:
#            print ("adding noise: ", noise[:,t]);
#            prevHist = 1.0*xHist[:,t];
            xHist[:,t] = noise[:,t] + xHist[:,t];
#            print ("difference in history: ", xHist[:,t] - prevHist);
#    print ("------- end of nonlinear gradient -------");
    return xHist;
    
def runGradient(A, x0 = None, noise=None, T=1000, offset = None):
    if isinstance(x0,(np.ndarray)) == False:
        print("Found empty initial condition");
        n,m = A.shape;
        x0 = np.random.rand(n);
#    else:
#        print ("initial condition is given");
    n  = x0.shape;
    xHist = np.zeros((n[0], T));
    for t in range(T):
        if t == 0:
            xHist[:,0] = 1.*x0;
        else:
            xHist[:, t] = A.dot(xHist[:, t-1]);
            if offset is not None:
                xHist[:,t] += offset;
        if noise is not None:
#            print ("adding noise: ", noise[:,t]);
#            prevHist = 1.0*xHist[:,t];
            xHist[:,t] = noise[:,t] + xHist[:,t];
#            print ("difference in history: ", xHist[:,t] - prevHist);
    return xHist;

def fullGraph(A,E, C):
    n,n = A.shape;
    n,m = E.shape;
    y, n = C.shape;
    R1 = np.concatenate((A, E, np.zeros((n,y))), axis = 1);
    R2 = np.zeros((m,n+m+y));
    R3 = np.concatenate((C, np.zeros((y,m+y))), axis = 1);
    return np.concatenate((R1, R2, R3));
    
def returnStepSize(J):
    S = 0.5*(J + J.T);
    w, e = np.linalg.eig(S.T.dot(S));
    alpha = min(w);
    w2, e2 = np.linalg.eig(J.T.dot(J));
    beta = max(w2);
    
    return alpha, beta, alpha/beta;

def plot3D(xHist):
    twoN, M = xHist.shape;
    N = twoN/2;
    plt.figure();
    ax = plt.axes(projection='3d')
    # Data for a three-dimensional line
    zline = np.linspace(0, M, M, endpoint = False);
#    print (zline.shape);
#    print (xHist[0,:].shape)
    for i in range(int(N)):
        ax.plot3D(xHist[i*2,:], xHist[i*2+1,:], zline, '--', label = str(i+1));
#        ax.plot3D(noisyx[i*n,:].real,  noisyx[n*i+1,:].real, zline, '--'); #,label ='noisy '+str(i+1)
    plt.legend();
    plt.show(); 
    
def plot2D(xHist, xAxis = None):
    twoN, M = xHist.shape;
    N = twoN/2;
    if xAxis is not None:
        timeLine = xAxis;
    else:
        timeLine = np.linspace(0, M, M);
    plt.figure();
    for i in range(int(N)):
        plt.plot(timeLine, 
                 np.linalg.norm(xHist[i*2:(i+1)*2,:], ord = 2, axis = 0), 
                 label = str(i+1), 
                 linewidth=3.0,
                 alpha = 1);
    plt.grid();
    plt.legend();
    plt.show();

def plotDiff(x1, x2,N,T):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    timeLine = np.linspace(0, T, T);
    plt.figure();
    for i in range(N):
        plt.plot(timeLine, 
                 np.linalg.norm(x1[2*i:2*(i+1), :] - x2[2*i:2*(i+1)], ord = 2, axis=0), 
                 label = str(i+1),
                 color = colors[i]);
    plt.grid();
    plt.legend();
    
    plt.yscale('log')
    plt.show();