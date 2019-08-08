#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:33:38 2019

@author: sarahli
"""
import numpy as np

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

def runGradient(A, x0, noise=None, T=1000):
    n  = x0.shape;
    xHist = np.zeros((n[0], T));
    for t in range(T):
        if t == 0:
            xHist[:,0] = x0;
        else:
            xHist[:, t] = A.dot(xHist[:, t-1]);

        if noise is not None:
            xHist[:,t] = noise[:,t] + xHist[:,t];
    return xHist;

def fullGraph(A,E, C):
    n,n = A.shape;
    n,m = E.shape;
    y, n = C.shape;
    R1 = np.concatenate((A, E, np.zeros((n,y))), axis = 1);
    R2 = np.zeros((m,n+m+y));
    R3 = np.concatenate((C, np.zeros((y,m+y))), axis = 1);
    return np.concatenate((R1, R2, R3));
    
