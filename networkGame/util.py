#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:33:38 2019

@author: sarahli
"""
import numpy as np

def truncate(mat, eps = 1e-10):
    M = mat.shape;
    for i in range(M[0]): 
        if mat[i] <= eps:
            mat[i] = 0;

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
        