# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:15:43 2019

@author: craba
"""
import numpy as np
def ISA(K, A, B, eps = 0.0):
    V0 = K;

        
    return D

def image(X, tol):
    U,D,V = np.linalg.svd(X);
    ind = np.argwhere(D < tol);
    return U[0:ind]; # range of X where singular value is larger than tolerance        