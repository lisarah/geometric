# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:35:36 2021

@author: Sarah Li
"""
import numpy.linalg as la
import numpy as np


def rank(X, eps = 1e-14):
    m,n = X.shape;
    if n == 0: 
        return 0;
    else:
        return la.matrix_rank(X, eps)

"""
The range of input matrix
    Args: 
        - X:matrix whose range is determined
    Returns: 
        - range of matrix, outputted as columns of a matrix     
"""
def image(X):
    U,D,V = la.svd(X)
    r = rank(X);
    return U[:,:r]

def control_subspace(A, B):
    n,_ = A.shape
    controllable_list = [la.matrix_power(A, i).dot(B) for i in range(n)]
    controllability_matrix = np.concatenate(controllable_list, axis=1)
    return rank(controllability_matrix)
    