# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:06:08 2019

@author: craba
"""

import isa as isa
import numpy as np
"""
Calculate the controller that renders V invariant
    Input:  V     the subspace to remain invariant
            (A,B) the linear dynamics
 
    Output: kernel of matrix, outputted as columns of a matrix     
"""
def ddController(V, A, B):
    print (B.shape)
    lhs = np.concatenate((V,-B), axis = 1);
    rhs = A.dot(V);
    sol = np.linalg.pinv(lhs).dot(rhs);
    rowV, colV = sol.shape;
    X = sol[0:colV, :];
    Q = sol[colV:, :];
    FT = np.linalg.pinv(V.T).dot(Q.T);
    F = FT.T;
    
    return F;