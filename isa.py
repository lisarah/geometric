# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:15:43 2019

@author: craba
"""
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
#----------------------------------------------------------------------------#
def ISA(V0, A, B, eps = 0.0):
    K = V0;
    Vt= 1.0*V0;
    isInvariant = False;
    while not isInvariant:
        Ainved =  AinvV(A, sumS(Vt, image(B)))
        Vnext = intersect(K, Ainved);
        
        if la.matrix_rank(Vnext) == la.matrix_rank(Vt):
            isInvariant = True;
        else:
            Vt = Vnext;
        
    return Vt
"""
The range of input matrix
    Input: matrix whose range is determined
    Output: range of matrix, outputted as columns of a matrix     
"""
#----------------------------------------------------------------------------#
def image(X, tol = 1e-14):
    U,D,V = la.svd(X);
    r = la.matrix_rank(X);
    return U[:,:r];

#----------------------------------------------------------------------------#
""" 
Sums two subspaces together
    Input: matrix A and matrix B
    Output: a matrix whose columns are orthonormal and span the intersection
    of span(A) and span(B)
"""
def sumS(A, B):
    m,n = A.shape;
    x,y = B.shape    
    if m != x:
        raise Exception('input matrices need to be of same shape');
    T = np.hstack((A, B));
    return image(T);
""" 
Calculates subspace A^{-1}(im(V))
    by noting that (A^{-1}im(V)) = ker((A^Tker(V.T)).T)
"""
def AinvV(A, V):
    print ("In AinvV")
    print (A);
    print (V)
    kerV = sla.null_space(V.T);
    AtV = A.T.dot(kerV);
    return sla.null_space(AtV.T);

"""
Given matrix A and matrix B  return the intersection of their ranges
     by noting that A intersect B = ker( (ker(A.T) + ker(B.T)).T )
"""
def intersect(A,B):
    kerA = sla.null_space(A.T, 1e-14);
    kerB = sla.null_space(B.T, 1e-14);
    return sla.null_space(np.hstack((kerA, kerB)).T, 1e-14);


A = np.random.rand(10,10); B =  np.random.rand(10,3);

#V = np.zeros((4,3)); V[0,0] = 1.;
#B = np.zeros((4,3)); B[1,1] = 1.; B[0,0] = 1.;
#print (AinvV(A.dot(A.T), V))
#print (intersect(V,B))
print (ISA(np.eye(10), A, B))