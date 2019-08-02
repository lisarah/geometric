# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:15:43 2019

@author: craba
"""
import numpy as np
import numpy.linalg as la
#np.random.seed(122323)
#----------------------------------------------------------------------------#
def ISA(V0, A, B, eps = 0.0):
    K = V0;
    Vt= 1.0*V0;
    isInvariant = False;
    iteration = 0;
    while not isInvariant:
        iteration += 1;
        print ("--------------in interation ", iteration);
#        print (Vt)
#        print (sumS(Vt, B))
        Ainved =  AinvV(A, sumS(Vt, B));
#        print (Ainved) 
        Vnext = intersect(K, Ainved);
        print ("rank of Vt ", rank(Vt));
        print ("rank of Vnext ", rank(Vnext));
        print ("Vnext is contaned in Vt ", contained(Vnext, Vt));
        if rank(Vnext) == rank(Vt):
            isInvariant = True;
            print ("ISA returns-----")
            print (Vnext);
            print (Vt);
            
        else:
            Vt = Vnext;
        
    return Vt
#----------------------------------------------------------------------------#
def rank(X, eps = 1e-14):
    m,n = X.shape;
    if n == 0: 
        return 0;
    else:
        return la.matrix_rank(X, eps);
#----------------------------------------------------------------------------#
"""
Return kernel of input matrix
    Input: matrix whose kernel is determined
    Output: kernel of matrix, outputted as columns of a matrix     
"""
def ker(A, eps = 1e-14):
    U,D, V = la.svd(A);
    r = rank(A, eps);
    m,n = V.shape
    if r == m:
        return np.zeros(V.T.shape)
    else:
        return V[r:, :].T
#----------------------------------------------------------------------------#
"""
The range of input matrix
    Input: matrix whose range is determined
    Output: range of matrix, outputted as columns of a matrix     
"""
def image(X, tol = 1e-14):
    U,D,V = la.svd(X);
    r = rank(X);
    return U[:,:r];

#----------------------------------------------------------------------------#
""" 
Sums two subspaces together
    Input: matrix A and matrix B
    Output: a matrix whose columns are orthonormal and span the union
    of span(A) and span(B)
"""
def sumS(A, B):
    m,n = A.shape;
    x,y = B.shape    
    if m != x:
        raise Exception('input matrices need to be of same shape');
    T = np.hstack((A, B));
    return image(T);
#----------------------------------------------------------------------------#
""" 
Calculates subspace A^{-1}(im(V))
    by noting that (A^{-1}im(V)) = ker((A^Tker(V.T)).T)
"""
def AinvV(A, V):
#    print ("In AinvV")
#    print (V)
    kerV = ker(V.T);
    
    AtV = A.T.dot(kerV);
    return ker(AtV.T);
#----------------------------------------------------------------------------#
"""
Given matrix A and matrix B  return the intersection of their ranges
     by noting that A intersect B = ker( (ker(A.T) + ker(B.T)).T )
"""
def intersect(A,B):
    kerA = ker(A.T, 1e-14);
    kerB = ker(B.T, 1e-14);
#    print("In intersect")
#    print (kerA.shape);
#    print (kerB.shape);
    return ker(np.hstack((kerA, kerB)).T, 1e-14);
#----------------------------------------------------------------------------#
def contained(A,B):
    # this calculates if range of A is contained in range of B
    # by seeing if null(B^T) is contained in null(A^T)
    kerB = ker(B.T);
    cap = kerB.T.dot(A); 
    m,n = cap.shape;
    Zero = np.zeros((m,n))
    if np.allclose(cap, Zero):   
        return True;
    else: 
#        print (cap)
        return False;
    
#N = 50;
#halfN = 48;
#B =  np.random.rand(N,4);
##A = np.zeros((N,N)); A[:halfN, :halfN] = np.eye(halfN);
##A[halfN:, halfN:]  = np.random.rand(N -halfN,N -halfN)
#A = np.eye(N)
#V = np.zeros((N,3)); V[0,0] = 1.;V[1,1] = 1.;
#U = np.zeros((N,2)); U[0,0] = 1;
#
#K = ker(B.T);
##K = np.random.rand(N,2);
##print (intersect(K, B));
#invariantSub = (ISA(ker(K.T), -A.dot(A.T), B));
#print (contained(B, invariantSub))
#if rank(image(B)) == rank(intersect(B, invariantSub)):
#    print ("can be disturbance decoupled");
#else: 
#    print ("no DD available")
