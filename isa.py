# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:15:43 2019

@author: craba
"""
import numpy as np
import numpy.linalg as la
import cvxpy as cvx
import matplotlib.pyplot as plt
import util as ut
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
        # print (f'vt = {Vt}')
        # print (sumS(Vt, B))
        Ainved =  AinvV(A, sumS(Vt, B));
#        print (Ainved) 
        Vnext = intersect(K, Ainved);
        print ("rank of Vt ", ut.rank(Vt));
        print ("rank of Vnext ", ut.rank(Vnext));
        print ("Vnext is contained in Vt ", contained(Vnext, Vt));
        if ut.rank(Vnext) == ut.rank(Vt):
            isInvariant = True;
            print ("ISA returns-----")
            print (Vnext);
            # print (Vt);
            
        else:
            Vt = Vnext;
        
    return Vt
#----------------------------------------------------------------------------#
;
#----------------------------------------------------------------------------#
"""
Return kernel of input matrix
    Input: matrix whose kernel is determined
    Output: kernel of matrix, outputted as columns of a matrix     
"""
def ker(A, eps = 1e-14):
    U,D, V = la.svd(A, full_matrices=True);
    r = ut.rank(A, eps);
    m,n = V.shape
    if r == m:
        return np.zeros(V.T.shape)
    else:
        return V[r:, :].T


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
    return ut.image(T);
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
    kerB = ker(B.T);
    cap = kerB.T.dot(A); 
    m,n = cap.shape;
    Zero = np.zeros((m,n))
    if np.allclose(cap, Zero):   
        return True;
    else: 
#        print (cap)
        return False;
#@title Code to simulate/visualize dynamics
def run_dynamics(A,B, E, F, T):
    """ Run the dynamics for T time steps, the dynamics, given random initial 
        conditions and random noise experienced by noisy_player.
        x_{k+1} = Ax_k + Bu_k + Ed_k 
      Args:
        A: system dynamics matrix
        B: control input matrix
        F: controller matrix
        T: number of time steps to simulate
    """
    N,M = B.shape
    x0 = np.random.rand(N)
    _,K = E.shape
    x_hist = [x0]
    for t in range(T):
        cur_x = x_hist[-1]
        next_x = A.dot(cur_x) + B.dot(F).dot(cur_x) + 0* E.dot(np.random.rand(K))
        x_hist.append(next_x)
    plt.figure()
    plt.plot(x_hist)
    plt.grid()
    plt.show()
  
np.random.seed(122323)
N = 10
M = 3
observed_player = 0
disturbed_player = 1
A = np.random.rand(N, N)
B = np.random.rand(N, M)
H = np.zeros((N, 1))
H[observed_player,0] = 1.

V = (ISA(ker(H.T), A, B))
VB = np.concatenate((V, B),axis=1)
# print(invariantSub)

# do optimization
F = cvx.Variable((M, N))
# P = cvx.Variable((N,N),PSD = True)
XS = cvx.Variable((9 + 3, 9))
constraint = [VB@XS == A.dot(V)]
constraint.append(XS[9:, :] == -F@V)
# constraint.append(A.T*P + P*A + (B*F).H*P + P*B*F << 0)
# obj = cvx.Minimize(cvx.norm2(F))# 
# obj = cvx.Minimize(cvx.norm2(A.T*P + P*A + F.H * B.T *P + P*B*F) )
obj = cvx.Minimize(cvx.norm2(A + B*F))
dd = cvx.Problem(obj, constraint)
F_norm = dd.solve(solver=cvx.MOSEK, verbose=True) #  solver=cvx.SCS,
F_opt = F.value
print(f'resuling controller: \n{np.round(A+B.dot(F_opt),1)}')


E = np.zeros((N,1))
E[disturbed_player] = 1
print(f'blue is the disturbance decoupled player')
run_dynamics(A,B,E,F_opt,int(4))