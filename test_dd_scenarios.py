# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 13:56:29 2022

@author: Sarah Li
"""
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

import disturbance_decouple as dd
import subspace_linalg as slin
# ------------ some linearized model of Mach number dynamics --------- #
# mounier rouchon some examples of linear systems with delays
a = 1.
k = 0.5 # k < 1 is required
delta = 3.
w = 2.
xi = 4.
A = np.array([[-a, k*a*delta, 0], [0, 0, 1],[0, -w**2, -2*xi*w]])
B = np.array([[0],[0],[w**2]])
C = np.array([[delta], [1], [0]])
E = np.array([[1], [-delta], [a*delta*(1+k*delta**2)]])

# ------------ linearized equations for liquid monopropellant rocket motor with pressure feeding system --------- #
# robust control of uncertain distributed delay systems with applicaitons 
# to the stabilization of combustion in rocket motor chambers 
# https://www.sciencedirect.com/science/article/pii/S0005109801002321
gamma = 0.6
A = np.array([[-gamma, 0, gamma, 0],
              [0,   0,  0,  -5],
              [-0.5556, 0, -0.556, 0.5556],
              [0, 1, -1, 0]])
B = np.zeros((4, 1)); B[2,0] = 5.
C = np.zeros((4, 1)); C[2, 0] = 1.
A[1, 3] = -5; A[2, 0] = -0.5556; A[2,2] = -0.556; A[2, 3] = 0.5556
A[3, 1] = 1; A[3, 2] = -1
E = np.array([[1], [-gamma], [0], [1]])

state_dim, action_dim = B.shape

# step 1: check if set up is disturbance decouplable.
V_star = dd.isa(slin.ker(C.T), A, B)
if V_star is None:
    print('System has no disturbance decouplable subspace')
else:
    if not slin.contained(slin.image(E), V_star):
        print('Disturbance matrix intersects the decoupling space.')
    # step 2: system is decouplable. 
    # Use the Vstar to solve for the dd controller.
    else:
        # solve for the disturbance decoupling controller 
        V_0, F, alpha_0, P  = dd.disturbance_decoupling(C, A, B, 
                                                        return_alpha=True)
        F_k, alpha_k, P_k, eig_history = dd.solve_bmi(A, B, alpha_0, F, P, V_0, 
                                                      rho = 1e-5,verbose=True)
            
        
    # step 3 check system:
    # plotting results if needed      
    plt.figure()
    plt.plot(eig_history)
    # plt.title('Maximum real part of eig(A+BF) ')
    plt.title('2 norm of F')
    plt.grid()
    plt.show()

    # final result should give the fastest convergence rate of A+BF that's also DD
    A_cl = A + B.dot(F_k)
    B_empty = np.zeros((state_dim, action_dim))
    F_empty = np.zeros((action_dim, state_dim))
    e, v = sla.eig(A + B.dot(F_k))
    print(f"Eigen-values of closed loop system = {e}")