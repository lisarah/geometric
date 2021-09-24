# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:42:30 2021

@author: t-sarahli
"""
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

import dd
import subspace_linalg as slin


state_dim = 10
action_dim = 5
observation_dim = 3
disturbance_dim = 3
A = np.random.random((state_dim, state_dim))
B = np.random.random((state_dim, action_dim))
# observation matrix
C = np.eye(state_dim)[:observation_dim, :].T
# disturbance matrix
E = np.eye(state_dim)[:, state_dim-observation_dim:]

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
        F_k, alpha_k, P_k, eig_history = dd.solve_bmi(A, B)
            
        
    # step 3 check system:
    # plotting results if needed        
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