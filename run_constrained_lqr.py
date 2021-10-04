# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 00:35:16 2021

@author: craba
"""
import cvxpy as cvx
import lti as lti
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
import slung.slung as slung
import disturbance_decouple as dd


plt.close('all')
delta_t = 1e-1
Time = int(200)
noise_mag = 1e-1

# payload definition
mass = 1 # kg.
J = [1,1,1] # moment of inertia (here assume diagonal).

# Cable attachment definition. (how each drone is connected to payload)
# Cable attachment position is the location at which the cables are attached
# to the payload, given in the payload NED frame in meters.
# g_list = [[1, 1, 0], [1, -1, 0], [1, 0, 0], [-1, 0, 0]] # natural decoupling
g_list = [[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]]

# slung dynamics generation.
A, B_list = slung.slung_dynamics_gen(mass, J, g_list)
B = np.concatenate(B_list, axis=1)
n,m = B.shape
H = slung.state_matrix(['down', 'pitch', 'roll'])
#--------------------- solve disturbance decoupling -----------------------#
V, F_0, alpha_0, P_0 = dd.disturbance_decoupling(H, A, B,return_alpha=True)
#------------------- run projection ------------------------------#
def project_to_dd(A, B, V, P, R):
    VB = np.concatenate((V,B.dot(R_inv).dot(B.T)), axis=1)
    
    _, v_col = V.shape
    n, _ = A.shape
    P = cvx.Variable((n, n),PSD=True)
    # X = cvx.Variable((v_col, v_col))
    XS = cvx.Variable((v_col + m, v_col))
    # constraint = [V@X - B@R_inv@B.T@P@V == A.dot(V)]
    constraint = [VB@XS == A.dot(V)]
    constraint.append(XS[v_col:, :] == -P@V) 
    dd_opt = cvx.Problem(cvx.Minimize(cvx.norm(P-P_0, 'nuc')), constraint)
    dd_opt.solve(solver=cvx.MOSEK, verbose=False) 
    P_proj = P.value
    
    # dd_condition = V.dot(X.value) - B.dot(R_inv).dot(B.T).dot(P_star).dot(V) - A.dot(V)
    return P_proj

#--------------------- solve LQR --------------------------#
# LQR parameter definition
R = 1e0 * np.eye(m)
R_inv = sla.inv(R)
Q = 8e0 * np.eye(n) 
P_k = sla.solve_continuous_are(A, B, Q, R)
# F_k = -R_inv.dot(B.T).dot(P_k)
# e, v = sla.eig(A + B.dot(F_k))
# print('eigenvalues from unconstrained CARE')
# print(np.real(e))
#---------------- solve LQR with Dynamic programming ----------#
P_k = Q
# P_k = project_to_dd(A, B, V, P_k, R)
norm_hist = [np.linalg.norm(P_k, 'fro')]
dd_condition = []
e, v = sla.eig(A - B.dot(R_inv).dot(B.T).dot(P_k))
eig_hist = [np.max(np.real(e))]
print(f'unsolved LQR eigenvalues {eig_hist[0]}')

total_steps = 200
progress_step = total_steps/5
for k in range(total_steps):
    if k%progress_step == progress_step-1:
        print(f'{k}/{total_steps}')
        P_k = project_to_dd(A, B, V, P_k, R)
        # e, v = sla.eig(A - B.dot(R_inv).dot(B.T).dot(P_k))
        # print(np.real(e))
        norm_hist.append(np.linalg.norm(P_k, 'fro'))
        e, v = sla.eig(A - B.dot(R_inv).dot(B.T).dot(P_k))
        eig_hist.append(np.max(np.real(e)))
        print(np.real(e))
        # print(np.linalg.norm(P_k, 'fro'))
    P_grad = A.T.dot(P_k) + P_k.dot(A) - P_k.dot(B).dot(R_inv).dot(B.T).dot(P_k) + Q
    # method 1: project and then gradient descent
    # P_proj_grad = P_grad
    # P_proj_grad = project_to_dd(A, B, V, P_grad, R)
    # P_k = P_k + P_proj_grad*1e-3
    # method 2: gradient descent and then project
    P_k =  P_k + P_grad*1e-3
    # P_k = project_to_dd(A, B, V, P_k, R)
    # P_k = project_to_dd(A, B, V, P_k, R)
    norm_hist.append(np.linalg.norm(P_k, 'fro'))
    e, v = sla.eig(A - B.dot(R_inv).dot(B.T).dot(P_k))
    eig_hist.append(np.max(np.real(e)))
    
plt.figure()
plt.plot(norm_hist)  
plt.show()

plt.figure()
plt.plot(eig_hist)  
plt.show()


    # F_k = -R_inv.dot(B.T).dot(P_k)
    # for k in range(100):
    #     #------------------- run projection ------------------------------#
    #     VB = np.concatenate((V,B), axis=1)
    #     _, v_col = V.shape
    #     P = cvx.Variable((n, n),PSD=True)
    #     X = cvx.Variable((v_col, v_col))
    #     constraint = [V@X - B@R_inv@B.T@P@V == A.dot(V)]
    #     dd_opt = cvx.Problem(cvx.Minimize(cvx.norm(P-P_0, 'fro')), constraint)
    #     dd_opt.solve(solver=cvx.MOSEK, verbose=True) 
    #     P_k = P.value
        
# abstract resulting controller and confirm constraints.
# F_array = F.value
# XS_array = XS.value
# X_0 = XS_array[:v_col, :]
# lhs = V.dot(X_0) - B.dot(F_array).dot(V)
# rhs = A.dot(V)

# x0 = np.random.rand(n)
# E = slung.state_matrix(['north velocity', 'east velocity'])
# A_d, B_d = lti.zero_hold_dynamics(A + B.dot(F_k), B, delta_t=delta_t)
# B_empty = np.zeros((n,m))
# F_empty = np.zeros((m,n))

# title = 'LQR derived DD controller WITH noise'
# x_hist_dd, _ = lti.run_dynamics(A_d, B_empty, E, F_empty, Time, 
#                           x_init=x0, noise=True, noise_mag=0e0)
# slung.plot_slung_states(x_hist_dd, title)
# slung.plot_slung_states(control_hist_dd, title +' control vals')
