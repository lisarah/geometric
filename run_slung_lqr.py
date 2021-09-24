# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:41:58 2021

@author: craba
"""

import lti as lti
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
import slung.slung as slung
import disturbance_decouple as dd


plt.close('all')
discretization = 1e-2 
Time = int(1000)

# payload definition
mass = 1 # kg.
J = [1,1,1] # moment of inertia (here assume diagonal).

# Cable attachment definition. (how each drone is connected to payload)
# Cable attachment position is the location at which the cables are attached
# to the payload, given in the payload NED frame in meters.
g_list = [[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]]
players = len(g_list)

A, B_list = slung.slung_dynamics_gen(mass, J, g_list)
n, m = B_list[0].shape
H = slung.state_matrix(['down', 'pitch', 'roll'])

#--------------------- solve LQR --------------------------#
# LQR parameter definition
R = 1e0 * np.eye(players*m)
Q = 8e1 * np.eye(n) 

# generate LQR controllers. 
A_d, B_d, K_total = lti.lqr_and_discretize(
    A, B_list, Q, R, discretization, continuous_lqr=True)

#-------------------- simulate LQR -----------------------------#
x0 = np.random.rand(n)
E = slung.state_matrix(['north velocity', 'east velocity'])
offset = np.zeros(n)
 
# plot continuous LQR result with no noise
title = 'continuous LQR feedback'
B_zero = np.zeros(B_d.shape)
F_zero = np.zeros((m,n))
x_hist = lti.run_dynamics(A_d, B_zero, None, K_total, Time, x0)
slung.plot_slung_states(x_hist, title +' no DD no noise')

# add random noise in the north and east velocity directions
x_hist = lti.run_dynamics(A_d, B_zero, E, K_total, Time, x_init=x0,
                          noise=True)
slung.plot_slung_states(x_hist, title +' no DD with noise')


#--------------------- solve disturbance decoupling -----------------------#
B = np.concatenate(B_list, axis=1)

P = sla.solve_continuous_are(A, B, Q, R)
K_lqr = -sla.inv(R).dot(B.T).dot(P)
A_lqr = A + B.dot(K_lqr)

V, F_0, alpha_0, P_0 = dd.disturbance_decoupling(H, A_lqr, B, return_alpha=True)
VB = np.concatenate((V, B),axis=1)
_,v_col = V.shape

#--------------------- solve BMI -----------------------#
F_k, alpha_k, P_k, eig_history = dd.solve_bmi(A_lqr, B, alpha_0, F_0, P_0, V,
                                              verbose=False)
plt.figure()
plt.plot(eig_history)
plt.title('Maximum real part of eig(A+BF) ')
plt.grid()
plt.show()

# plot continuous LQR + DD results
B_zero = np.zeros(B_d.shape)
F_zero = np.zeros((m,n))

# final result should give the fastest convergence rate of A+BF that's also DD
A_cl = A_lqr + B.dot(F_k)
A_cl_d, _ = lti.zero_hold_dynamics(A_cl, [B_zero], delta_t=1e-1)

x_hist = lti.run_dynamics(A_cl_d, B_zero, None, K_total, Time, x0)
slung.plot_slung_states(x_hist, title +' with DD no noise')

# add random noise in the north and east velocity directions
x_hist = lti.run_dynamics(A_cl_d, B_zero, E, K_total, Time, x_init=x0,
                          noise=True)
slung.plot_slung_states(x_hist, title +' with DD with noise')

