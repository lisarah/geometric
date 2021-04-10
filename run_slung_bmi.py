# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:53:14 2021

@author: Sarah Li
"""

import lti as lti
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
import slung.slung as slung
import disturbance_decouple as dd


plt.close('all')
discretization = 1e-2 
Time = int(200)

# payload definition
mass = 1 # kg.
J = [1,1,1] # moment of inertia (here assume diagonal).

# Cable attachment definition. (how each drone is connected to payload)
# Cable attachment position is the location at which the cables are attached
# to the payload, given in the payload NED frame in meters.
# g_list = [[1, 1, 0], [1, -1, 0], [1, 0, 0], [-1, 0, 0]] # natural decoupling
g_list = [[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]]
players = len(g_list)

# slung dynamics generation.
A, B_list = slung.slung_dynamics_gen(mass, J, g_list)
B = np.concatenate(B_list, axis=1)
n,m = B.shape
H = slung.state_matrix(['down', 'pitch', 'roll'])

#--------------------- solve disturbance decoupling -----------------------#
V, F_0, alpha_0, P_0 = dd.disturbance_decoupling(H, A, B,return_alpha=True)
VB = np.concatenate((V, B),axis=1)
_,v_col = V.shape

#--------------------- solve BMI -----------------------#
F_k, alpha_k, P_k, eig_history = dd.solve_bmi(A, B, alpha_0, F_0, P_0, V,
                                              verbose=True)
plt.plot(eig_history)
plt.title('Maximum real part of eig(A+BF) ')
plt.grid()
plt.show()

# final result should give the fastest convergence rate of A+BF that's also DD
A_cl = A + B.dot(F_k)
B_empty = np.zeros((n,m))
F_empty = np.zeros((m,n))
e, v = sla.eig(A + B.dot(F_k))
print(f"Eigen-values of closed loop system = {e}")

x0 = np.random.rand(n)
E = slung.state_matrix(['north velocity', 'east velocity'])
A_d, _ = lti.zero_hold_dynamics(A_cl, [B_empty], delta_t=1e-1)

# plot the discrete LQR with no noise
title = 'BMI derived DD controller and no noise'
x_hist = lti.run_dynamics(A_d, B_empty, E, F_empty, Time, x_init=x0)
slung.plot_slung_states(x_hist, title)

# add random noise in the north and east velocity directions
title = 'BMI derived DD controller WITH noise'
x_hist = lti.run_dynamics(A_d, B_empty, E, F_empty, Time, x_init=x0,
                          noise=True)
slung.plot_slung_states(x_hist, title)

