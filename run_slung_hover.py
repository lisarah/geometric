# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:54:59 2021

@author: Sarah Li
"""
import isa as isa
import numpy as np
import matplotlib.pyplot as plt
import slung.slung as slung
import scipy.linalg as sla
import util as ut
import cvxpy as cvx


plt.close('all')
continuous_lqr = True
discretization = 1e-2 if continuous_lqr else 1e-1 # 1e-1 for discrete LQR
Time = int(2e3)

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
n, _ = A.shape
_, m = B_list[0].shape
# B_total = np.concatenate(B_list, axis=1)
# print(f'systems controllable subspace has rank '
#       f'{ut.control_subspace(A, B_total)}')

# LQR parameter definition
R = 1e0 * np.eye(players*m)
Q = 1e1 * np.eye(n)

# generate LQR controllers. 
A_d, B_d, B_d_list, K_list = slung.lqr_and_discretize(
    A, B_list, Q, R, discretization, continuous_lqr=continuous_lqr)

x0 = np.random.rand(n)
offset = np.zeros(n)

# plot discretized LQR result with no noise
title = 'discretized LQR feedback'
x_hist = slung.sim_slung_dynamics(A_d, B_d_list, K_list, offset, Time, x_0=x0)
slung.plot_slung_states(x_hist, title)

#-------------- design disturbance decoupling controller -----------------#
# H is the observation matrix
observed_states = [
    # 0, # north
    2, # down
    3, # pitch
    4, # roll
    # 6, # north velocity
    # 8, # down velocity
    # 9, # pitch velocity
    # 10 # roll velocity
    ]
H = np.zeros((n, len(observed_states))) # decouple z and pitch and roll
for i in range(len(observed_states)):
    H[observed_states[i], i] = 1.
    
V = isa.ISA(isa.ker(H.T), A_d, B_d)
print(np.round(V,2))

# add random noise in the north and east velocity directions
# without any disturbance decoupling feedback yet. 
E = np.zeros((n, 2))
E[6, 0] = 1. # north velocity
E[7, 1] = 1. # east velocity
empty_F = np.zeros((m*players, n))
title = 'discretized LQR feedback with noise'
isa.run_noisy_dynamics(A_d ,B_d, E, empty_F, Time, offset, x0, title)
# check if noise is contained in V.
print(f'range of E is contained in V {isa.contained(E, V)}')

VB = np.concatenate((V, B_d),axis=1)
_, dd_dim = V.shape

# declare optimization variables
F = cvx.Variable((m*players, n))
XS = cvx.Variable((dd_dim + m * players, dd_dim))
# enforce disturbance decoupling constraint.
constraint = [VB@XS == A_d.dot(V)]
constraint.append(cvx.norm2(A_d + B_d@F) <=1.05)
constraint.append(XS[dd_dim:, :] == -F@V) 
# run optimization for minimal control effort
obj = cvx.Minimize(cvx.norm2(B_d@F))
dd = cvx.Problem(obj, constraint)
system_norm = dd.solve(solver=cvx.MOSEK, verbose=True) #  solver=cvx.SCS,

# abstract resulting controller and confirm constraints.
F_opt = F.value
XS_array = XS.value
X_0 = XS_array[:dd_dim, :]
lhs = V.dot(X_0) - B_d.dot(F_opt).dot(V)
rhs = A_d.dot(V)
print(f'DD condition is satisfied {np.allclose(lhs,rhs)}')

# plot the discrete LQR with no noise
title = 'discretized LQR feedback with DD controller and no noise'
dF_list = [F_opt[i*m:(i+1)*m, :] for i in range(players)]
x_hist = slung.sim_slung_dynamics(A_d, B_d_list, dF_list, offset, Time, x_0=x0)
slung.plot_slung_states(x_hist, title)


# add random noise in the north and east velocity directions
title = 'discretized LQR feedback with DD controller and noise'
isa.run_noisy_dynamics(A_d, B_d, E, F_opt, T=Time, offset=np.squeeze(offset), 
                       x_init=x0, title=title)


# check that the controller F is not unique on the range(V).
# F_1 = np.zeros((12,12))
# F_1[:,:9]  = 1e-4 * V
# X_1 = X_0 - np.linalg.inv(V.T.dot(V)).dot(V.T).dot(
#     B_d).dot(F_opt + F_1).dot(V)
# lhs = V.dot(X_1) - B_d.dot(F_1).dot(V)
# rhs = A_d.dot(V)
# print(f'(X_1, F_1): DD condition is satisfied {np.allclose(lhs,rhs)}')