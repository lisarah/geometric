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
players = 4
# positions of cable attachment to payload.
g_list = [[1, 1, 0], [1, -1, 0], [1, 0, 0], [-1, 0, 0]]
mass = 1 # kg.
J = [1,1,1] # moment of inertia (here assume diagonal).

A, B_list = slung.slung_dynamics_gen(mass, J, g_list)

n, _ = A.shape
_, m = B_list[0].shape

# find LQ solution to this
B_total = np.concatenate(B_list, axis=1)

print(f'systems controllable subspace has rank '
      f'{ut.control_subspace(A, B_total)}')
R = np.eye(players*m)
R[2,2] = 10.
Q = 100 * np.eye(n)
K_list = []
P = sla.solve_continuous_are(A, B_total, Q, R)
lhs = (A.T.dot(P) + P.dot(A) 
        - P.dot(B_total).dot(sla.inv(R)).dot(B_total.T).dot(P))
print('CARE equation is satisfied',np.allclose(lhs, -Q))
K_total = -sla.inv(R).dot(B_total.T).dot(P)
K_list = []
for i in range(players):
    K_list.append(K_total[i*m:(i+1)*m, :])
e,v = np.linalg.eig(A + B_total.dot(K_total))
print(f'continuous time eigen values {e}')

A_net_cts = A + B_total.dot(K_total)

# define gravity vector input
gravity_vector = np.zeros((n,1))
gravity_vector[8, 0] = 1
A_net, _ = slung.zero_hold_dynamics(A_net_cts, B_list+[gravity_vector])
A_d, B_d_list = slung.zero_hold_dynamics(A, B_list+[gravity_vector], 1e-2)
gravity_d_vector = B_d_list.pop(-1)  # exclude gravity vector.
B_d = np.concatenate(B_d_list, axis=1)

e,v = np.linalg.eig(A_d + B_d.dot(K_total))
print(f'discretized A + BK eigen values {abs(e)}')
B_zero_list = [np.zeros((n, m)) for p in range(players)]
x_hist = slung.sim_slung_dynamics(A_d, B_d_list, K_list, gravity_d_vector, 
                                  int(1e4), x_0 = None)


plot_history = np.array(x_hist)
labels = ['north', 'east', 'down', 'phi', 'theta', 'psi']

plt.figure()
for ind in range(3):
    plt.plot(plot_history[:,ind], label=labels[ind])
plt.grid()
plt.legend()
plt.show()

plt.figure()
for ind in range(3,6):
    plt.plot(plot_history[:,ind], label=labels[ind])
plt.grid()
plt.legend()
plt.show()

#----------- design disturbance decoupling controller-------------#
#----------- discrete case -----------#
# H = np.zeros((n,6)) # observe the angles
# H[3, 0] = 1. # phi angle.
# H[4, 1] = 1. # theta angle.
# H[5, 2] = 1. # psi angle.
# H[9, 3] = 1. # phi velocity.
# H[10, 4] = 1. # theta velocity.
# H[11, 5] = 1. # psi velocity
H = np.zeros((n,3)) # decouple z and pitch and roll
H[8, 0] = 1.
H[9, 1] = 1.
H[10, 2] = 1.

V = isa.ISA(isa.ker(H.T), A_net, B_d)
print(np.round(V,2))
# # V = V[:, 3:5] # just the v_x, v_y noises

VB = np.concatenate((V, B_d),axis=1)
_, dd_dim = V.shape

# # do optimization
F = cvx.Variable((m*players, n))
# P = cvx.Variable((N,N),PSD = True)
# dim(XS) = dd_dim + dim(B, col) by dd_dim
XS = cvx.Variable((dd_dim + m * players, dd_dim))
constraint = [VB@XS == A_net.dot(V)]
constraint.append(XS[dd_dim:, :] == -F@V) 
obj = cvx.Minimize(cvx.norm2(A_net + B_d@F))
dd = cvx.Problem(obj, constraint)
system_norm = dd.solve(solver=cvx.MOSEK, verbose=True) #  solver=cvx.SCS,
F_opt = F.value
print(f'resuling dynamics: \n{np.round(A_d+B_d.dot(F_opt),1)}')
DD_list = []
for i in range(players):
    DD_list.append(F_opt[i*m:(i+1)*m, :])
    
    
x_hist = slung.sim_slung_dynamics(A_net, B_d_list, DD_list, gravity_d_vector, 
                                  int(1e2), x_0 = None)


plot_history = np.array(x_hist)
labels = ['north', 'east', 'down', 'phi', 'theta', 'psi']

plt.figure()
for ind in range(3):
    plt.plot(plot_history[:,ind], label=labels[ind])
plt.grid()
plt.legend()
plt.show()

plt.figure()
for ind in range(3,6):
    plt.plot(plot_history[:,ind], label=labels[ind])
plt.grid()
plt.legend()
plt.show()

# add random noise in the north and east velocity directions
E = np.zeros((n, 2))
E[6, 0] = 1.
E[7, 1] = 1.
isa.run_noisy_dynamics(A_net,B_d, E, F_opt, T=int(1e3))