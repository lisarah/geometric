# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:54:59 2021

@author: Sarah Li
"""
import numpy as np
import matplotlib.pyplot as plt
import slung.slung as slung
import scipy.linalg as sla
import util as ut


plt.close('all')
players = 4
g_list = [[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]]
mass = 1
J = [1,1,1]

A, B_list = slung.slung_dynamics_gen(mass, J, g_list)

n, _ = A.shape
_, m = B_list[0].shape

# find LQ solution to this
B_total = np.concatenate(B_list, axis=1)

print(f'systems controllable subspace has rank '
      f'{ut.control_subspace(A, B_total)}')
R = np.eye(players*m)
Q = 1e5 * np.eye(n)
K_list = []
P = sla.solve_continuous_are(A, B_total, Q, R)
lhs = (A.T.dot(P) + P.dot(A) 
        - P.dot(B_total).dot(sla.inv(R)).dot(B_total.T).dot(P))
print('CARE equation is satisfied',np.allclose(lhs, -Q))
K_total = -sla.inv(R).dot(B_total.T).dot(P)
K_list = []
for i in range(players):
    K_list.append(K_total[i*m:(i+1)*m, :])
# e,v = np.linalg.eig(A + B_total.dot(K_total))
# print(e)

# define gravity vector input
gravity_vector = np.zeros((n,1))
gravity_vector[8, 0] = 1
A_d, B_d_list = slung.zero_hold_dynamics(A, B_list+[gravity_vector])
gravity_d_vector = B_d_list.pop(-1)  # exclude gravity vector.
# B_d_total = np.concatenate(B_d_list, axis=1)
# e,v = np.linalg.eig(A_d + B_d_total.dot(K_total))
# print(abs(e))
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