# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:15:52 2021

Test the optimal design of an output feedback controller.

@author: Sarah Li
"""
import cvxpy as cvx
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

A = np.array([[2.45, -0.9, 1.53, -1.26, 1.76],
              [-0.12, -0.44, -0.01, 0.69, 0.9],
              [2.07, -1.2, -1.14, 2.04, -0.76],
              [-0.59, 0.07, 2.91, -4.63, -1.15],
              [-0.74, -0.23, -1.19, -0.06, -2.52]])
B = np.array([[0.81, -0.79, 0, 0, -0.95],
              [-0.34, -0.5, 0.06, 0.22, 0.92],
              [-1.32, 1.55, -1.22, -0.77, -1.14],
              [-2.11, 0.32, 0, -0.83, 0.59],
              [0.31, -0.19,-1.09, 0, 0]])
C = np.array([[0, 0, 0.16, 0, -1.78],
              [1.23, -0.38, 0.75, -0.38, 0],
              [0.46, 0, -0.05, 0, 0],
              [0, -0.12, 0.23, -0.12, 1.14]])

n, m  = B.shape
l, _= C.shape
def H(alpha, F, P):
    A_alpha = A + alpha * np.eye(n)
    B_F = B.dot(F).dot(C)
    return A_alpha + B_F - P

# find initial point (alpha_k , F_k, P_k)
F_k = np.zeros((m,l))
P = cvx.Variable((n,n), PSD=True)
e, v = sla.eig(A.T + A)
alpha_k = max(np.real(e)) * -0.5
A_BFC = A + B.dot(F_k).dot(C)
constraint = [A_BFC.T @ P + P @ A_BFC + 2*alpha_k*P << 0]
obj = cvx.Minimize(0)
initial_point = cvx.Problem(obj, constraint)
feasiblity_sol = initial_point.solve(verbose=False)
P_k = P.value
e, v = sla.eig(A + B.dot(F_k).dot(C))
print(f"Eigenvalues = {np.max(np.real(e))}")

F = cvx.Variable((m, l))
P = cvx.Variable((n,n), PSD=True)
alpha = cvx.Variable()
a_I = alpha * np.eye(n)
rho = 1e-2
eig_history = [np.max(np.real(e))]
alpha_history = alpha_k
T = 50
for k in range(T):
    print(f"--------- {k} -----------")
    H_k = H(alpha_k, F_k, P_k)
    H_var = H_k @ (2 * (A + a_I + B@F@C - P) - H_k)
    lmi_block = cvx.bmat([[H_var, (A + B@F@C + a_I  + P).T],
                          [A + B@F@C + a_I  + P,  np.eye(n)]])
    obj = cvx.Maximize(alpha - rho/2 * cvx.power(cvx.norm(F - F_k, 'fro'),2) 
                       - rho/2 * cvx.power(cvx.norm(P - P_k, 'fro'), 2) )
    k_iteration = cvx.Problem(obj, [lmi_block >> 0])
    sol = k_iteration.solve(solver=cvx.MOSEK, verbose=False)
    F_k = F.value
    P_k = P.value
    alpha_k = alpha.value
    e, v = sla.eig(A + B.dot(F_k).dot(C))
    print(f"Eigen-values = {np.max(np.real(e))}")
    print(f"alpha =  {alpha.value}")
    eig_history.append(np.max(np.real(e)))

plt.plot(eig_history)
plt.title('Maximum real part of eig(A-BFC) ')
plt.grid()
plt.show()

F_opt = F.value

e, v = np.linalg.eig(A_d + B_d.dot(F_opt))
print(f'eign values of DD dynamics:{abs(e)}')
# plot the discrete LQR with no noise
# title = 'discretized LQR feedback with DD controller and no noise'
x_hist = lti.run_dynamics(A_d, B_d, None, F_opt, Time, x0)
slung.plot_slung_states(x_hist, title + ' with DD no noise')


# final result should give the fastest convergence rate of A+BFC
# e, v = sla.eig(A + B.dot(F_res).dot(C))
# print(f"Eigen-values = {e}")
