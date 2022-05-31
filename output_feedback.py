# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:57:25 2022

@author: Sarah Li
"""
import numpy as np
import cvxpy as cvx
import slung.slung as slung


def output_feedback(A, B, return_alpha=False, verbose=False):
    """ Generate a distrubance decoupling controller F. 
    
    Args:
        - A: system dynamics matrix [ndarray, nxn].
        - B: control matrix [ndarray, nxm].
        - return_alpha: True if you want the corresponding laypunov pontential.
        - verbose: True if want to see cvx output, False otherwise.
    Returns:
        - alpha: largest eigen value of A - BF [float].
        - F: a stable controller [ndarray, mxn].. 
    """        
    n, m = B.shape
    L = cvx.Variable((m, n)) # L = FW
    W = cvx.Variable((n, n), symmetric=True) # W = P_inv
    alpha = cvx.Variable()
    constraints = [A@W + W.T@A.T - B@L - L.T@B.T + alpha*np.eye(n) << 0]
    constraints.append(W >> 0)
    constraints.append(alpha <= 1e-10)
    # constraint.append(XS[v_col:, :] == -F@V) 
    stab_feedback = cvx.Problem(cvx.Maximize(alpha), constraints)
    stab_feedback.solve(solver=cvx.SCS, verbose=True) # solver=cvx.MOSEK,
    L_val = L.value
    W_val = W.value
    F = L_val.dot(np.linalg.inv(W_val))
    e, v = np.linalg.eig(A - B.dot(F))
    print(f' eigen values {e}')
    if not return_alpha:
        return F
    else:
        return F, max(e)

# payload definition
mass = 1 # kg.
J = [1,1,1] # moment of inertia (here assume diagonal).

# Cable attachment definition. (how each drone is connected to payload)
g_list = [[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]]
A, B_list = slung.slung_dynamics_gen(mass, J, g_list)
B = np.concatenate(B_list, axis=1)
F = output_feedback(A, B, return_alpha=True, verbose=True)
