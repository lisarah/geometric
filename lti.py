# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:48:08 2021

@author: Sarah Li
"""
import numpy as np
import cvxpy as cvx
import scipy.linalg as sla


def run_dynamics(A, B, E, F, T, offset=None, x_init=None, noise=False):
    """ Run LTI dynamics for T time steps.
    The dynamics, given random initial conditions and random noise experienced 
    by noisy_player.
    x_{k+1} = Ax_k + Bu_k + Ed_k 
    
    Args:
      A: system dynamics matrix
      B: control input matrix
      F: controller matrix
      T: number of time steps to simulate
    Returns:
      x_hist: history of state values
    """
    N,M = B.shape
    if x_init is None:
        x_init = np.random.rand(N)
    if offset is None:
        offset = np.zeros(N)
    if noise:
        _,K = E.shape
    x_hist = [x_init]
    for t in range(T):
        cur_x = x_hist[-1]    
        next_x = A.dot(cur_x) + B.dot(F).dot(cur_x - offset) 
        if noise:
            noise_multiplier = 1e-1
            next_x += noise_multiplier * E.dot(np.random.rand(K))
        x_hist.append(next_x)
    return x_hist

def zero_hold_dynamics(A, B_list, delta_t =1e-1):
    """ Generate zero hold discretized (A, B)'s from continuous time dynamics.
    
    A_d = exp(A delta_T)
    B_d = A^{-1}(A_d - I)B
    g_d = A^{-1}(A_d - I)
    
    Args:
        - A: continuous time system dynamics.
        - B_list: continuous time controller dynamics for players.
        - delta_t: time interval for discretization.
    Returns:
        - A_d: discretized zero-hold system dynamics.
        - B_d_list: discretized zero-hold controller dynamics
    """
    B_total = np.concatenate(B_list, axis = 1)
    n, m_total = B_total.shape
    exponent_up = delta_t * np.concatenate((A,B_total), axis=1)
    exponent_low = delta_t * np.concatenate(
        (np.zeros((m_total, n)), np.eye(m_total)), axis=1)
    exponent_mat = np.concatenate((exponent_up, exponent_low), axis=0)
    
    discretized_mat= sla.expm(exponent_mat)
    A_d = discretized_mat[:n, :n]
    B_d = discretized_mat[:n, n:]
 
    return A_d, B_d 

def lqr_and_discretize(A, B_list, Q, R, discretization, continuous_lqr = True):
    """ Discretize slung load dynamics under cooperative control.
    
    Discretize the slung load dynamics and either introduce LQR before or after
    the discretization. 
    Good discretization values: 
        - Q = 2e6, discretizaiton = 1e-1, T = 1e2
        - Q = 5e7, discretization = 1e-3, T = 1e4
        
    Args:
        - A: system dynamics.
        - B_list: control matrices.
        - Q: lqr's Q.
        - R: lqr's R.
        - disretization: amount of discretization in time. 
        - continuous_lqr: if True, implement LQR before discretization, 
            if False, implement LQR after discretization. 
    Returns:
        - A_d: discretized closed loop dynamics.
        - B_d: discretized control matrix.
        - B_d_list: discretized control matrices, if continuous LQR is
            implemented, this list will be zero matrices.
        - K_total: discretized feedback controller.
    """
    # K_list = []
    B = np.concatenate(B_list, axis=1)
    n, m = B_list[0].shape
    # player_num = len(B_list)
    
    if continuous_lqr: # solve riccati for controller before discretization.
        P = sla.solve_continuous_are(A, B, Q, R)
        K_total = -sla.inv(R).dot(B.T).dot(P)
        A_net = A + B.dot(K_total)
    else: # discretize first 
        P = sla.solve_discrete_are(A, B, Q, R)
        A_net = 1*A
        
    # discretization - zero hold.    
    A_sys_d, B_d = zero_hold_dynamics(A_net, B_list, discretization)  
    
    if continuous_lqr: 
        A_d = A_sys_d
        lhs = A.T.dot(P) + P.dot(A) + P.dot(B).dot(K_total)
    else: # solve discrete riccati for controller and discrete dynamics.
        K_total = -sla.inv(R + B_d.T.dot(P).dot(B_d)).dot(
            B_d.T).dot(P).dot(A_sys_d)
        lhs = A_sys_d.T.dot(P).dot(A_sys_d) - P + A_sys_d.T.dot(
            P).dot(B_d).dot(K_total)
        A_d = A_sys_d + B_d.dot(K_total)

    if continuous_lqr:
        print('---- continuous time LQR.')
        e,v = np.linalg.eig(A_net)
        print(f'continuous time eigen values {(np.real(e))}')
        
    print('CARE/DARE equation is satisfied',np.allclose(lhs, -Q))  
    # for i in range(player_num):
    #     K_list.append(K_total[i*m:(i+1)*m, :])
    e,v = np.linalg.eig(A_d) 
    print(f'discretized A + BK eigen values {abs(e)}')
    return A_d, B_d, K_total
