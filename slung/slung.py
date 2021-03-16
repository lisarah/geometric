# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:14:06 2021

@author: Sarah Li
"""
import numpy as np
import scipy.linalg as sla


def slung_dynamics_gen(mass, J, g_list):
    """ Generate the multi-lift slung load hovering dynamics.
    
    A payload is carried by multiple smaller vehicles. Dynamics are described
    in North-East-Down frame with Euler angles. See Chapter 2 of 
    https://etda.libraries.psu.edu/files/final_submissions/10269
    for further description.
        dot_x = Ax + Bu
    Dynamic variables (x):
        - (n, e, d): paylod's Center of gravity (CoG) in the NED frame.
        - (phi, theta, psi): Pitch, row, yaw Euler angles.
        - (v_n, v_e, v_d): Velocities of payload's CoG in NED axis directions.
        - (p, q, r): Angular velocity of payload's Euler angles. 
    
    Control is defined for each smaller carrier vehicle i. 
    Control variables (u_i):
        - (Fn, Fe, Fd): Forces in NED framework exerted by vehicle i. 
        
    Args:
        - mass: Mass of the payload.
        - J: 3D ndarray vector of payload's inertia at its center of gravity.
        - g_list: list of the locations of payload's cable connections to
        lifting UAVs, centered at the payload's CoG.
    Returns:
        - A: Continuous time system dynamics at hovering. 
        - [B_1,...B_m]: Control matrices for each carrier vehicle. 
    """
    n = 12 # number of independent dynamic variables.
    m = 6 # number of independent input variables per carrier vehicle.
    A = np.zeros((n, n))
    B = np.zeros((n,m))
    A[0:3, 6:9] = np.eye(3)
    A[3:6, 9:12] = np.eye(3)
    B[6:9, 0:3] = 1/mass*np.eye(3)
    for ind in range(3):
        B[9+ind, 3+ind] = 1/J[ind] 
    B_list = []
    for g in g_list:
        G_i = np.zeros((m, 3))
        G_i[0:3, 0:3] = np.eye(3)
        G_i[3,1] = -g[2] # -g_z
        G_i[3,2] = g[1] # g_y
        G_i[4,0] = g[2] # g_z
        G_i[4,2] = -g[0] # -g_x
        G_i[5,0] = -g[1] # -g_y
        G_i[5,1] = g[0] # g_x
        B_list.append(B.dot(G_i))
        
    return A, B_list


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
    B_d_list = []
    b_len = int((m_total - 1)/(len(B_list) - 1))
    print(f'number of players is {len(B_list)-1}, B_list length {m_total}, dim of each B {b_len}')
    for b_ind in range(len(B_list)-1):
        b_start = n + b_len * b_ind
        B_d_list.append(discretized_mat[:n, b_start: b_start + b_len])
    B_d_list.append(discretized_mat[:n, -1])    
    return A_d, B_d_list 
    
    
def sim_slung_dynamics(A, B_list, K_list, gravity_vector, time, x_0 = None):
    """ Simulate discretized slung dynamics.
    
    x_{t+1} = Ax_t + \sum_i B_i K_i x_t
    
    Args: 
        - A: discretized time system dynamics.
        - B_list: discretized controller dynamics.
        - K_list: feedback controllers corresponding to B_list.
        - gravity_vector: discretized gravity input.
        - time: total number of time steps.
        - x_0: initial system state, defaults to random point. 
    Returns:
        - x_hist: a list of system states for the first time time steps.
    """
    n,_ = A.shape
    if x_0 == None:
        x_0 = 1 * np.random.rand(n)
    x_hist = [x_0]
    for t in range(time):
        x_cur = x_hist[-1]
        x_next = A.dot(x_cur) + 9.81 * gravity_vector # downward acceleration
        for i in range(len(K_list)):
            x_next += B_list[i].dot(K_list[i]).dot(x_cur)
        x_hist.append(1*x_next)
    
    return x_hist