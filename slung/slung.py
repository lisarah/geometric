# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:14:06 2021

@author: Sarah Li
"""
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

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
    B_d = discretized_mat[:n, n:]
    # B_d_list = []
    # b_len = int((m_total - 1)/(len(B_list) - 1))
    # print(f'number of players is {len(B_list)-1}, B_list length {m_total}, '
    #       f'dim of each B {b_len}')
    # for b_ind in range(len(B_list)):
    #     b_start = n + b_len * b_ind
    #     B_d_list.append(discretized_mat[:n, b_start: b_start + b_len])  
    return A_d, B_d 


def lqr_and_discretize(A, B_list, Q, R, discretization, continuous_lqr = True):
    """ Discretize slung load dynamics under cooperative control.
    
        Discretize the slung load dynamics and either introduce LQR before or
    after the discretization. 
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


def sim_slung_dynamics(A, B_list, K_list, offset, time, x_0=None):
    """ Simulate discretized slung dynamics.
    
    x_{t+1} = Ax_t + \sum_i B_i K_i x_t
    
    Args: 
        - A: discretized time system dynamics.
        - B_list: discretized controller dynamics.
        - K_list: feedback controllers corresponding to B_list.
        - offset: offset vector to x_t at every time step.
        - time: total number of time steps.
        - x_0: initial system state, defaults to random point. 
    Returns:
        - x_hist: a list of system states for the first time time steps.
    """
    n,_ = A.shape
    if x_0 is None:
        x_0 = 1 * np.random.rand(n)
    x_hist = [x_0]
    for t in range(time):
        x_cur = x_hist[-1]
        diff = x_cur - offset
        x_next = A.dot(x_cur)
        for i in range(len(K_list)):
            x_next += B_list[i].dot(K_list[i]).dot(diff)
        x_hist.append(1*x_next)
    
    return x_hist

def plot_slung_states(x_hist, title):
    """Plot the output of sim_slung_dynamics.
    
    Args:
        - x_hist: a list of payload state history.
        - title: title of plot.
    """
    labels = ['north', 'east', 'down', 'phi', 'theta', 'psi']
    history = np.array(x_hist)
    plt.figure()
    plt.title(title)
    [plt.plot(history[:,i], label=labels[i]) for i in range(len(labels))]
    plt.grid()
    plt.legend()
    plt.show()