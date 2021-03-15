# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:14:06 2021

@author: Sarah
"""
import numpy as np


def slung_load_dynamics(mass, J, g_list):
    """ Define the multi-lift slung load hovering dynamics.
    
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
        - A: System dynamics at hovering. 
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
        G_i[0,1] = -g[2] # -g_z
        G_i[0,2] = g[1] # g_y
        G_i[1,0] = g[2] # g_z
        G_i[1,2] = -g[1] # g_x
        G_i[2,0] = -g[1] # g_y
        G_i[2,1] = g[0] #g_x
        B_list.append(B.dot(G_i))
    return A, B_list

