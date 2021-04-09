# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:48:08 2021

@author: Sarah Li
"""
import numpy as np


def run_dynamics(A, B, E, F, T, offset=None, x_init=None, noise=False):
    """ Run the dynamics for T time steps, the dynamics, given random initial 
        conditions and random noise experienced by noisy_player.
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