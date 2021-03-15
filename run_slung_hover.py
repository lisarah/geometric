# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:54:59 2021

@author: Sarah Li
"""
import slung.slung as slung

g_list = [[1, 1, 0], [1, -1, 0], [0, 1, 0]]
mass = 1
J = [1,1,1]

A, B_list = slung.slung_load_dynamics(mass, J, g_list)

# find LQ solution to this