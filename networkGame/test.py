# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:43:39 2019

@author: craba
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import util as ut
import isa as isa
import decouplingController as dd
A = np.zeros((4,4));
A[1,0] = 1.; A[2,0] = 1.; A[3,1] = 1.; A[3,2] = 1.;
B = np.array([[0], [1], [0], [0]]);
E = np.array([[1], [0], [0], [0]]);
C = np.array([[0,0,0,1]]);

invariantSub = isa.ISA(isa.ker(C), A, B)

if (isa.contained(E, invariantSub)):
    print ("disturbance decouplable");
else:
    print ("not disturbance decouplable");
    

# if decouplable we do the next step: 
F = dd.ddController(invariantSub, A, B);
print (F)
print (C.dot(A+B.dot(F)).dot(E))