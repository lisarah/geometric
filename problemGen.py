#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:17:32 2019

@author: sarahli
"""
import numpy as np
import control as ctl
import random as rdn
import isa as isa
import subspace_linalg as subspace

multiplier = 3;
case2 = False;
A = np.zeros((3*multiplier, 3*multiplier)); 
A[1*multiplier:2*multiplier, 0:1*multiplier] = np.random.rand(1*multiplier,1*multiplier);#np.eye(1*multiplier);
A[2*multiplier:3*multiplier, 1*multiplier:2*multiplier] = np.random.rand(1*multiplier,1*multiplier); #np.eye(1*multiplier);

B = np.zeros((3*multiplier, 1*multiplier));
B[1*multiplier:2*multiplier, :] = np.eye(1*multiplier);

E = np.zeros((3*multiplier, 1*multiplier));
E[0:1*multiplier, :] = np.eye(1*multiplier);

nullC = np.zeros((3*multiplier,2*multiplier));
nullC[0:1*multiplier, 0:1*multiplier] = np.eye(1*multiplier);
nullC[1*multiplier:2*multiplier,1*multiplier:2*multiplier] = np.eye(1*multiplier);

#V0 = np.zeros((3*multiplier, 3*multiplier));
#V0[0:1*multiplier, 0:1*multiplier] = np.eye(1*multiplier);
#V0[1*multiplier:2*multiplier, 1*multiplier:2*multiplier] = np.eye(1*multiplier);


if case2:
    temp = 1.0*E;
    E = 1.0*B;
    B = temp;
V0 = 1.0*E; 
invariantSub = (isa.ISA(V0, A, B));

print (subspace.contained(E, invariantSub))
print (subspace.contained(invariantSub, nullC ))
#allDen = rdn.sample(range(20), 3)
#Tnum  = []; Tden = [];
#
#for i in range(3):
#    Tnum.append(rdn.sample(range(5), 2));
#    Tden.append(allDen);
#Ts = ctl.tf2ss([Tnum], [Tden])
#print (Ts)

    
#num = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
#den = [[[9., 8., 7.], [6., 5., 4.]], [[3., 2., 1.], [-1., -2., -3.]]]
#sys1 = ctl.tf2ss(num, den)