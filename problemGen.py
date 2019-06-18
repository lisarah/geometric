#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:17:32 2019

@author: sarahli
"""
#import numpy as np
import control as ctl
import random as rdn


allDen = rdn.sample(range(20), 3)
Tnum  = []; Tden = [];

for i in range(3):
    Tnum.append(rdn.sample(range(5), 2));
    Tden.append(allDen);
Ts = ctl.tf2ss([Tnum], [Tden])
print (Ts)

    
#num = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
#den = [[[9., 8., 7.], [6., 5., 4.]], [[3., 2., 1.], [-1., -2., -3.]]]
#sys1 = ctl.tf2ss(num, den)