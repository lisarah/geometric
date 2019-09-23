# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 22:39:05 2019

@author: craba
"""
import numpy as np
import util as ut
import matplotlib.pyplot as plt
plt.close('all');
N = 4;

J = np.eye(8)*2;
J[2:4, 0:2] = np.eye(2)*1.5;
J[4:6, 0:2] = np.eye(2)*1.5;
J[6:8, 2:4] = np.eye(2)*0.5;
J[6:8, 4:6] = np.eye(2)*-0.5;
alpha, beta, gamma = ut.returnStepSize(J);

# generate offset
c = np.array([1,1, 1,-1, -1,-1, -1,1])*gamma;
epsilon = 1.0;

def nonLinearGrad(xt):
    Df = np.zeros((8));
    Df[0:2] = 2*(xt[0:2] - c[0:2]);
    
    Df[2:4] = 2*(xt[2:4] - c[2:4]) + 1.5*xt[0:2] + epsilon*0.05*xt[0:2].dot(xt[0:2])*xt[0:2];
    
    Df[4:6] = 2*(xt[4:6] - c[4:6]) + 1.5*xt[0:2] + epsilon*0.05*xt[0:2].dot(xt[0:2])*xt[0:2];
    Df[6:8] = 2*(xt[6:8] - c[6:8]) + 0.5*(xt[2:4] - xt[4:6]) + epsilon*0.1*(xt[2:4].dot(xt[2:4]) - xt[4:6].dot(xt[4:6]));
    return xt - gamma*Df;


# generate NE
T = 1000;
x0 = np.random.rand(2*N);
linear = ut.runGradient(np.eye(2*N) - gamma*J, x0 = x0, T = T, offset=c);
timeLine = np.linspace(0, T, T);
ut.plot2D(linear);

# generate noise NE -- linear dynamics
# input noise is at player 1
T = 1000; 
halfT = int(T/2);
noise = np.zeros((N*2,T))
noise[0:2,:] = 0.01*np.random.rand(2,T);
linearNoisy = ut.runGradient(np.eye(2*N) - gamma*J,  x0 = x0, noise = noise, T = T, offset=c);
ut.plot2D(linearNoisy);

#----------- non linear version of everything that happened just now -------#
# generate NE
T = 1000;
x0 = np.random.rand(2*N);
nonLinear = ut.runNonlinearGradient(nonLinearGrad, x0 = x0, T = T);
ut.plot2D(nonLinear);

# input noise is at player 1
noise = np.zeros((2*N,T))
noise[0,:] = 0.01*np.random.rand(T);
nonLinearNoisy = ut.runNonlinearGradient(nonLinearGrad,  x0 = x0, noise = noise, T = T);
ut.plot2D(nonLinearNoisy);
