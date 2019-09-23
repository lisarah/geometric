# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:52:44 2019

@author: craba

This file outputs figures for noise level vs strategy displacement from x_eq 
for both linear and nonlinear dynamics
"""
import numpy as np
import util as ut
import matplotlib.pyplot as plt
plt.close('all');
N = 4;
drawGraph = False;
J = np.eye(8)*2;
J[2:4, 0:2] = np.eye(2)*1.5;
J[4:6, 0:2] = np.eye(2)*1.5;
J[6:8, 2:4] = np.eye(2)*0.5;
J[6:8, 4:6] = np.eye(2)*-0.5;
alpha, beta, gamma = ut.returnStepSize(J);

# generate Offset
c = np.array([1,1, 1,-1, -1,-1, -1,1])*gamma;

# generate NE in the noiseless case
T = 1000;
x0 = np.random.rand(2*N);
linear = ut.runGradient(np.eye(2*N) - gamma*J, x0 = x0, T = T, offset=c);
timeLine = np.linspace(0, T, T);
ut.plot2D(linear);


# generate noise NE -- linear dynamics
# input noise is at player 1
T = 1000; 
noise = np.zeros((N*2,T));
samples = 100;
diff = np.zeros((N*2, samples));
epsilon = np.zeros(samples);
for sample in range(samples):
    epsilon[sample] = sample*0.1;
    noise[0:2,:] = epsilon[sample]*np.random.rand(2,T);
    
    linearNoisy = ut.runGradient(np.eye(2*N) - gamma*J,  
                                 x0 = x0, 
                                 noise = noise, 
                                 T = T, offset=c);
    diff[:, sample] = np.linalg.norm(linearNoisy - linear, axis = 1, ord = 2);
    
#    for player in range(N):
#        normDiff[player, sample] = np.sum( np.linalg.norm(
#            linear[player*2:(player+1)*2, :] - linearNoisy[player*2:(player+1)*2, :]
#            , ord = 2, axis = 0));
        

ut.plot2D(diff/T, epsilon);
plt.xlabel(r'$\epsilon$', fontsize = 15);
plt.yscale('log');
plt.ylabel(r'$\|\| y_i - x_i \|\|_2$', fontsize = 15);

#----------- non linear version of everything that happened just now -------#
nonLinearEps = 1.0;
def nonLinearGrad(xt):
    Df = np.zeros((8));
    Df[0:2] = 2*(xt[0:2] - c[0:2]);
    
    Df[2:4] = 2*(xt[2:4] - c[2:4]) + 1.5*xt[0:2] + nonLinearEps*0.05*xt[0:2].dot(xt[0:2])*xt[0:2];
    
    Df[4:6] = 2*(xt[4:6] - c[4:6]) + 1.5*xt[0:2] + nonLinearEps*0.05*xt[0:2].dot(xt[0:2])*xt[0:2];
    Df[6:8] = 2*(xt[6:8] - c[6:8]) + 0.5*(xt[2:4] - xt[4:6]) + nonLinearEps*0.1*(xt[2:4].dot(xt[2:4]) - xt[4:6].dot(xt[4:6]));
    return xt - gamma*Df;

# generate NE
T = 1000;
x0 = np.random.rand(2*N);
nonLinear = ut.runNonlinearGradient(nonLinearGrad, x0 = x0, T = T);
ut.plot2D(nonLinear);
#--------------------- noisy version ------------------------------#
noise = np.zeros((N*2,T));
samples = 100;
diff = np.zeros((N*2, samples));
epsilon = np.zeros(samples);
for sample in range(samples):
    epsilon[sample] = sample*0.1;
    noise[0:2,:] = epsilon[sample]*np.random.rand(2,T);
    
    nonLinearNoisy = ut.runNonlinearGradient(nonLinearGrad,  
                                 x0 = x0, 
                                 noise = noise, 
                                 T = T);
    diff[:, sample] = np.linalg.norm(nonLinearNoisy - nonLinear, 
                                     axis = 1, 
                                     ord = 2); 
#    if sample ==50:
#        ut.plot2D(nonLinearNoisy);

ut.plot2D(diff/T, epsilon);
plt.yscale('log');
plt.xlabel(r'$\epsilon$', fontsize = 15);
plt.ylabel(r'$\|\| y_i - x_i \|\|_2$', fontsize = 15);
