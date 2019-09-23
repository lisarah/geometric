# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 22:39:05 2019

Increasing non linearity with fixed noise
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
#--------------- non linear version -----------------#
def nonLinearGrad(xt, eps):
    Df = np.zeros((8));
    Df[0:2] = 2*(xt[0:2] - c[0:2]);
    
    Df[2:4] = 2*(xt[2:4] - c[2:4]) + 1.5*xt[0:2] + eps*0.05*xt[0:2].dot(xt[0:2])*xt[0:2];
    
    Df[4:6] = 2*(xt[4:6] - c[4:6]) + 1.5*xt[0:2] + eps*0.05*xt[0:2].dot(xt[0:2])*xt[0:2];
    Df[6:8] = 2*(xt[6:8] - c[6:8]) + 0.5*(xt[2:4] - xt[4:6]) + eps*0.1*(xt[2:4].dot(xt[2:4]) - xt[4:6].dot(xt[4:6]));
    return xt - gamma*Df;

# generate NE in the noiseless case
T = 1000;
x0 = np.random.rand(2*N);
nonLinear = ut.runNonlinearGradient(nonLinearGrad, x0 = x0, T = T);
ut.plot2D(nonLinear);

#--------------------- noisy version ------------------------------#
noise = np.zeros((N*2,T));
noise[0:2,:] = 1.0*np.random.rand(2,T);
samples = 100;
diff = np.zeros((N*2, samples));
epsilon = np.zeros(samples);
for sample in range(samples):
    epsilon[sample] = sample*0.1;
    nonLinear = ut.runNonlinearGradient(nonLinearGrad, 
                                        x0 = x0, 
                                        eps = epsilon[sample],
                                        T = T);
    nonLinearNoisy = ut.runNonlinearGradient(nonLinearGrad,  
                                 x0 = x0, 
                                 noise = noise,
                                 eps = epsilon[sample],
                                 T = T);
    diff[:, sample] = np.linalg.norm(nonLinearNoisy - nonLinear, 
                                     axis = 1, 
                                     ord = 2); 

ut.plot2D(diff/T, epsilon);
plt.yscale('log');
plt.xlabel(r'$\epsilon$', fontsize = 15);
plt.ylabel(r'$\|\| y_i - x_i \|\|_2$', fontsize = 15);
