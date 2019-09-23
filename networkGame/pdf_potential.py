# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:19:50 2019

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
J[0:2, 2:4] = J[2:4, 0:2].T;
J[0:2, 4:6] = J[4:6, 0:2].T;
J[2:4, 6:8] = J[6:8, 2:4].T;
J[4:6, 6:8] = J[6:8, 4:6].T;
alpha, beta, gamma = ut.returnStepSize(J);

# generate Offset
c = np.zeros(8);
T = 1000;
x0 = np.random.rand(2*N);


#---------------------- noise in player 1 ----------------------------#
# generate NE in the noiseless case
linear = ut.runGradient(np.eye(2*N) - gamma*J, x0 = x0, T = T, offset=c);
timeLine = np.linspace(0, T, T);
ut.plot2D(linear);

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
    

ut.plot2D(diff/T, epsilon);
plt.xlabel(r'$\epsilon$', fontsize = 15);
plt.yscale('log');
plt.ylabel(r'$\|\| y_i - x_i \|\|_2$', fontsize = 15);

#---------------------- noise in player 4 ----------------------------#
# generate noise NE -- linear dynamics
# input noise is at player 4
noise = np.zeros((N*2,T))
diff4 = np.zeros((N*2, samples));
epsilon = np.zeros(samples);
for sample in range(samples):
    epsilon[sample] = sample*0.1;
    noise[6:8,:] = epsilon[sample]*np.random.rand(2,T);
    
    linearNoisy = ut.runGradient(np.eye(2*N) - gamma*J,  
                                 x0 = x0, 
                                 noise = noise, 
                                 T = T, offset=c);
    diff4[:, sample] = np.linalg.norm(linearNoisy - linear, axis = 1, ord = 2);
    

ut.plot2D(diff4/T, epsilon);
plt.xlabel(r'$\epsilon$', fontsize = 15);
plt.yscale('log');
plt.ylabel(r'$\|\| y_i - x_i \|\|_2$', fontsize = 15);




