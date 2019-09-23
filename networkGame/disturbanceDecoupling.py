# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:10:11 2019

@author: craba
"""

import numpy as np
import util as ut
import networkx as nx
import matplotlib.pyplot as plt
plt.close('all');
N = 4;
drawGraph = False;
J = np.array([[2, 0, 0, 0], [1.5, 2, 0, 0], [1.5, 0, 2, 0],[0,0.5, -0.5, 2]])
alpha, beta, gamma = ut.returnStepSize(J);
# generate a communication graph
G = nx.DiGraph();
for i in range(1,N+1):
    for j in range(1, N+1):
        if i != j and J[i-1,j-1] != 0:
            G.add_edge(j,i);
if drawGraph:
    plt.figure();
    nx.draw_circular(G);
    nx.draw_networkx_nodes(G,pos=nx.circular_layout(G), node_size=200)
    nx.draw_networkx_labels(G,pos=nx.circular_layout(G),font_size=15, font_color='w' )
    nx.draw_networkx_edges(G,pos=nx.circular_layout(G),arrows =True)
    plt.show();
# generate Offset
c = np.array([1,2, 3,4])*gamma;

def nonLinearGrad(xt):
    Df = np.zeros((4));
    Df[0] = 2*(xt[0] - c[0]);
    Df[1] = 2*(xt[1] - c[1]) + 1.5*xt[0] + 0.05*np.power(xt[0],3);
    Df[2] = 2*(xt[2] - c[2]) + 1.5*xt[0] + 0.05*np.power(xt[0],3);
    Df[3] = 2*(xt[3] - c[3]) + 0.5*(xt[1] - xt[2]) + 0.1*(np.power(xt[1],2) - np.power(xt[2],2));

    return xt - gamma*Df;


# generate NE
T = 100;
x0 = np.random.rand(N);
xHist = ut.runGradient(np.eye(N) - gamma*J, x0 = x0, T = T, offset=c);
timeLine = np.linspace(0, T, T);
plt.figure();
for i in range(N):
    plt.plot(timeLine, xHist[i,:], label = str(i+1));
plt.grid();
plt.legend();
plt.show();



# generate noise NE -- linear dynamics
# input noise is at player 1
T = 100; 
halfT = int(T/2);
noise = np.zeros((N,T))
noise[0,:] = 0.5*np.random.rand(T);
xHist = ut.runGradient(np.eye(N) - gamma*J,  x0 = x0, noise = noise, T = T, offset=c);
timeLine = np.linspace(0, T, T);
plt.figure();
for i in range(N):
    plt.plot(timeLine, xHist[i,:], label = str(i+1));
plt.grid();
plt.legend();
plt.show();

#----------- non linear version of everything that happened just now -------#
# generate NE
T = 100;
x0 = np.random.rand(N);
xHist = ut.runNonlinearGradient(nonLinearGrad, x0 = x0, T = T);
timeLine = np.linspace(0, T, T);
plt.figure();
for i in range(N):
    plt.plot(timeLine, xHist[i,:], label = str(i+1));
plt.grid();
plt.legend();
plt.show();
# input noise is at player 1
T = T; 
halfT = int(T/2);
noise = np.zeros((N,T))
noise[0,:] = 0.1*np.random.rand(T);
xHist = ut.runNonlinearGradient(nonLinearGrad,  x0 = x0, noise = noise, T = T);
timeLine = np.linspace(0, T, T);
plt.figure();
for i in range(N):
    plt.plot(timeLine, xHist[i,:], label = str(i+1));
plt.grid();
plt.legend();
plt.show();