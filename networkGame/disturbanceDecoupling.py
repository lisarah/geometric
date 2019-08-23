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
J = np.array([[2, 0, 0, 0], [0.5, 2, 0, 0], [1.5, 0, 2, 0],[0,1.5, -0.5, 2]])
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

# generate NE
T = 1000;
x0 = np.random.rand(N);
xHist = ut.runGradient(np.eye(N) - gamma*J, x0 = x0, T = 1000);
timeLine = np.linspace(0, T, T);
plt.figure();
for i in range(N):
    plt.plot(timeLine, xHist[i,:], label = str(i+1));
plt.grid();
plt.legend();
plt.show();

# generate noise NE
# input noise is at player 1
T = 1000; 
halfT = int(T/2);
noise = np.zeros((N,T))
noise[0,:] = 0.1*np.random.rand(T);
xHist = ut.runGradient(np.eye(N) - gamma*J,  x0 = x0, noise = noise, T = 1000);
timeLine = np.linspace(0, T, T);
plt.figure();
for i in range(N):
    plt.plot(timeLine, xHist[i,:], label = str(i+1));
plt.grid();
plt.legend();
plt.show();
