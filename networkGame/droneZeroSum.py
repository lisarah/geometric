# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:06:34 2019

@author: craba
"""
import numpy as np
import util as ut
import networkx as nx
import matplotlib.pyplot as plt
plt.close('all');
N = 4;
drawGraph = False;
J = np.zeros((8,8));
J[2:4, 0:2] = np.random.rand(2,2);
J[4:6, 0:2] = np.random.rand(2,2);
J[6:8, 2:4] = np.random.rand(2,2);
J[6:8, 4:6] = np.random.rand(2,2);
J[0:2, 2:4] = -J[2:4, 0:2].T;
J[0:2, 4:6] = -J[4:6, 0:2].T;
J[2:4, 6:8] = -J[6:8, 2:4].T;
J[4:6, 6:8] = -J[6:8, 4:6].T;
#J = np.array([[2, 0, 0, 0], [1.5, 2, 0, 0], [1.5, 0, 2, 0],[0,0.5, -0.5, 2]])
alpha, beta, gamma = ut.returnStepSize(J);
#gamma = gamma/10;
noiseMag = 0.5;
# generate a communication graph
G = nx.DiGraph();
for i in range(1,N+1):
    for j in range(1, N+1):
        if i != j and J[i*2-1,j*2-1] != 0:
            G.add_edge(j,i);
if drawGraph:
    plt.figure();
    nx.draw_circular(G);
    nx.draw_networkx_nodes(G,pos=nx.circular_layout(G), node_size=200)
    nx.draw_networkx_labels(G,pos=nx.circular_layout(G),font_size=15, font_color='w' )
    nx.draw_networkx_edges(G,pos=nx.circular_layout(G),arrows =True)
    plt.show();
# generate Offset
c = np.zeros(8);

def nonLinearGrad(xt):
    Df = np.zeros((8));
    Df[0:2] = 2*(xt[0:2] - c[0:2]) + 1.5*(xt[2:4] + xt[4:6]) + 0.05;
    
    Df[2:4] = 2*(xt[2:4] - c[2:4]) + 1.5*xt[0:2] + 0.05*xt[0:2].dot(xt[0:2])*xt[0:2];
    
    Df[4:6] = 2*(xt[4:6] - c[4:6]) + 1.5*xt[0:2] + 0.05*xt[0:2].dot(xt[0:2])*xt[0:2];
    Df[6:8] = 2*(xt[6:8] - c[6:8]) + 0.5*(xt[2:4] - xt[4:6]) + 0.01*(xt[2:4].dot(xt[2:4]) - xt[4:6].dot(xt[4:6]));
    return xt - gamma*Df;


# noise in player 4
T = 100;
x0 = np.random.rand(2*N);
linear = ut.runGradient(np.eye(2*N) - gamma*J, x0 = x0, T = T, offset=c);
timeLine = np.linspace(0, T, T);
ut.plot2D(linear);

# generate noise NE -- linear dynamics
# input noise is at player 1
T = 100; 
halfT = int(T/2);
noise = np.zeros((N*2,T))
noise[6:8,:] = noiseMag*np.random.rand(2,T);
linearNoisy = ut.runGradient(np.eye(2*N) - gamma*J,  x0 = x0, noise = noise, T = T, offset=c);
ut.plot2D(linearNoisy)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
plt.figure();
for i in range(N):
    plt.plot(timeLine, 
             np.linalg.norm(linear[2*i:2*(i+1), :] - linearNoisy[2*i:2*(i+1)], ord = 2, axis=0), 
             label = str(i+1),
             color = colors[i]);
plt.grid();
plt.legend();

plt.yscale('log')
plt.show();


## noise in player 1
#T = 100;
#x0 = np.random.rand(2*N);
#linear = ut.runGradient(np.eye(2*N) - gamma*J, x0 = x0, T = T, offset=c);
#timeLine = np.linspace(0, T, T);
#ut.plot2D(linear);
#
## generate noise NE -- linear dynamics
## input noise is at player 1
#T = 100; 
#halfT = int(T/2);
#noise = np.zeros((N*2,T))
#noise[6:8,:] = noiseMag*np.random.rand(2,T);
#linearNoisy = ut.runGradient(np.eye(2*N) - gamma*J,  x0 = x0, noise = noise, T = T, offset=c);
#ut.plot2D(linearNoisy)
#
#colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
#plt.figure();
#for i in range(N):
#    plt.plot(timeLine, 
#             np.linalg.norm(linear[2*i:2*(i+1), :] - linearNoisy[2*i:2*(i+1)], ord = 2, axis=0), 
#             label = str(i+1),
#             color = colors[i]);
#plt.grid();
#plt.legend();
#
#plt.yscale('log')
#plt.show();
