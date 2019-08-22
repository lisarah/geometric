# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:29:43 2019

@author: craba
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import networkx as nx
plt.close('all')

def droneDyn(graph, couplingCoeff, N, n):
    # graph: communication adjacency matrix
    # coupling coefficient: coefficient that the coupling matrices are multiplied by
    # N: number of agents
    # n: state space
    # returns a N*n x N*n matrix DW for quadratic games
    DW = np.eye(n*N);
    A = np.eye(n,n);
    B = couplingCoeff*np.eye(n,n);
    # Complete graph
    for i in range(N):
        DW[n*i:n*(i+1), n*i:n*(i+1)] = (A + A.T); 
        for j in range(N):
            if graph[i,j] != 0:    
                DW[n*i:n*(i+1), n*i:n*(i+1)] += -graph[i,j]*(B + B.T); 
                DW[n*i:n*(i+1), n*j:n*(1+j)] = B+B.T;

    # step sizes
    w, eigVecs = np.linalg.eig(0.25*(DW + DW.T).T.dot(DW + DW.T));
    alpha = np.min(w);
    w2, eigVecs = np.linalg.eig(DW.T.dot(DW));
    beta = np.max(w2);
    gamma = np.eye(n*N)*alpha/beta; 
    if alpha == beta:
        gamma = gamma*couplingCoeff;
    L= np.eye(n*N) - gamma.dot(DW);
    wL, eigL = np.linalg.eig(L);
    print (wL);
    print ("Beta: ", beta)
    print ("Alpha: ", alpha);
    return L, gamma;

# drone cost functions
N = 5; # number of drones
n = 2; # state space size of each drone
T = 1000;
noiseBound = 0.7;
A = np.eye(n,n); 
ref = np.random.rand(n*N)*5.;

noise = np.random.rand(n,T)*noiseBound - noiseBound/2.;
#---------- game graph ------------------
k4Graph= np.tril(np.ones((N,N)), k = -1);

print ("K4 graph");
k4L, k4Gamma = droneDyn(k4Graph, 0.18, N,n);
G = nx.DiGraph();
#G.add_nodes_from(np.linspace(1,N,num=N));
for i in range(N):
    for j in range(N):
        if i == j:
            G.add_edge(i,j);
        elif k4Graph[i,j] == 1:
            G.add_edge(i,j);

plt.figure();
nx.draw_circular(G);
nx.draw_networkx_nodes(G,pos=nx.circular_layout(G), node_size=200)
nx.draw_networkx_labels(G,pos=nx.circular_layout(G),font_size=15, font_color='w' )
nx.draw_networkx_edges(G,pos=nx.circular_layout(G),arrows =True)
plt.show();
diagA = np.zeros((n*N, n*N));
for i in range(N):
    diagA[i*n:(i+1)*n, i*n:(i+1)*n] = A + A.T;

#--------------propogating dynamics -----------------#
subStep  = 1;
x = np.zeros((n*N,T*subStep),dtype=np.complex_);
k4x =  np.zeros((n*N, T*subStep), dtype=np.complex_);
noisyx = np.zeros((n*N, T*subStep),dtype=np.complex_);


for t in range(T*subStep):
    if t == 0:
        noisyx[:,0] = np.random.rand(n*N)*3.;
        k4x[:,0] = 1.0*x[:,0];

    else: 
        k4x[:,t] = k4L.dot(k4x[:,t-1]);
        noisyx[:,t] = k4L.dot(noisyx[:,t-1]);
        k4x[:,t] += k4Gamma.dot(diagA).dot(ref);
        noisyx[:,t] += k4Gamma.dot(diagA).dot(ref);


    # noise
    noisyx[4:4+n, t] += noise[:,t];
plt.figure();
ax = plt.axes(projection='3d')
# Data for a three-dimensional line
zline = np.linspace(0, T*subStep, T*subStep, endpoint = False)

for i in range(N):
    ax.scatter(k4x[i*n,:].real, k4x[i*n+1,:].real, zline, '--', label = str(i));
    ax.plot3D(noisyx[i*n,:].real,  noisyx[n*i+1,:].real, zline, '--'); #,label ='noisy '+str(i+1)
plt.legend();
plt.show();


# plot the distance of noise affected point to the NE
plt.figure();
for i in range(N):
    plt.plot(np.sqrt(np.abs(noisyx[i*n, :] - k4x[i*n,:])**2 + np.abs(noisyx[i*n+1, :] - k4x[i*n+1,:])**2),
             label = str(i));
plt.xscale('log')
plt.ylabel('log($\||x_n - x^\star {\||}_2)$')
plt.xlabel('Iterations')
plt.grid();
plt.legend();
plt.show();
