#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 21:14:29 2019

@author: sarahli
"""

import numpy as np

def ssp(sources, sinks, incidence, capacity, cost):
    V, E = incidence.shape;

    bVec = np.zeros((V));
    B = 1;
    flowVec = np.zeros((E));
    while B > 0:
        s = np.where(bVec == 1)[0];
        t = np.where(bVec == -1)[0];
        resG = residualGraph(flowVec);
        path, minCost = dijkstra(resG, s, t, cost);
        augment(s,t,path, flowVec, B);
    
    return flowVec
        
def residualGraph(capacity, flowVec, incidence):
    resGraph = 1.0*capacity;
    for i in range(len(flowVec)):
        if flowVec[i] != 0:
            source = np.where(incidence[:,i] == 1);
            sink = np.where(incidence[:,i] == -1);
            resGraph[source,sink] += -flowVec[i];
            resGraph[sink,source] += flowVec[i];
            
    return resGraph;
def selectMinDistance(queue, length):
    minInd = queue[0]; 
    minLen = length[queue[0]];
    for nodeInd in queue:
        if length[nodeInd] < minLen:
            minInd = nodeInd;
            minLen = length[nodeInd];
    return minInd, minLen;

def dijkstra(G,s,t, cost):
    V,E = G.shape;
    length = np.zeros((V));
    visited =[False]*V;
    prevNode = np.zeros((V));
    length[:] = np.nan;
    length[s] = 0;
    prevNode[:] = np.nan;
    prevNode[s] = s;
    queue = [];
    queue.append(s);
    while len(queue) > 0:
        nextNode, nextNodeDist = selectMinDistance(queue);
        visited[nextNode] = True;
        for n in G.neighbors(nextNode):
            if length[n] > nextNodeDist + cost[nextNode, n]:
                length[n] = nextNodeDist + cost[nextNode,n];
                prevNode[n] = nextNode;
            if visited[n] == False:
                queue.append(n);
   
    path = [t];
    
    while path[-1] != s:
        curNode = path[-1];
        path.append(prevNode[curNode]);
    
    path.reverse();
    return path, minCost;
        
        
def augment(s,t,path, flow, B):
    B = B -np.min([])
    print ("Nothing written for augemntation");
    
    
    
    
    
    
    
    
    
    