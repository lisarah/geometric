# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:15:43 2019

@author: Sarah Li
"""
import subspace_linalg as subspace


def ISA(V0, A, B, eps = 0.0):
    K = V0;
    Vt= 1.0*V0;
    isInvariant = False;
    iteration = 0;
    while not isInvariant:
        iteration += 1;
        print ("--------------in interation ", iteration);
        # print (f'vt = {Vt}')
        # print (sumS(Vt, B))
        Ainved =  subspace.a_inv_v(A, subspace.sum(Vt, B));
#        print (Ainved) 
        Vnext = subspace.intersect(K, Ainved);
        print ("rank of Vt ", subspace.rank(Vt));
        print ("rank of Vnext ", subspace.rank(Vnext));
        print ("Vnext is contained in Vt ", subspace.contained(Vnext, Vt))
        if subspace.rank(Vnext) == subspace.rank(Vt):
            isInvariant = True;
            print ("ISA returns-----")
            # print (Vnext);
            # print (Vt);
            
        else:
            Vt = Vnext;
        
    return Vt

  
# np.random.seed(122323)
# N = 10
# M = 3
# observed_player = 0
# disturbed_player = 1
# A = np.random.rand(N, N)
# B = np.random.rand(N, M)
# H = np.zeros((N, 1))
# H[observed_player,0] = 1.

# V = (ISA(ker(H.T), A, B))
# VB = np.concatenate((V, B),axis=1)
# # print(invariantSub)

# # do optimization
# F = cvx.Variable((M, N))
# # P = cvx.Variable((N,N),PSD = True)
# XS = cvx.Variable((9 + 3, 9))
# constraint = [VB@XS == A.dot(V)]
# constraint.append(XS[9:, :] == -F@V)
# # constraint.append(A.T*P + P*A + (B*F).H*P + P*B*F << 0)
# # obj = cvx.Minimize(cvx.norm2(F))# 
# # obj = cvx.Minimize(cvx.norm2(A.T*P + P*A + F.H * B.T *P + P*B*F) )
# obj = cvx.Minimize(cvx.norm2(A + B*F))
# dd = cvx.Problem(obj, constraint)
# F_norm = dd.solve(solver=cvx.MOSEK, verbose=True) #  solver=cvx.SCS,
# F_opt = F.value
# print(f'resuling controller: \n{np.round(A+B.dot(F_opt),1)}')


# E = np.zeros((N,1))
# E[disturbed_player] = 1
# print(f'blue is the disturbance decoupled player')
# run_noisy_dynamics(A,B,E,F_opt,int(4))