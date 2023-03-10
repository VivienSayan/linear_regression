#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 20:21:09 2021

@author: VivienSayan
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x,a,b):
    return a*x + b

def J(x,y,a,b):
    m = x.size
    som = 0
    for i in range(1,m):
        som += (a*x[i]+b-y[i])**2
    res = som/(2*m)
    return res  

def dJda(x,y,a,b):
    m = x.size
    som = 0
    for i in range(1,m):
        som += x[i]*(a*x[i]+b-y[i])
    res = som/m
    return res

def dJdb(x,y,a,b):
    m = x.size
    som = 0
    for i in range(1,m):
        som += a*x[i]+b-y[i]
    res = som/m
    return res


def gradient_descent(alpha,eps,X,Y,a,b):
    xvec = np.linspace(0,10,100) 
    yvec = np.linspace(0,10,100) 
    
    print("Cost function BEFORE gradient descent:")
    print("J(a={},b={}) = {}".format(a,b,J(X,Y,a,b)))
    print()    
    plt.figure(1)
    plt.plot(X,Y,"g.")
    plt.plot(xvec,f(xvec,a,b),"r")
    plt.xlabel("x")
    plt.ylabel("y")
    
    
    while J(X,Y,a,b) > eps:
        a = a - alpha * dJda(X,Y,a,b)
        b = b - alpha * dJdb(X,Y,a,b)
        
    
    print()
    print("Cost function AFTER gradient descent:")
    print("J(a={},b={}) = {}".format(a,b,J(X,Y,a,b)))    
    plt.figure(2)
    plt.plot(X,Y,"g.")
    plt.plot(xvec,f(xvec,a,b),"b")
    plt.xlabel("x")
    plt.ylabel("y")
    
  

def least_square(z,phi):
    print("z = phi*theta")
    print("=> theta = pseudo_inv(phi)*z")
    phiT = np.transpose(phi)
    phiTphi = np.matmul(phiT,phi)
    pinv = np.matmul(np.linalg.inv(phiTphi), phiT)

    print("theta = ",np.matmul(pinv,z))
    
    
    

#---------------- MAIN --------------
X = np.array([1,2,3,4,5])
Y = np.array([1*2+1,2*2+1,3*2+1,4*2+1,5*2+1])

a = 3.
b = 1.

alpha = 0.1
eps = 0.0001

gradient_descent(alpha,eps,X,Y,a,b)

print()
print("--------- least square method solution ----------")
print()

phi = np.array([(1,1),
                (2,1),
                (3,1),
                (4,1),
                (5,1)])

least_square(Y,phi)





