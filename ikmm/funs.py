
#----------------------------------------------------------------------------------------------------
# together.py
# This file contains the definition of some essential calculation methods. 
#
# Coded by Miao Cheng
#
# All Rights Reserved.
#----------------------------------------------------------------------------------------------------

import numpy as np
from cala import *
from numpy import linalg as la
from cala import *


def getMean(X):
    xDim, xSam = np.shape(X)
    mx = np.mean(X, axis = 1)
    
    return mx


def ginv(U, V, s, r):
    assert r > 0, 'The desired rank data is incorrect !'
    
    tmp = s[0:r]
    tmp = tmp ** (-1)
    tmp = np.diag(tmp)
    
    V = V[:, 0:r]
    U = U[:, 0:r]
    
    tmv = np.dot(V, tmp)
    minv = np.dot(tmv, np.transpose(U))
    
    return minv


def locKxx(X, k):
    xDim, xSam = np.shape(X)
    
    K = eudist(X, X, False)
    nRow, nCol = np.shape(K)
    D = np.sqrt(K)
    index = np.argsort(D, axis=1)
    
    tmp = np.zeros((nRow, nCol))
    for i in range(nRow):
        ind = index[i, :]
        tmp[i, :] = D[i, ind]
    
    d = tmp[:, k]
    del tmp
    T = np.zeros((xSam, xSam))
    
    for i in range(xSam):
        for j in range(xSam):
            T[i, j] = d[i] * d[j]
            D[i, j] = -D[i, j] / (2*T[i, j])
    
    K = np.exp(D)
    
    return K


def locKxy(X, Y, k):
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    
    nSam = xSam + ySam
    XY = np.column_stack((X, Y))
    
    D = eudist(XY, XY, False)
    Dist = np.sqrt(D)
    index = np.argsort(Dist, axis=1)
    
    tmp = np.zeros((nSam, nSam))
    for i in range(nSam):
        ind = index[i, :]
        tmp[i, :] = D[i, ind]
        
    d = tmp[:, k]
    del D, Dist, tmp
    T = np.zeros((nSam, nSam))
    
    K = eudist(X, Y, False)
    for i in range(xSam):
        for j in range(ySam):
            T[i, j] = d[i] * d[xSam + j]
            K[i, j] = - K[i, j] / (2*T[i, j])
            
    K = np.exp(K)
    
    return K




            


