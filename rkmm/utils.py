
#----------------------------------------------------------------------------------------------------
# utils.py
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
from utils import *


def getMean(X):
    xDim, xSam = np.shape(X)
    mx = np.mean(X, axis = 1)
    
    return mx


def maxZero(M):
    nRow, nCol = np.shape(M)
    for i in range(nRow):
        for j in range(nCol):
            if M[i, j] < 0:
                M[i, j] = 0
                
    return M


def getStd(X, mx):
    xDim, xSam = np.shape(X)
    
    tmp = X - repVec(mx, xSam)
    tmp = tmp ** 2
    tmp = np.sum(tmp, axis=0)
    stdx = np.std(tmp)
    
    return stdx


def getKWidth(X):
    xDim, xSam = np.shape(X)
    D = eudist(X, X, False)
    assert np.isnan(D).any() == False, 'There exists nan number in array !'
    
    tm = []
    for i in range(xSam):
        for j in range(i+1, xSam):
            if D[i, j] < 0:
                tmp = 0
            else:
                tmp = np.sqrt(D[i, j])
            
            tm.append(tmp)
            
    med = np.median(tm)
    
    return med


def getProb(X, mx, stdx):
    xDim, xSam = np.shape(X)
    tmp = X - repVec(mx, xSam)
    tmp = tmp ** 2
    tmp = np.sum(tmp, axis=0)
    tmp = - tmp / stdx
    
    p = np.exp(tmp)
    
    return p


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
    for i in range(xSam):
        for j in range(xSam):
            if K[i, j] < 0:
                K[i, j] = 0
    
    nRow, nCol = np.shape(K)
    D = np.sqrt(K)
    for i in range(xSam):
        D[i, i] = 1e10
    
    index = np.argsort(D, axis=1)
    
    tmp = np.zeros((nRow, nCol))
    for i in range(xSam):
        for j in range(xSam):
            ind = index[i, j]
            tmp[i, j] = D[i, ind]
    
    d = tmp[:, k]
    del tmp
    T = np.zeros((xSam, xSam))
    
    for i in range(xSam):
        for j in range(xSam):
            T[i, j] = d[i] * d[j]
            if T[i, j] < 1e-7:
                T[i, j] = 1e-7
            
            D[i, j] = -D[i, j] / (2*T[i, j])
    
    K = np.exp(D)
    
    return K


def locKxy(X, Y, k):
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    
    nSam = xSam + ySam
    XY = np.column_stack((X, Y))
    
    D = eudist(XY, XY, False)
    for i in range(nSam):
        for j in range(nSam):
            if D[i, j] < 0:
                D[i, j] = 0
                
    Dist = np.sqrt(D)
    index = np.argsort(Dist, axis=1)
    
    tmp = np.zeros((nSam, nSam))
    for i in range(nSam):
        for j in range(nSam):
            ind = index[i, j]
            tmp[i, j] = D[i, ind]
        
    d = tmp[:, k]
    del D, Dist, tmp
    T = np.zeros((nSam, nSam))
    
    K = eudist(X, Y, False)
    for i in range(xSam):
        for j in range(ySam):
            T[i, j] = d[i] * d[xSam + j]
            if T[i, j] < 1e-7:
                T[i, j] = 1e-7
            
            K[i, j] = - K[i, j] / (2*T[i, j])
            
    K = np.exp(K)
    
    return K


def mlocKxr(X, rX, k):
    xDim, xSam = np.shape(X)
    rDim, rSam = np.shape(rX)
    
    nSam = xSam + rSam
    XY = np.column_stack((X, rX))
    
    D = eudist(XY, XY, False)
    Dist = np.sqrt(D)
    for i in range(xSam):
        D[i, i] = 1e10
    
    index = np.argsort(D, axis=1)
    
    tmp = np.zeros((nSam, nSam))
    for i in range(nSam):
        for j in range(nSam):
            ind = index[i, j]
            tmp[i, j] = D[i, ind]
        
    d = tmp[:, k]
    del D, Dist, tmp
    T = np.zeros((nSam, nSam))
    
    K = eudist(X, rX, False)
    for i in range(xSam):
        for j in range(rSam):
            T[i, j] = d[i] * d[xSam + j]
            K[i, j] = - K[i, j] / (2*T[i, j])
            
    K = np.exp(K)
    
    return K


def mlocKyr(X, rX, index, k):
    xDim, xSam = np.shape(X)
    rDim, rSam = np.shape(rX)
    
    nSam = xSam + rSam
    #XY = np.column_stack((X, rX))
    
    D = eudist(X, X, False)
    Dist = np.sqrt(D)
    for i in range(xSam):
        D[i, i] = 1e10
    
    ind = np.argsort(D, axis=1)
    
    d = np.zeros((nSam, 1))
    for i in range(nSam):
        if i < xSam:
            tid = ind[i, k]
            d[i] = D[i, tid]
            
        elif i >= xSam:
            tid = i - xSam
            idx = index[tid]
            tid = ind[idx, k]
            d[i] = D[idx, tid]
    
    T = np.zeros((nSam, nSam))
    
    K = eudist(X, rX, False)
    for i in range(xSam):
        for j in range(rSam):
            T[i, j] = d[i] * d[xSam + j]
            K[i, j] = - K[i, j] / (2*T[i, j])
            
    K = np.exp(K)
    
    
    return K


