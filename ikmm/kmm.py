
#----------------------------------------------------------------------------------------------------
'''
kmm.py
This file contains the definition of related functions for kernal mean matching 

Coded by Miao Cheng
Date: 2018-11-25

All Rights Reserved.

'''
#----------------------------------------------------------------------------------------------------

import numpy as np
import random
import scipy.linalg as la
from datetime import *
from cala import *
from kernel import *
from nmse import *


def updMean(X, mx, Y):
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    assert xDim == yDim, 'The dimensionality of X and Y are not identical !'
    
    n = xSam + ySam
    
    for i in range(xDim):
        mx[i] = mx[i] * xSam
        
        for j in range(ySam):
            mx[i] = mx[i] + Y[i][j]
            
        mx[i] = mx[i] / n
        
    return mx


def updY(X, tX):
    xDim, xSam = np.shape(X)
    tDim, tSam = np.shape(Y)
    assert xDim == tDim, 'The dimensionality of X and tX are not identical !'
    
    n = xSam + tSam
    
    Y = np.column_stack((X, tX))
    
    return Y


def getAind(X, n):
    xDim, xSam = np.shape(X)
    
    tmk = xyK(X, X, 'Sinc')
    tm = np.sum(tmk, axis=0)
    
    assert len(tm) == xSam, 'The direction of operation may be incorrect !'
    idx = np.argsort(tm)
    ix = idx[0:n]
    
    return ix
    

def getBind(X, n, rn):
    xDim, xSam = np.shape(X)
    
    index = np.arange(xSam)
    random.shuffle(index)
    
    ind = index[0:rn]
    tX = X[:, ind]
    tmk = xyK(tX, X, 'Sinc')
    tm = np.sum(tmk, axis=0)
    
    assert len(tm) == xSam, 'The direction of operation may be incorrect !'
    idx = np.argsort(tm)
    ix = idx[0:n]
    
    return ix


def ginv(M):
    mRow, mCol = np.shape(M)
    U, s, V = la.svd(M)
    V = np.transpose(V)
    s, r = getRank(s)
    U = U[:, 0:r]
    V = V[:, 0:r]
    s = s[0:r]
    S = np.diag(s)
    
    tmp = np.dot(V, S)
    tmq = np.transpose(U)
    tm = np.dot(tmp, tmq)
    
    return tm


def getWeight(X, ind):
    xDim, xSam = np.shape(X)
    #tDim, tSam = np.shape(tX)
    #assert xDim == tDim, 'The dimensionality of X and tX are not identical !'
    
    mx = np.mean(X, axis=1)
    mw = np.zeros((xSam, 1))
    for i in range(xSam):
        tmp = X[:, i] - mx
        tmp = tmp * tmp
        tmp = np.sum(tmp)
        tmp = np.exp(-tmp)
        mw[i, 0] = tmp
    
    tmw = mw[ind, 0]
    sw = np.sum(mw)
    stw = np.sum(tmw)
    weight = float(stw) / sw
    
    return weight

    
# +++++ The kmm functions +++++
def setLayer(b, P, k):
    bDep, bRow, bCol = np.shape(b)
    pRow, pCol = np.shape(P)
    assert bRow == pRow, 'The dimensionality of b and P are not identical !'
    assert bCol == pCol, 'The dimensionality of b and P are not identical !'
    
    for i in range(pRow):
        for j in range(pCol):
            b[k, i, j] = P[i, j]
            
    return b


def together(b):
    bDep, bRow, bCol = np.shape(b)
    assert bDep > 1, 'The depth of b is incorrect !'
    
    m = np.zeros((bRow, bCol))
    for i in range(bRow):
        for j in range(bCol):
            for k in range(bDep):
                m[i, j] = m[i, j] + b[k, i, j]
                
                
    return m

def iTogether(B):
    bDep, bRow, bCol = np.shape(B)
    assert bDep >= 1, 'The depth of b is incorrect !'
    
    sKxx = xysK(self.__X, self.__X, 'Gaussian', self.__kw)
    sKxy = xysK(self.__X, self.__Y, 'Gaussian', self.__kw) 
    
    P = np.zeros((bDep, bDep))
    q = np.zeros((bDep, 1))
    for i in range(bDep):
        tmb = B[i, :, :]
        tmp = np.dot(np.transpose(tmb), sKxx)
        tmp = np.dot(tmp, tmb)
        tmq = np.sum(np.sum(tmp))
        tm = 1 / 2
        P[i, i] = tm * tmq
        
        tmp = np.dot(np.transpose(tmb), sKxy)
        tmq = np.sum(np.sum(tmp))
        tm = self.__xSam / self.__ySam
        q[i] = tm * tmq
        
    # +++++ quadprog +++++
    q = q.reshape((bDep, ))
    G = np.zeros((bDep, bDep))
    for i in range(bDep):
        G[i, i] = -1
    
    h = np.zeros((bDep, 1)).reshape((bDep, ))
    A = None
    b = None
    eff = solve_qp(P, q, G, h, A, b)
    
    
    # +++++ cvxopt +++++
    #G = np.zeros((bDep, bDep))
    #for i in range(bDep):
        #G[i, i] = -1
        
        
    #h = np.zeros((bDep, 1))
    #M = np.ones((1, bDep))
    #N = 1
    
    #P = matrix(P, tc='d')
    #q = matrix(q, tc='d')
    #G = matrix(G, tc='d')
    #h = matrix(h, tc='d')
    #M = None
    #N = None
    
    #sol = solvers.qp(P, q, G, h, M, N)
    #eff = np.array(sol['x'])
    
    # +++++ Calculate the final matrix +++++
    m = np.zeros((bRow, bCol))
    for i in range(bDep):
        tmp = eff[i] * B[i, :, :]
        m = m + tmp
    
    return m

# +++++ global kmm +++++
def glokmm(X, Y, n):
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    assert xDim == yDim, 'The dimensionality of X and Y are not identical !'
    
    sKxx = xyK(X, X, 'Sinc')
    
    inv = ginv(sKxx)
    inv = inv * 0.5
    
    ind = getAind(Y, n)
    tY = Y[:, ind]
    
    tmk = xyK(X, tY, 'Sinc')
    P = np.dot(inv, tmk)
    
    trs = float(n) / ySam
    P = P * trs
    
    weight = getWeight(Y, ind)
    P = P * weight
    
    return P


def iglokmm(X, Y, n):
    P = glokmm(X, Y, n)
    tmp = inmse(X, Y, P)
    
    return tmp


def tglokmm(X, Y, cY, n, nSam):
    yDim, ySam = np.shape(X)
    cDim, cSam = np.shape(cY)
    assert yDim == cDim, 'The dimensionality of Y and cY are not identical !'
    
    n = int(np.floor(cSam / nSam))
    nmse = np.zeros((n, 1))
    cost = np.zeros((n, 1))
    tmy = Y
    for i in range(n):
        tY = cY[:, i*nSam:(i+1)*nSam]
        tmy = np.column_stack((tmy, tY))
        
        oldtime = datetime.now()
        tmp = iglokmm(X, tmy, n)
        newtime = datetime.now()
        tmq = (newtime - oldtime).seconds
        nmse[i] = tmp
        cost[i] = tmq
        
        ch = str(i) + '-th slice: ' + str(tmp)
        th = str(i) + '-th cost time:' + str(tmq)
        print(ch)
        print(th)
        
    return nmse
    
    
# +++++ skmm +++++
def skmm(X, Y, n, rn):       # skmm(X, Y, n, rn, k)
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    assert xDim == yDim, 'The dimensionality of X and Y are not identical !'
    
    sKxx = xyK(X, X, 'Sinc')
    
    inv = ginv(sKxx)
    inv = inv * 0.5
    
    ind = getBind(Y, n, rn)
    tY = Y[:, ind]
    
    tmk = xyK(X, tY, 'Sinc')
    P = np.dot(inv, tmk)
    
    trs = float(n) / ySam
    P = P * trs
    
    weight = getWeight(Y, ind)
    P = P * weight
    
    return P


def iskmm(X, Y, n, rn, times):       # iskmm(X, Y, n, rn, k, times)
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    assert xDim == yDim, 'The dimensionality of X and Y are not identical !'
    
    b = np.zeros((times, xSam, n))
    for i in range(times):
        ch = str(i) + '-th running'
        print(ch)
        
        P = skmm(X, Y, n, rn)
        setLayer(b, P, i)
    
    m = together(b)
    tmp = inmse(X, Y, m)
    
    return tmp

# +++++ Temporal skmm +++++
def tskmm(X, Y, tY, n, rn, times):
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    assert xDim == yDim, 'The dimensionality of X and Y are not identical !'
    
    Y = np.column_stack((Y, tY))
    b = np.zeros((times, xSam, n))
    for i in range(times):
        #ch = str(i) + '-th running'
        #print(ch)
        
        P = skmm(X, Y, n, rn)
        setLayer(b, P, i)
        
    #m = together(b)
    m = iTogether(b)
    tmp = inmse(X, Y, m)
    
    return tmp


def itskmm(X, Y, cY, n, rn, times, nSam):
    yDim, ySam = np.shape(X)
    cDim, cSam = np.shape(cY)
    assert yDim == cDim, 'The dimensionality of Y and cY are not identical !'
    
    n = int(np.floor(cSam / nSam))
    nmse = np.zeros((n, 1))
    cost = np.zeros((n, 1))
    for i in range(n):
        tY = cY[:, i*nSam:(i+1)*nSam-1]
        
        oldtime = datetime.now()
        tmp = tskmm(X, Y, tY, n, rn, times)
        newtime = datetime.now()
        tmq = (newtime - oldtime).seconds        
        nmse[i] = tmp
        cost[i] = tmq
        
        ch = str(i) + '-th slice: ' + str(tmp)
        th = str(i) + '-th cost time:' + str(tmq)
        print(ch)
        print(th)        
        
    return nmse
     
    
    
    