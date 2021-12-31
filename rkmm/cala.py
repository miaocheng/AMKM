
#-----------------------------------------------------------------------------------------
# Name:        cala.py
# This python file contains several implementation of essential functions 
# for data analysis.
# Author:      Miao Cheng
#
# Created Date: 2017-9-19
# Last Modified Date: 2018-2-4
# Copyright:   (c) Miao Cheng 2017
# Licence:     <your licence>
#-----------------------------------------------------------------------------------------

import numpy as np
#from __future__ import *

from datetime import datetime
from dateutil.relativedelta import relativedelta


def mstd(s):
    nLen = np.shape(s)
    mx = np.sum(s) / nLen
    
    tmp = s - mx
    tmp = sum(tmp * tmp)
    tm = tmp / nLen
    stdx = np.sqrt(tm)
    
    return mx, stdx
    

def meanX(X):
    xDim, xSam = np.shape(X)
    mx = np.mean(X, axis = 1)
    #tmp = np.tile(mx, (xSam, 1))
    #X = X - np.transpose(tmp)
    
    for i in range(xSam):
        tmp = X[:, i] - mx
        X[:, i] = tmp
    
    return X, mx


def iMean(X, mx):
    xDim, xSam = np.shape(X)
    nDim = len(mx)
    assert xDim == nDim, 'The feature dimension is not identifical !'
    
    for i in range(xSam):
        tmp = X[:, i] - mx
        X[:, i] = tmp
        
    return X


def meanX2D(X):
    xRow, xCol, xSam = np.shape(X)
    mx = np.zeros((xRow, xCol))
    
    for i in range(xSam):
        mx += X[:, :, i]
        
    mx = mx / xSam
    
    #for i in range(xSam):
        #X[:, :, i] = X[:, :, i] - mx
        
    X = subMX(X, mx)
    
    return X, mx


def subMX(X, mx):
    xRow, xCol, xSam = np.shape(X)
    
    for i in range(xSam):
        tmp = X[:, :, i]
        tmp -= mx
        X[:, :, i] = tmp
        
    return X


def prePro(X, mx):
    xDim, xSam = np.shape(X)
    X = X - repVec(mx, xSam)
    
    X = justNorm(X)
    
    return X


def justNorm(X):
    xDim, xSam = np.shape(X)
    
    tmp = X * X
    d = np.sum(tmp, axis=0)
    d = np.sqrt(d)
    tmp = np.tile(d, (xDim, 1))
    X = X / tmp
    
    return X
    
    
def normX(X):
    xDim, xSam = np.shape(X)
    mx = np.mean(X, axis = 1)
    tmp = np.tile(mx, (xSam, 1))
    X = X - np.transpose(tmp)
    
    tmp = X * X
    d = np.sum(tmp, axis=0)
    d = np.sqrt(d)
    tmp = np.tile(d, (xDim, 1))
    X = X / tmp
    
    return X


def lprj2D(V, X):
    vRow, vCol = np.shape(V)
    if X.ndim == 2:
        xRow, xCol = np.shape(X)
        xSam = 1
    elif X.ndim == 3:
        xRow, xCol, xSam = np.shape(X)
        
    if xSam == 1:
        PX = np.dot(np.transpose(V), X)
    elif xSam > 1:
        PX = np.zeros((vCol, xCol, xSam))
        for i in range(xSam):
            tmp = X[:, :, i]
            tmp = np.dot(np.transpose(V), tmp)
            PX[:, :, i] = tmp
            
    return PX


def rprj2D(V, X):
    vRow, vCol = np.shape(V)
    if X.ndim == 2:
        xRow, xCol = np.shape(X)
        xSam = 1
    elif X.ndim == 3:
        xRow, xCol, xSam = np.shape(X)
        
    #PX = np.zeros((vCol, xCol, xSam))
    
    if xSam == 1:
        PX = np.dot(X, V)
    elif xSam > 1:
        PX = np.zeros((xRow, vCol, xSam))
        for i in range(xSam):
            tmp = X[:, :, i]
            tmp = np.dot(tmp, V)
            PX[:, :, i] = tmp
        
    return PX


def dist(x, y, tosqrt):
    x = np.array(x)
    y = np.array(y)
    
    if x.ndim == 1:
        xDim = len(x)
        x = np.reshape(x, (xDim, 1))
        
    if y.ndim == 1:
        yDim = len(y)
        y = np.reshape(y, (yDim, 1))
        
    xx = np.sum(x*x, axis=0)
    yy = np.sum(y*y, axis=0)
    xy = np.dot(np.transpose(x), y)
    
    d = xx + yy - 2*xy
    
    if tosqrt == True:
        d = np.sqrt(d)
    
    return d


def dist2D(x, y, tosqrt):
    xRow, xCol = np.shape(x)
    yRow, yCol = np.shape(y)
    
    if (xRow != yRow) or (xCol != yCol):
        AssertionError('The dimensionality of data does not match !')
    
    xx = x*x
    yy = y*y
    xy = x*y
    
    tmp = xx + yy - 2*xy
    
    d = 0
    for i in range(xRow):
        for j in range(xCol):
            d += tmp[i, j]
            
    if tosqrt == True:
        d = np.sqrt(d)
        
    return d


def eudist(X, Y, tosqrt):
    X = np.array(X)
    Y = np.array(Y)
    
    if X.ndim == 1:
        xDim = len(X)
        X = np.reshape(X, (xDim, 1))
    
    if Y.ndim == 1:
        yDim = len(Y)
        Y = np.reshape(Y, (yDim, 1))
    
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    
    XX = X * X
    YY = Y * Y
    sX = np.sum(XX, axis=0)
    sY = np.sum(YY, axis=0)
    
    tX = np.tile(sX, (ySam, 1))
    mX = np.transpose(tX)
    mY = np.tile(sY, (xSam, 1))
    XY = np.dot(np.transpose(X), Y)
    
    D = mX + mY - 2 * XY
    
    if tosqrt == True:
        D = np.sqrt(D)
    
    return D


def eudist2D(X, Y, tosqrt):
    if X.ndim == 2:
        xRow, xCol = np.shape(X)
        xSam = 1
    elif X.ndim == 3:
        xRow, xCol, xSam = np.shape(X)
    
    if Y.ndim == 2:
        yRow, yCol = np.shape(Y)
        ySam = 1
    elif Y.ndim == 3:
        yRow, yCol, ySam = np.shape(Y)
    
    D = np.zeros((xSam, ySam))
    
    if X.ndim == 2:
        for i in range(xSam):
            for j in range(ySam):
                y = Y[:, :, j]
                D[i, j] = dist2D(X, y, tosqrt)        
        
    elif X.ndim == 3:
        for i in range(xSam):
            for j in range(ySam):
                x = X[:, :, i]
                y = Y[:, :, j]
                D[i, j] = dist2D(x, y, tosqrt)
            
    return D
    

def cosdist(X, Y):
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    
    XX = X * X
    sx = np.sum(XX, axis=0)
    sx = np.sqrt(sx)
    YY = Y * Y
    sy = np.sum(YY, axis=0)
    sy = np.sqrt(sy)
    
    XY = np.dot(np.transpose(X), Y)
    #sxy = np.dot(np.transpose(sx), sy)
    
    #sx = np.reshape(sx, [ySam, 1])
    #sy = np.reshape(sy, [xSam, 1])
    sx = repVec(sx, ySam)
    sy = repVec(sy, xSam)
    sxy = sx * np.transpose(sy)
    
    D = XY / sxy
    
    return D


def vvdist(X, Y):
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    
    assert xSam == ySam, 'The num of X and Y are not identical!'
    
    d = 0
    for i in range(xSam):
        d = d + dist(X[:, i], Y[:, i], False)
    
    return d


def setdist(X, idr, idn):
    assert len(idr) == len(idn), 'The obtained NN number does not match !'
    
    idr = set(idr)
    idn = set(idn)
    idt = idr & idn
    idr = idr - idt
    idn = idr - idt
    idr = list(idr)
    idn = list(idn)
    
    #tmx = X[:, idr]
    #tmy = X[:, idn]
    
    n = len(idr)
    #D = eudist(tmx, tmy, False)
    #d = np.triu(D)
    #d = sum(sum(d))
    
    return n
    
    
def findData(val, M):
    nRow, nCol = np.shape(M)
    
    r = []
    c = []
    for i in xrange(nRow):
        for j in xrange(nCol):
            if M[i, j] == val:
                r.insert(len(r), i)
                c.insert(len(c), j)
    
    return r, c


def seArr(val, L):
    if L.ndim == 1:
        nDim = len(L)
        L = np.reshape(L, (nDim, 1))
    
    nDim = len(L)
    ind = []
    for i in range(nDim):
        if L[i, 0] == val:
            ind.insert(len(ind), i)
    
    return ind


def revArr(v):
    if v.ndim == 1:
        nDim = len(v)
        v = np.reshape(v, (nDim, 1))
    
    rv = []
    for i in range(nDim).reverse():
        rv.insert(len(rv), v[i, 0])
    
    return rv


def rev2D(X):
    xRow, xCol, xSam = np.shape(X)
    
    tmx = np.zeros((xRow, xCol))
    for i in range(xSam):
        tmp = X[:, :, i]
        tmp = tmp[:, ::-1]
        tmx[:, :, i] = tmp
        
    return tmx


def repVec(v, times):
    if v.ndim == 1:
        nDim = len(v)
        v = np.reshape(v, (nDim, 1))
    else:
        nDim, tmp = np.shape(v)
        assert tmp == 1, 'The dimensionality of data does not match !'
        #if (tmp != 1):
            #AssertionError('The dimensionality of data does not match !')
        
    M = np.zeros((nDim, 1))
    
    for i in range(times):
        M = np.column_stack((M, v))
    
    M = M[:, 1:times+1]
    
    return M


def getRank(v):
    n = len(v)
    m = 0
    for i in range(n):
        if (abs(v[i]) > 1e-7):
            m = m+1
        else:
            break
    
    v = v[0:m]
    return v, m
    
    
def ranmp(X, mDim, fDim):
    xDim, xSam = np.shape(X)
    P = np.random.randn(mDim, xDim)
    PX = np.dot(P, X)
    
    Q = np.random.randn(fDim, mDim)
    PX = np.dot(Q, PX)
    
    return PX, P, Q


def ranp(X, fDim):
    xDim, xSam = np.shape(X)
    P = np.random.randn(fDim, xDim)
    PX = np.dot(P, X)
    
    return PX, P


def PDim(P, X, ap, n):
    nDim, nSam = np.shape(P)
    ls = len(ap)
    assert n < ls, 'The index is out of range.'
    
    m = ap[n]
    tmp = P[:, 0:m]
    PX = np.dot(np.transpose(tmp), X)
    
    return PX


def diff(t_a, t_b):
    t_diff = relativedelta(t_a, t_b)
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)


def seqData(X, k, ind):
    xDim, xSam = np.shape(X)
    x = X[:, ind]
    n = xDim / k
    
    data = x[range(k)]
    for i in range(n-1):
        tmp = x[(i+1)*k:(i+2)*k]
        data = np.column_stack((data, tmp))
        
    np.transpose(data)
    return data


def getKern(X, Y, **kwargs):
    ktype = kwargs['ktype']
    t = kwargs['t']
    
    if (ktype == 'Gaussian'):
        D = eudist(X, Y, False)
        tmp = 2*(t**2)
        tmp = - D / tmp
        K = np.exp(tmp)
        
    elif (ktype == 'Cosine'):
        K = cosdist(X, Y)
        
    elif (ktype == 'linear'):
        K = np.dot(np.transpose(X), Y)
        
    return K


def midVal(v):
    if v.ndim == 1:
        nDim = len(v)
        v = np.reshape(v, (nDim, 1))
    else:
        nDim, tmp = np.shape(v)
        assert tmp == 1, 'The dimensionality of data does not match !'
        
    s = np.sum(v, axis=0)
    mval = float(s) / 2
    
    return mval
    

# ####################################################################################
# This function generates the binary labels for input labels.
# Input: L - input labels
# Output: bL - output binary Labels
# ####################################################################################
def binaryLabel(L):
    if L.ndim == 1:
        nSam = len(L)
        L = np.reshape(L, (nSam, 1))
    else:
        nSam, tmp = np.shape(L)
        assert tmp == 1, 'The dimensionality of label data is incorrect !'
        
    uL = np.unique(L)
    nL = len(uL)
    
    bL = np.ones((nL, nSam))
    bL = bL * (-1)
    for i in range(nSam):
        tmp = L[i, 0]
        ind = seArr(tmp, uL)
        
        bL[ind, i] = 1
        
    return bL


############################## Classification Function ##############################
def getMaxind(X):
    xDim, xSam = np.shape(X)
    
    maxind = np.zeros((xSam, 1))
    for i in range(xSam):
        tmp = X[0, i]
        ind = 0
        
        for j in range(xDim):
            if tmp < X[j, i]:
                tmp = j
                ind = j
            
        maxind[i] = ind
        
    return maxind
    

def rankClassifier(X, xL):
    xDim, xSam = np.shape(X)
    
    tL = getMaxind(X)
    aL = getMaxind(xL)
    
    acc = (0)
    err = (0)
    for i in range(xSam):
        if tL[i] == aL[i]:
            acc = acc + 1
        else:
            err = err + 1
            
    accuracy = float(acc) / xSam
    
    return accuracy
    

def binaryKnn(X, Y, xL, yL):
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    
    acc = (0)
    err = (0)
    for i in range(ySam):
        #print acc
        
        tmp = Y[:, i]
        d = eudist(tmp, X, False)
        
        d = np.array(d)
        idx = np.argsort(d)
        ind = idx[0, 0]
        
        tmx = xL[:, ind]
        tmy = yL[:, i]
        
        d = dist(tmx, tmy, False)
        
        if (d == 0):
            acc = acc + 1
        else:
            err = err + 1
            
    accurency = float(acc) / ySam
    
    return accurency


def knn(X, Y, xL, yL, k):
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    
    acc = (0)
    err = (0)
    for i in range(ySam):
        #print acc
        
        tmp = Y[:, i]
        d = eudist(tmp, X, False)
        
        d = np.array(d)
        idx = np.argsort(d)
        ind = idx[0, range(k)]
        
        tL = xL[ind, 0]
        count = 0
        for j in range(k):
            if (tL[j] == yL[i]):
                count = count + 1
                
        #print count
        
        if (count >= (float(k) / 2)):
            acc = acc + 1
        else:
            err = err + 1
            
    accurency = float(acc) / ySam
    
    return accurency
    
    
def knn2D(X, Y, xL, yL, k):
    xRow, xCol, xSam = np.shape(X)
    yRow, yCol, ySam = np.shape(Y)
    
    acc = (0)
    err = (0)
    for i in range(ySam):
        #print acc
        
        tmp = Y[:, :, i]
        d = eudist2D(tmp, X, False)
        
        d = np.array(d)
        idx = np.argsort(d)
        ind = idx[0, range(k)]
        
        tL = xL[ind, 0]
        count = 0
        for j in range(k):
            if (tL[j] == yL[i]):
                count = count + 1
                
        #print count
        
        if (count >= (float(k) / 2)):
            acc = acc + 1
        else:
            err = err + 1
            
    accurency = float(acc) / ySam
    
    return accurency
    
    
    