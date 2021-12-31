
#--------------------------------------------------------------------------------------------
#ikmm.py
#This file contains definitions of adaptive KMM methods.
#
#Coded by Miao Cheng
#Date: 2019-4-13
#
#All Rights Reserved.
#--------------------------------------------------------------------------------------------

import numpy as np
from kernel import *
from numpy import linalg as la
import random
from qpsolvers import solve_qp
from cvxopt import matrix, solvers

from funs import *
from nmse import *
import datetime


class ikmm(object):
    def __init__(self, X, Y, **kwargs):
        self.__X = X
        self.__Y = Y
        
        self.__xDim, self.__xSam = np.shape(X)
        self.__yDim, self.__ySam = np.shape(Y)
        
        self.__xMean = getMean(X)
        self.__xStd = getStd(X, self.__xMean)
        self.__xBeta = getProb(X, self.__xMean, self.__xStd)
        
        self.__kw = getKWidth(X)
        self.__Kxx = xysK(X, X, 'Gaussian', self.__kw)
        self.__Kxy = xysK(X, Y, 'Gaussian', self.__kw)
        
        #self.__Kxx = xyK(X, X, 'Gaussian')
        #self.__Kxy = xyK(X, Y, 'Gaussian')        
        
        
    def getWeight(self, X, ind):
        xDim, xSam = np.shape(X)
        
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
        
    
    def getAind(self, X, n):
        xDim, xSam = np.shape(X)
        
        #tmk = xyK(X, X, 'Gaussian')
        tmk = xysK(X, X, 'Gaussian', self.__kw)
        tm = np.sum(tmk, axis=0)
        
        assert len(tm) == xSam, 'The direction of operation may be incorrect !'
        idx = np.argsort(- tm)
        ix = idx[0:n]
        
        return ix
    
    
    def getBind(self, X, n, rn):
        xDim, xSam = np.shape(X)
        
        index = np.arange(xSam)
        random.shuffle(index)
        
        ind = index[0:rn]
        tX = X[:, ind]
        #tmk = xyK(tX, X, 'Gaussian')
        tmk = xysK(tX, X, 'Gaussian', self.__kw)
        tm = np.sum(tmk, axis=0)
        
        assert len(tm) == xSam, 'The direction of operation may be incorrect !'
        idx = np.argsort(- tm)
        ix = idx[0:n]
        
        return ix    
    
    
    # +++++ global kmm +++++
    def glokmm(self, n):
        xSam = self.__xSam
        ySam = self.__ySam
        
        #sKxx = xyK(self.__X, self.__X, 'Gaussian')
        sKxx = xysK(self.__X, self.__X, 'Gaussian', self.__kw)
        
        U, s, V = la.svd(sKxx)
        V = np.transpose(V)
        s, r = getRank(s)
        minv = ginv(U, V, s, r)
        
        #d = np.ones((xSam, 1)) * 0.0001
        #d = np.diag(d[:, 0])
        #tmp = self.__Kxx + d
        #minv = la.inv(tmp)
        
        minv = minv * 0.5
        
        ind = self.getAind(self.__Y, n)
        tY = self.__Y[:, ind]
        
        #tmk = xyK(self.__X, tY, 'Gaussian')
        tmk = xysK(self.__X, tY, 'Gaussian', self.__kw)
        P = np.dot(minv, tmk)
        
        trs = float(n) / ySam
        P = P * trs
        
        weight = self.getWeight(self.__Y, ind)
        P = P * weight
        
        return P
    
    
    def iglokmm(self, n):
        P = self.glokmm(n)
        #mse = inmse(self.__X, self.__Y, P)
        mse = nmser(P, self.__Kxx, self.__Kxy)
        
        return mse
    
    
    # +++++ skmm +++++
    def setLayer(self, b, P, k):
        bDep, bRow, bCol = np.shape(b)
        pRow, pCol = np.shape(P)
        assert bRow == pRow, 'The dimensionality of b and P are not identical !'
        assert bCol == pCol, 'The dimensionality of b and P are not identical !'
        
        #for i in range(pRow):
            #for j in range(pCol):
                #b[k, i, j] = P[i, j]
                
        b[k, :, :] = P
        
        return b
    
    
    def together(self, b):
        bDep, bRow, bCol = np.shape(b)
        assert bDep >= 1, 'The depth of b is incorrect !'
        
        m = np.zeros((bRow, bCol))
        for i in range(bRow):
            for j in range(bCol):
                for k in range(bDep):
                    m[i, j] = m[i, j] + b[k, i, j]
                    
        return m
    
    
    def iTogether(self, B):
        bDep, bRow, bCol = np.shape(B)
        assert bDep >= 1, 'The depth of b is incorrect !'
        
        #sKxx = xysK(self.__X, self.__X, 'Gaussian', self.__kw)
        #sKxy = xysK(self.__X, self.__Y, 'Gaussian', self.__kw) 
        
        P = np.zeros((bDep, bDep))
        q = np.zeros((bDep, 1))
        for i in range(bDep):
            tmb = B[i, :, :]
            tmp = np.dot(np.transpose(tmb), self.__Kxx)
            tmp = np.dot(tmp, tmb)
            tmq = np.sum(np.sum(tmp))
            tm = 1 / 2
            P[i, i] = tm * tmq
            
            tmp = np.dot(np.transpose(tmb), self.__Kxy)
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
    
    
    def jTogether(self, B, X, Y):
        bDep, bRow, bCol = np.shape(B)
        assert bDep >= 1, 'The depth of b is incorrect !'
        
        sKxx = xysK(X, X, 'Gaussian', self.__kw)
        sKxy = xysK(X, Y, 'Gaussian', self.__kw) 
        
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
        
        
    def skmm(self, X, Y, n, rn):       # skmm(X, Y, n, rn, k)
        xDim, xSam = np.shape(X)
        yDim, ySam = np.shape(Y)
        assert xDim == yDim, 'The dimensionality of X and Y are not identical !'
        
        #sKxx = xyK(X, X, 'Gaussian')
        sKxx = xysK(X, X, 'Gaussian', self.__kw)
        
        U, s, V = la.svd(sKxx)
        V = np.transpose(V)
        s, r = getRank(s)
        minv = ginv(U, V, s, r)        
        
        #d = np.ones((xSam, 1)) * 0.0001
        #d = np.diag(d[:, 0])
        #tmp = sKxx + d
        #minv = la.inv(tmp)
        
        minv = minv * 0.5
        
        ind = self.getBind(Y, n, rn)
        tY = Y[:, ind]
        
        #tmk = xyK(X, tY, 'Gaussian')
        tmk = xysK(X, tY, 'Gaussian', self.__kw)
        P = np.dot(minv, tmk)
        
        trs = float(n) / ySam
        P = P * trs
        
        weight = self.getWeight(Y, ind)
        P = P * weight
        
        return P
    
    
    def iskmm(self, X, Y, n, rn, times):       # iskmm(X, Y, n, rn, k, times)
        xDim, xSam = np.shape(X)
        yDim, ySam = np.shape(Y)
        assert xDim == yDim, 'The dimensionality of X and Y are not identical !'
        
        b = np.zeros((times, xSam, n))
        for i in range(times):
            ch = str(i) + '-th running'
            print(ch)
            
            P = self.skmm(X, Y, n, rn)
            self.setLayer(b, P, i)
        
        #m = self.iTogether(b)
        m = self.jTogether(b, X, Y)
        
        #tmp = inmse(X, Y, m)
        tmp = nmser(m, self.__Kxx, self.__Kxy)
        
        return tmp
    
    
    # ++++++++++ Temporal AMKM +++++++++++
    def tskmm(self, X, Y, tY, n, rn, times):
        xDim, xSam = np.shape(X)
        yDim, ySam = np.shape(Y)
        assert xDim == yDim, 'The dimensionality of X and Y are not identical !'
        
        Y = np.column_stack((Y, tY))
        b = np.zeros((times, xSam, n))
        for i in range(times):
            #ch = str(i) + '-th running'
            #print(ch)
            
            P = self.skmm(X, Y, n, rn)
            self.setLayer(b, P, i)
        
        m = self.jTogether(b, X, Y)
        
        #tmp = inmse(X, Y, m)
        sKxy = xysK(X, Y, 'Gaussian', self.__kw)
        tmp = nmser(m, self.__Kxx, sKxy)
        
        return tmp
    
    
    def itskmm(self, X, Y, cY, n, rn, times, nSam):
        yDim, ySam = np.shape(X)
        cDim, cSam = np.shape(cY)
        assert yDim == cDim, 'The dimensionality of Y and cY are not identical !'
        
        t = int(np.floor(cSam / nSam))
        nmse = np.zeros((t, 1))
        cost = np.zeros((t, 1))
        for i in range(t):
            tY = cY[:, i*nSam:(i+1)*nSam-1]
            
            oldtime = datetime.datetime.now()
            tmp = self.tskmm(X, Y, tY, n, rn, times)
            newtime = datetime.datetime.now()
            tmq = (newtime - oldtime).microseconds        
            nmse[i] = tmp
            cost[i] = tmq
            
            ch = str(i) + '-th slice: ' + str(tmp)
            th = str(i) + '-th cost time:' + str(tmq)
            print(ch)
            print(th)
            
        return nmse, cost
    
    
       