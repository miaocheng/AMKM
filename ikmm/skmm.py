
#----------------------------------------------------------------------------------------------------
'''
skmm.py
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


class skmm(object):
    def __init__(self, X, Y, cY, m, nSam, **kwargs):
        self.__X = X
        self.__Y = Y
        self.__cY = cY
        self.__m = m
        self.__nSam = nSam
        
        self.__mx = getMean(Y)
        
        self.__xDim, self.__xSam = np.shape(X)
        self.__yDim, self.__ySam = np.shape(Y)
        self.__cDim, self.__cSam = np.shape(cY)
        
        self.__xMean = getMean(X)
        self.__xStd = getStd(X, self.__xMean)
        self.__xBeta = getProb(X, self.__xMean, self.__xStd)
        
        self.__kw = getKWidth(X)
        self.__Kxx = xysK(X, X, 'Gaussian', self.__kw)
        self.__Kxy = xysK(X, Y, 'Gaussian', self.__kw)
        
        #self.__Kxx = xyK(X, X, 'Gaussian')
        #self.__Kxy = xyK(X, Y, 'Gaussian')
        
        
    #def updMean(self, X, mx, Y):
    def updMean(self, X, Y):
        xDim, xSam = np.shape(X)
        yDim, ySam = np.shape(Y)
        assert xDim == yDim, 'The dimensionality of X and Y are not identical !'
        
        mx = self.__mx
        
        n = xSam + ySam
        
        for i in range(xDim):
            mx[i] = mx[i] * xSam
            
            for j in range(ySam):
                mx[i] = mx[i] + Y[i][j]
                
            mx[i] = mx[i] / n
            
        self.__mx = mx
            
        return mx


    def updY(self, X, tX):
        xDim, xSam = np.shape(X)
        tDim, tSam = np.shape(Y)
        assert xDim == tDim, 'The dimensionality of X and tX are not identical !'
        
        n = xSam + tSam
        
        Y = np.column_stack((X, tX))
        
        return Y


    def getAind(self, X, n):
        xDim, xSam = np.shape(X)
        
        tmk = xysK(X, X, 'Gaussian', self.__kw)   # cannot replaced with self.__Kxy
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
        tmk = xysK(tX, X, 'Gaussian', self.__kw)
        tm = np.sum(tmk, axis=0)
        
        assert len(tm) == xSam, 'The direction of operation may be incorrect !'
        idx = np.argsort(- tm)
        ix = idx[0:n]
        
        return ix


    def getWeight(self, X, ind, mx):
        xDim, xSam = np.shape(X)
        #tDim, tSam = np.shape(tX)
        #assert xDim == tDim, 'The dimensionality of X and tX are not identical !'
        
        #mx = np.mean(X, axis=1)
        mx = self.__mx
        mw = np.zeros((xSam, 1))
        for i in range(xSam):
            tmp = X[:, i] - mx
            tmp = tmp * tmp
            tmp = np.sum(tmp)
            tmp = np.exp(-tmp / self.__kw)
            mw[i, 0] = tmp
        
        tmw = mw[ind, 0]
        sw = np.sum(mw)
        stw = np.sum(tmw)
        weight = float(stw) / sw
        
        return weight

    
    # +++++ The kmm functions +++++
    def setLayer(self, b, P, k):
        bDep, bRow, bCol = np.shape(b)
        pRow, pCol = np.shape(P)
        assert bRow == pRow, 'The dimensionality of b and P are not identical !'
        assert bCol == pCol, 'The dimensionality of b and P are not identical !'
        
        for i in range(pRow):
            for j in range(pCol):
                b[k, i, j] = P[i, j]
                
        return b
    
    
    def together(self, b):
        bDep, bRow, bCol = np.shape(b)
        assert bDep > 1, 'The depth of b is incorrect !'
        
        m = np.zeros((bRow, bCol))
        for i in range(bRow):
            for j in range(bCol):
                for k in range(bDep):
                    m[i, j] = m[i, j] + b[k, i, j]
                    
                    
        return m


    # +++++ global kmm +++++
    def glokmm(self, X, Y, n):
        xDim, xSam = np.shape(X)
        yDim, ySam = np.shape(Y)
        assert xDim == yDim, 'The dimensionality of X and Y are not identical !'
        
        sKxx = xysK(X, X, 'Gaussian', self.__kw)
        #sKxx = self.__Kxy
        
        U, s, V = la.svd(sKxx)
        V = np.transpose(V)
        s, r = getRank(s)
        minv = ginv(U, V, s, r)
        
        minv = minv * 0.5
        
        ind = self.getAind(Y, n)
        tY = Y[:, ind]
        
        tmk = xysK(X, tY, 'Gaussian', self.__kw)
        P = np.dot(minv, tmk)
        
        trs = float(n) / ySam
        P = P * trs
        
        weight = self.getWeight(Y, ind, self.__mx)
        P = P * weight
        
        return P, sKxx
    
    
    def iglokmm(self, X, Y, n):
        P, sKxx = self.glokmm(X, Y, n)
        sKxy = xysK(X, Y, 'Gaussian', self.__kw)
        
        #tmp = inmse(X, Y, P)
        tmp = nmser(P, sKxx, sKxy)
        
        return tmp
    
    
    #def tglokmm(self, m, nSam):
    def tglokmm(self):
        X = self.__X
        Y = self.__Y
        cY = self.__cY
        
        #yDim, ySam = np.shape(X)
        #cDim, cSam = np.shape(cY)
        #assert yDim == cDim, 'The dimensionality of Y and cY are not identical !'
        
        ySam = self.__ySam
        cSam = self.__cSam
        m = self.__m
        nSam = self.__nSam
        
        n = int(np.floor(cSam / nSam))
        nmse = np.zeros((n, 1))
        cost = np.zeros((n, 1))
        tmy = Y
        for i in range(n):
            tY = cY[:, i*nSam:(i+1)*nSam]
            tmy = np.column_stack((tmy, tY))
            
            oldtime = datetime.now()
            tmp = self.iglokmm(X, tmy, m)
            newtime = datetime.now()
            tmq = (newtime - oldtime).microseconds
            nmse[i] = tmp
            cost[i] = tmq
            
            ch = str(i) + '-th slice: ' + str(tmp)
            th = str(i) + '-th cost time:' + str(tmq)
            print(ch)
            print(th)
            print('-------------------------------------')
            
        return nmse, cost
        
        
    # +++++ skmm +++++
    def skmm(self, X, Y, n, rn, mx):       # skmm(X, Y, n, rn, k)
        xDim, xSam = np.shape(X)
        yDim, ySam = np.shape(Y)
        assert xDim == yDim, 'The dimensionality of X and Y are not identical !'
        
        #Kxx = xysK(X, X, 'Gaussian', self.__kw)
        
        #d = np.ones((xSam, 1)) * 0.0001
        #d = np.diag(d[:, 0])
        #tmp = self.__Kxx + d
        #minv = la.inv(tmp)
        
        U, s, V = la.svd(self.__Kxx)
        V = np.transpose(V)
        s, r = getRank(s)
        minv = ginv(U, V, s, r)        
        
        minv = minv * 0.5
        
        ind = self.getBind(Y, n, rn)
        tY = Y[:, ind]
        
        #tmk = xyK(X, tY, 'Gaussian')
        tmk = xysK(X, tY, 'Gaussian', self.__kw)
        P = np.dot(minv, tmk)
        
        trs = float(n) / ySam
        P = P * trs
        
        weight = self.getWeight(Y, ind, mx)
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
        
        m = self.together(b)
        tmp = inmse(X, Y, m)
        
        return tmp
    
    
    # +++++ Temporal skmm +++++
    def tskmm(self, X, Y, tY, n, rn, times):
        xDim, xSam = np.shape(X)
        yDim, ySam = np.shape(Y)
        assert xDim == yDim, 'The dimensionality of X and Y are not identical !'
        
        Y = np.column_stack((Y, tY))
        b = np.zeros((times, xSam, n))
        
        mx = self.updMean(Y, tY)
        
        for i in range(times):
            #ch = str(i) + '-th running'
            #print(ch)
            
            P = self.skmm(X, Y, n, rn, mx)
            self.setLayer(b, P, i)
            
            
        Kxy = xysK(X, Y, 'Gaussian', self.__kw)
        m = self.together(b)
        m = m / times
        
        tmp = nmser(m, self.__Kxx, Kxy)
        
        return tmp, Y
    
    
    def itskmm(self, im, rn, times):
        X = self.__X
        Y = self.__Y
        cY = self.__cY
        
        ySam = self.__ySam
        cSam = self.__cSam
        nSam = self.__nSam
        
        #yDim, ySam = np.shape(X)
        #cDim, cSam = np.shape(cY)
        #assert yDim == cDim, 'The dimensionality of Y and cY are not identical !'
        
        n = int(np.floor(cSam / nSam))
        nmse = np.zeros((n, 1))
        cost = np.zeros((n, 1))
        for i in range(n):
            tY = cY[:, i*nSam:(i+1)*nSam]
            
            oldtime = datetime.now()
            tmp, Y = self.tskmm(X, Y, tY, im, rn, times)
            newtime = datetime.now()
            tmq = (newtime - oldtime).microseconds        
            nmse[i] = tmp
            cost[i] = tmq
            
            ch = str(i) + '-th slice: ' + str(tmp)
            th = str(i) + '-th cost time:' + str(tmq)
            print(ch)
            print(th)        
            
        return nmse, cost
     
    
    # +++++ temporal enskmm +++++
    def senkmm(self, X, Y, k):
        xDim, xSam = np.shape(X)
        yDim, ySam = np.shape(Y)
        
        #U, s, V = la.svd(self.__Kxx)
        #V = np.transpose(V)
        #s, r = getRank(s)
        #minv = ginv(U, V, s, r)
        
        Kxx = xysK(X, X, 'Gaussian', self.__kw)
        
        d = np.ones((xSam, 1)) * 0.0001
        d = np.diag(d[:, 0])
        tmp = Kxx + d
        minv = la.inv(tmp)
        
        #U, s, V = la.svd(Kxx)
        #V = np.transpose(V)
        #s, r = getRank(s)
        #minv = ginv(U, V, s, r)        
        
        minv = minv * 0.5
        
        #ran = list(range(self.__ySam))
        #random.shuffle(ran)
        #tY = Y[:, ran]
        Kxy = xysK(X, Y, 'Gaussian', self.__kw)

        num = int(np.floor(ySam / k))
        P = np.zeros((self.__xSam, num))

        for i in range(k):
            if i != k-1:
                start = i*num
                end = (i+1)*num
            else:
                start = i*num
                end = self.__ySam

            tmk = Kxy[:, start:end]
            tmp = np.dot(minv, tmk)

            d = end - start
            trs = float(d) / self.__ySam
            tmp = tmp * trs
            tmp = tmp * (float(1) / k)

            for ii in range(self.__xSam):
                for jj in range(d):
                    P[ii, jj] = P[ii, jj] + tmp[ii, jj]


        return P, Kxx
    
    
    def ienkmm(self, X, Y, k):
        P, sKxx = self.senkmm(X, Y, k)
        sKxy = xysK(X, Y, 'Gaussian', self.__kw)
        
        #tmp = inmse(X, Y, P)
        tmp = nmser(P, sKxx, sKxy)        
        
        return tmp    
    
    
    def tenkmm(self, k):
        X = self.__X
        Y = self.__Y
        cY = self.__cY
        xSam = self.__xSam
        ySam = self.__ySam
        cSam = self.__cSam
        
        nSam = self.__nSam
        
        #U, s, V = la.svd(self.__Kxx)
        #V = np.transpose(V)
        #s, r = getRank(s)
        #minv = ginv(U, V, s, r)
        
        #d = np.ones((xSam, 1)) * 0.0001
        #d = np.diag(d[:, 0])
        #tmp = self.__Kxx + d
        #minv = la.inv(tmp)
        #minv = minv * 0.5
        
        n = int(np.floor(cSam / nSam))
        nmse = np.zeros((n, 1))
        cost = np.zeros((n, 1))
        tmy = Y
        
        for iter in range(n):
            tY = cY[:, iter*nSam:(iter+1)*nSam]
            tmy = np.column_stack((tmy, tY))
            
            oldtime = datetime.now()
            tmp = self.ienkmm(X, tmy, k)
            newtime = datetime.now()
            tmq = (newtime - oldtime).microseconds
            nmse[iter] = tmp
            cost[iter] = tmq
            
            ch = str(iter) + '-th slice: ' + str(tmp)
            th = str(iter) + '-th cost time:' + str(tmq)
            print(ch)
            print(th)
            print('-------------------------------------')            
        

        return nmse, cost
    
    
    
    