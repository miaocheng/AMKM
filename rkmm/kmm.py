
#----------------------------------------------------------------------------------------------------
# kmm.py
# This file contains the definition of the class of kernel mean matching.
#
# Coded by Miao Cheng
#
# All Rights Reserved.
#----------------------------------------------------------------------------------------------------

import numpy as np
from kernel import *
from numpy import linalg as la
from utils import *
import random


class kmm(object):
    def __init__(self, X, Y, **kwargs):
        self.__X = X
        self.__Y = Y
        
        self.__xDim, self.__xSam = np.shape(X)
        self.__yDim, self.__ySam = np.shape(Y)
        
        self.__xMean = getMean(X)
        self.__xStd = getStd(X, self.__xMean)
        self.__xBeta = getProb(X, self.__xMean, self.__xStd)
        
        self.__kw = getKWidth(X)
        #self.__Kxx = xysK(X, X, 'Gaussian', self.__kw)
        #self.__Kxy = xysK(X, Y, 'Gaussian', self.__kw)
        
        self.__Kxx = xyK(X, X, 'Gaussian')
        self.__Kxy = xyK(X, Y, 'Gaussian')         
        
        
    def kmm(self):
        xSam = self.__xSam
        ySam = self.__ySam
        
        #U, s, V = la.svd(self.__Kxx)
        #s, r = getRank(s)
        #minv = ginv(U, V, s, r)
        
        d = np.ones((xSam, 1)) * 0.0001
        d = np.diag(d[:, 0])
        tmp = self.__Kxx + d
        minv = la.inv(tmp)
        
        minv = minv * 0.5
        
        trs = float(xSam) / ySam
        tmp = self.__Kxy * trs
        P = np.dot(minv, tmp)
        
        return P
        
        
    def lockmm(self, k):
        xSam = self.__xSam
        ySam = self.__ySam
        
        lkx = locKxx(self.__X, k)
        lky = locKxy(self.__X, self.__Y, k)
        
        #U, s, V = la.svd(lkx)
        #s, r = getRank(s)
        #minv = ginv(U, V, s, r)        
        
        d = np.ones((xSam, 1)) * 0.0001
        d = np.diag(d[:, 0])
        tmp = lkx + d
        minv = la.inv(tmp)
        minv = minv * 0.5
        
        trs = float(xSam) / ySam
        tmp = lky * trs
        
        P = np.dot(minv, lky)
        
        return P
        
    
    def enkmm(self, k):
        xSam = self.__xSam
        ySam = self.__ySam
        
        #U, s, V = la.svd(self.__Kxx)
        #s, r = getRank(s)
        #minv = ginv(U, V, s, r)
        
        d = np.ones((xSam, 1)) * 0.0001
        d = np.diag(d[:, 0])
        tmp = self.__Kxx + d
        minv = la.inv(tmp)
        minv = minv * 0.5
        
        #ran = list(range(self.__ySam))
        #random.shuffle(ran)
        
        num = int(np.floor(self.__ySam / k))
        P = np.zeros((self.__xSam, num))
        
        for i in range(k):
            if i != k-1:
                start = i*num
                end = (i+1)*num -1
            else:
                start = i*num
                end = self.__ySam-1
                
            tmk = self.__Kxy[:, start:end+1]
            tmp = np.dot(minv, tmk)
            
            d = end - start + 1
            trs = float(d) / self.__ySam
            tmp = tmp * trs
            #tmp = tmp * (float(1) / k)
            
            for ii in range(self.__xSam):
                for jj in range(num):
                    P[ii, jj] = P[ii, jj] + tmp[ii, jj]
                    
                    
        return P
    
    
    # +++++ The related functions of standard NMSE +++++
    def impTestl(self, P):
        nRow, nCol = np.shape(P)
        
        Kxy = self.__Kxy
        Kxx = self.__Kxx
        
        for i in range(self.__xSam):
            Kxx[i, i] = 0
        
        tmp = np.sum(Kxy, axis=1)
        tmp = tmp * (float(self.__xSam) / self.__ySam)
        temp = P
        tmq = np.dot(np.transpose(temp), Kxx)
        tmq = np.sum(tmq, axis=0)
        tmq = tmq / nCol
        
        tm = tmp - tmq
        t = np.sum(tm)
        tm = tm / t
        
        return tm
        
    
    def impTest1(self, P):
        nRow, nCol = np.shape(P)
        
        #tmp = np.sum(P, axis=1)
        #tm = tmp / nCol
        
        P = maxZero(P)
        P = justNorm(P)
        tmp = np.sum(P, axis=1)
        tm = tmp / nCol
        
        #tm = P[:, 0]
        
        s = np.sum(tm)
        
        est = tm / s
        
        return est
    
    
    # +++++ For enkmm +++++
    def impTest2(self, P, k):
        xSam = self.__xSam
        ySam = self.__ySam
        nRow, nCol = np.shape(P)
        Kxy = self.__Kxy
        
        num = int(np.floor(ySam / k))
        tmp = np.transpose(P)
        W = np.dot(tmp, self.__Kxy)
        sM = W[0:xSam, 0:ySam]
        
        tm = np.sum(sM, axis=1)
        s = np.sum(tm)
        tm = tm / s
        
        return tm
    
    
    def impTrain0(self):
        Kxx = self.__Kxx
        for i in range(self.__xSam):
            Kxx[i, i] = 0
            
        tmp = np.sum(Kxx, axis=0)
        tmp = tmp / (self.__xSam - 1)
        s = np.sum(tmp)
        tm = tmp / s
        
        return tm
    
    
    def impTrainl(self):
        X = self.__X
        xDim, xSam = np.shape(X)
        mx = getMean(X)
        tmp = X - repVec(mx, xSam)
        tmp = tmp ** 2
        tmp = np.sum(tmp, axis=0)
        tmp = np.exp(-tmp)
        
        tm = 1 / tmp
        s = np.sum(tm)
        tm = tm / s
        
        return tm
        
    
    def impTrain(self):
        tmp = self.__xBeta
        tm = 1 / tmp
        s = np.sum(tm)
        tm = tm / s
        
        return tm
        
        
    # ++++++ nmse for standard method +++++
    def nmse1(self, P):
        imte = self.impTest1(P)
        imtr = self.impTrain()
        
        tmp = imte - imtr
        tmp = tmp ** 2
        mse = np.sum(tmp)
        
        mse = mse / self.__xSam
        
        return mse
    
    
    # +++++ nmse for enkmm +++++
    def nmse2(self, P, k):
        imte = self.impTest2(P, k)
        imtr = self.impTrain()
        
        mse = 0
        tmp = imte - imtr
        tmp = tmp ** 2
        mse = np.sum(tmp)
        
        mse = mse / self.__xSam
        
        return mse
    
    
    def nmsel(self, P):
        P = maxZero(P)
        
        imte = self.impTestl(P)
        imtr = self.impTrain0()
        
        tmp = imte - imtr
        tmp = tmp ** 2
        mse = np.sum(tmp)
        
        mse = mse / self.__xSam
        
        return mse
    
    
    # +++++ nmse for essentials +++++
    def impTestr(self):
        Kxy = self.__Kxy
        tmp = np.sum(Kxy, axis=1)
        tmp = tmp / self.__ySam
        
        s = np.sum(tmp)
        tm = tmp / s
        
        return tm
    
    
    def impTrainr(self, P):
        Kxx = self.__Kxx
        for i in range(self.__xSam):
            Kxx[i, i] = 0
            
        tmp = np.dot(np.transpose(P), Kxx)
        tmp = np.sum(tmp, axis=0)
        tmp = tmp / (self.__xSam - 1)
        
        s = np.sum(tmp)
        tm = tmp / s
        
        return tm
        
    
    def nmser(self, P):
        P = maxZero(P)
        
        imte = self.impTestr()
        imtr = self.impTrainr(P)
        
        tmp = imte - imtr
        tmp = tmp ** 2
        mse = np.sum(tmp)
        
        mse = mse / self.__xSam
        
        return mse
    
    
    