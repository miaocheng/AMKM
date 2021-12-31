
#----------------------------------------------------------------------------------------------------
# Name:        nmse.py
#
# This code implements the calculation methods of NMSE.
#
# Coded by:      Miao Cheng
# E-mail: mewcheng@gmail.com
# Current Version: v0.91
# Created Date:     11/26/2018
# All Rights Reserved
#----------------------------------------------------------------------------------------------------

import numpy as np
from cala import *
from kernel import *
from utils import *


def impTrain(Kxx):
    kRow, kCol = np.shape(Kxx)
    assert kRow == kCol, 'The dimensionality of data are not identical !'
    
    tm = np.sum(Kxx, axis=0)
    s = 0
    for i in range(kRow):
        s += tm[i]
        
    for i in range(kRow):
        tm[i] = tm[i] / s
        
    return tm


def iimpTest(Kxy, P):
    kRow, kCol = np.shape(Kxy)
    pRow, pCol = np.shape(P)
    assert pRow == kRow, 'The dimensionality of Kxy and P are not identical !'
    
    pkxy = np.dot(np.transpose(P), Kxy)
    tm = np.sum(pkxy, axis=0)
    s = 0
    for i in range(pCol):
        s += tm[i]
        
    for i in range(pCol):
        tm[i] = tm[i] / s
        
    return tm
    

def inmse(X, Y, P):       # inmse(X, Y, P, nCol)
    xDim, xSam = np.shape(X)
    yDim, ySam = np.shape(Y)
    
    Kxx = xyK(X, X, 'Gaussian')
    Kxy = xyK(X, Y, 'Gaussian')
    
    imte = iimpTest(Kxy, P)
    imtr = impTrain(Kxx)
    
    mse = 0
    for i in range(xSam):
        tmp = imte[i] - imtr[i]
        tmp = tmp * tmp
        tmp = np.sum(tmp)
        mse += tmp
        
    mse = mse / xSam
    
    return mse

# +++++ nmse of Amsa +++++
def impTestr(Kxy):
    tmp, ySam = np.shape(Kxy)
    tmp = np.sum(Kxy, axis=1)
    tmp = tmp / ySam
    
    s = np.sum(tmp)
    tm = tmp / s
    
    return tm


def impTrainr(P, Kxx):
    tmp, xSam = np.shape(Kxx)
    
    for i in range(xSam):
        Kxx[i, i] = 0
        
    tmp = np.dot(np.transpose(P), Kxx)
    tmp = np.sum(tmp, axis=0)
    tmp = np.transpose(tmp)
    tmp = tmp / xSam
    
    s = np.sum(tmp)
    tm = tmp / s
    
    return tm


def nmser(P, Kxx, Kxy):
    P = maxZero(P)
    xSam, ySam = np.shape(Kxy)
    
    imte = impTestr(Kxy)
    imtr = impTrainr(P, Kxx)
    
    tmp = imte - imtr
    tmp = tmp ** 2
    mse = np.sum(tmp)
    
    mse = mse / xSam
    
    return mse


