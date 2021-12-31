
#!/usr/bin/env python
'''

CREATED: 2018-11-06
Modified: 2019-11-03

'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from cala import *
from ikmm import *

#import time
import datetime
import random


data = sio.loadmat('./data/cifar20_ind.mat')
Data = data['Data']
index = data['index']
index = index.astype(int)
index -= 1

nDim, nSam = np.shape(Data)

inmse = []
itime = []
for t in range(5):
    
    # +++++ monks +++++
    #idx = index[t, 0:50]
    #idy = index[t, 50:300]
    
    # +++++ ionoshpere +++++
    #idx = index[t, 0:100]
    #idy = index[t, 100:340]
    
    # +++++ Climate +++++
    #idx = index[t, 0:100]
    #idy = index[t, 100:500]    
    
    # +++++ Forest +++++
    #idx = index[t, 0:500]
    #idy = index[t, 500:7500]
    
    # +++++ Letter +++++
    #idx = index[t, 0:500]
    #idy = index[t, 500:7500]
    
    # +++++ Cifar +++++
    idx = index[t, 0:500]
    idy = index[t, 500:7500]
    
    X = Data[:, idx]
    Y = Data[:, idy]
    
##########################################################################
    #data = sio.loadmat('./data/Lett500_5000.mat')
    #X = data['X']
    #Y = data['Y']
    
    #Data = np.column_stack((X, Y))
    #nDim, nSam = np.shape(Data)
    #ind = list(range(nSam))
    #random.shuffle(ind)
    
    #tmp = Data
    #for i in range(nDim):
        #for j in range(nSam):
            #tid = ind[j]
            #Data[i, j] = tmp[i, tid]
            
    
    #X = Data[:, 0:500]
    #Y = Data[:, 500::]
    
    ############################################################################
    starttime1 = datetime.datetime.now()
    starttime2 = datetime.datetime.now().microsecond
    
    m = ikmm(X, Y)
    
    #nmse = m.iglokmm(100)
    ##nmse = inmse(X, Y, P)
    
    nmse = m.iskmm(X, Y, 100, 50, 5)
    
    endtime1 = datetime.datetime.now()
    endtime2 = datetime.datetime.now().microsecond
    
    duration1 = (endtime1 - starttime1).microseconds
    duration2 = endtime2 - starttime2
    
    print(nmse)
    print(duration1)
    print(duration2)
    
    inmse.append(nmse)
    itime.append(duration1)
    # -------------------------------------------------------------------
    
    
mx, stdx = mstd(inmse)
mt, stdt = mstd(itime)
print('The mean nmse: ')
print(mx)
print('The mean time: ')
print(mt)


print('Done !')






