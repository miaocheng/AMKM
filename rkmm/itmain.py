
#!/usr/bin/env python
'''

CREATED: 2018-11-06
Modified: 2019-11-03

'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from cala import *
from kmm import *

#import time
import datetime
import random


data = sio.loadmat('./data/monks_ind.mat')
Data = data['Data']
index = data['index']
index = index.astype(int)
index -= 1

nDim, nSam = np.shape(Data)


inmse = []
itime = []
for t in range(5):
    idx = index[t, 0:80]
    idy = index[t, 80:580]
    
    X = Data[:, idx]
    Y = Data[:, idy]
    
# --------------------------------------------------------------------------
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
    
# --------------------------------------------------------------------------

    starttime1 = datetime.datetime.now()
    starttime2 = datetime.datetime.now().microsecond
    
    m = kmm(X, Y)
    
    #P = m.kmm()
    #nmse = m.nmser(P)
    
    #P = m.lockmm(5)
    #nmse = m.nmser(P)
    
    #P = m.enkmm(5)
    ##nmse = m.nmse2(P, 2)
    #nmse = m.nmser(P)
    
    #nmse = tglokmm(X, tY, cY, 100, 500)
    #print(nmse)
    
    #nmse = itskmm(X, tY, cY, 100, 500, 5, 500)
    
    endtime1 = datetime.datetime.now()
    endtime2 = datetime.datetime.now().microsecond
    
    duration1 = endtime1 - starttime1
    duration2 = endtime2 - starttime2
    
    print(nmse)
    print(duration1)
    print(duration2)  
    
    
    inmse.append(nmse)
    itime.append(duration2)
# -------------------------------------------------------------------

mx, stdx = mstd(inmse)
mt, stdt = mstd(itime)
print('The mean nmse: ')
print(mx)
print('The mean time: ')
print(mt)


print('Done !')






