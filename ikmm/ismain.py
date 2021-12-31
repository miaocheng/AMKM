
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
from ikmm import *

#import time
import datetime


data = sio.loadmat('./data/Lett_ind.mat')
Data = data['Data']
index = data['index']
index = index.astype(int)
index -= 1

nDim, nSam = np.shape(Data)

#snm = 0
#sti = 0
#sdu1 = 0
#sdu2 = 0

n = 5


inmse = []
itime = []
for t in range(n):
    idx = index[t, 0:500]
    idy = index[t, 500:3500]
    idc = index[t, 3500:7500]
    
    X = Data[:, idx]
    tY = Data[:, idy]
    cY = Data[:, idc]
    
    #############################################################
    
    m = ikmm(X, tY)
    
    #nmse = m.iglokmm(100)
    ##nmse = inmse(X, Y, P)
    #print(nmse)
    
    #nmse = m.iskmm(X, Y, 100, 50, 5)
    #print(nmse)
    
    #tY = Y[:, 0:500]
    #cY = Y[:, 500:5000]
    
    ##nmse = tglokmm(X, tY, cY, 100, 500)
    ##print(nmse)
    
    nmse, cost = m.itskmm(X, tY, cY, 100, 50, 5, 500)
    
    inmse.append(nmse)
    itime.append(cost)    
    # -------------------------------------------------------------------


mx, stdx = mnstd(inmse)
mt, stdt = mnstd(itime)
print('The mean nmse: ')
print(mx)
print('The mean time: ')
print(mt)



print('Done !')






