
#!/usr/bin/env python
'''

CREATED: 2018-11-06

'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from cala import *
from kmm import *

#import time
import datetime


data = sio.loadmat('./data/ionosphere.mat')
Data = data['Data']
#nDim, nSam = np.shape(Data)
#ind = list(range(nSam))
#random.shuffle(ind)

#tmp = Data
#for i in range(nDim):
    #for j in range(nSam):
        #tid = ind[j]
        #Data[i, j] = tmp[i, tid]


#X = Data[:, 0:100]
#Y = Data[:, 100:351]

#data = sio.loadmat('iono_index.mat')
#index = data['index']
#index = index.astype(int)

#i = 4
#ind = index[i, :]
#idx = ind[0:80]
#idy = ind[80::]

#X = Data[:, idx]
#Y = Data[:, idy]

###################################################################################
#data = sio.loadmat('Lett500_5000.mat')
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

#####################################################################
data = sio.loadmat('./data/cifar20_ind.mat')
Data = data['Data']
index = data['index']
index = index.astype(int)
index -= 1

nDim, nSam = np.shape(Data)

snm = 0
sti = 0
sdu1 = 0
sdu2 = 0
n = 5

#for t in range(n):
t = 0

idx = index[t, 0:500]
idy = index[t, 500:3500]
idc = index[t, 3500:7500]

X = Data[:, idx]
tY = Data[:, idy]
cY = Data[:, idc]


#############################################################

#m = ikmm(X, Y)


#nmse = m.iglokmm(100)
##nmse = inmse(X, Y, P)
#print(nmse)

#nmse = m.iskmm(X, Y, 100, 50, 5)
#print(nmse)


#tY = Y[:, 0:500]
#cY = Y[:, 500:5000]


##nmse = tglokmm(X, tY, cY, 100, 500)
##print(nmse)

nmse = itskmm(X, tY, cY, 100, 500, 5, 500)
print(nmse)
# -------------------------------------------------------------------


print('Done !')






