
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
import random

data = sio.loadmat('./data/ionosphere.mat')
Data = data['Data']
#nDim, nSam = np.shape(Data)
#ind = list(range(nSam))
#random.shuffle(ind)


##################################################################################
#kk = 5
#index = np.zeros((kk, nSam))
#for i in range(kk):
    #ind = list(range(nSam))
    #random.shuffle(ind)
    #index[i, :] = ind
    
    
#sio.savemat('iono_index.mat', {'index': index})

data = sio.loadmat('iono_index.mat')
index = data['index']
index = index.astype(int)

i = 4
ind = index[i, :]
idx = ind[0:80]
idy = ind[80::]

X = Data[:, idx]
Y = Data[:, idy]



#tmp = Data
#for i in range(nDim):
    #for j in range(nSam):
        #tid = ind[j]
        #Data[i, j] = tmp[i, tid]


#X = Data[:, 0:80]
#Y = Data[:, 80:351]

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

m = kmm(X, Y)


P = m.kmm()
nmse = m.nmser(P)

#P = m.lockmm(5)
#nmse = m.nmser(P)

#P = m.enkmm(5)
##nmse = m.nmse2(P, 2)
#nmse = m.nmser(P)

#nmse = tglokmm(X, tY, cY, 100, 500)
#print(nmse)

#nmse = itskmm(X, tY, cY, 100, 500, 5, 500)

print(nmse)
# -------------------------------------------------------------------


print('Done !')






