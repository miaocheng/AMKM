
#!/usr/bin/env python
'''

CREATED: 2018-11-06

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

t = 4
# +++++ monks +++++
#idx = index[t, 0:100]
#idy = index[t, 100:600]

# +++++ ionoshpere +++++
idx = index[t, 0:500]
idy = index[t, 500:4500]

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
# -------------------------------------------------------------------


print('Done !')






