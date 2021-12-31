
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


data = sio.loadmat('./data/iono_ind.mat')
Data = data['Data']
index = data['index']
index = index.astype(int)
index -= 1

nDim, nSam = np.shape(Data)

snm = 0
sdu1 = 0
sdu2 = 0
n = 5
for t in range(n):
    idx = index[t, 0:70]
    idy = index[t, 70:3500]
    
    X = Data[:, idx]
    Y = Data[:, idy]
    
############################################################################
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
    
    nmse = tglokmm(X, tY, cY, 100, 500)
    print(nmse)
    
    nmse = itskmm(X, tY, cY, 100, 500, 5, 500)
    
    endtime1 = datetime.datetime.now()
    endtime2 = datetime.datetime.now().microsecond
    
    duration1 = (endtime1 - starttime1).microseconds
    duration2 = endtime2 - starttime2
    
    print(duration1)
    strs = 'duration2: ' + '%f' %duration2
    print(strs)
    
    snm += nmse
    sdu1 = sdu1 + duration1
    sdu2 += duration2


nmse = snm / n
duration1 = sdu1 / n
duration2 = sdu2 / n

print('------------------------------------------------------------')
print(nmse)
print(duration1)
print(duration2)
# -------------------------------------------------------------------


print('Done !')






