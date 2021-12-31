
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


data = sio.loadmat('./data/letter_ind.mat')
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
    idx = index[t, 0:500]
    idy = index[t, 500:4500]
    
    X = Data[:, idx]
    Y = Data[:, idy]
    
############################################################################
    starttime1 = datetime.datetime.now()
    starttime2 = datetime.datetime.now().microsecond
    
    m = ikmm(X, Y)
    
    #nmse = m.iglokmm(400)
    ##nmse = inmse(X, Y, P)
    
    nmse = m.iskmm(X, Y, 100, 50, 5)
    
    
    endtime1 = datetime.datetime.now()
    endtime2 = datetime.datetime.now().microsecond
    
    duration1 = (endtime1 - starttime1).microseconds
    duration2 = endtime2 - starttime2
    
    strs = 'duration1: ' + '%f' %duration1
    print(strs)
    strs = 'duration2: ' + '%f' %duration2
    print(strs)
    
    snm += nmse
    sdu1 = sdu1 + duration1
    sdu2 += duration2


nmse = snm / n
duration1 = sdu1 / n
duration2 = sdu2 / n

print(nmse)
print(duration1)
print(duration2)
# -------------------------------------------------------------------


print('Done !')






