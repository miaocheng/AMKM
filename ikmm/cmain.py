
#!/usr/bin/env python
'''

CREATED: 2018-11-06

'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from cala import *
from skmm import *

#import time
import datetime
import random


data = sio.loadmat('./data/cover_ind.mat')
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
for t in range(n):
    idx = index[t, 0:500]
    idy = index[t, 500:3500]
    idc = index[t, 3500:7500]
    
    X = Data[:, idx]
    tY = Data[:, idy]
    cY = Data[:, idc]
    
############################################################################
    starttime1 = datetime.datetime.now()
    starttime2 = datetime.datetime.now().microsecond
    
    m = skmm(X, tY, cY, 100, 500)
    
    
    #nmse, cost = m.tenkmm(5)
    
    #nmse, cost = m.tglokmm()
    #print(nmse)
    
    
    nmse, cost = m.itskmm(100, 50, 5)
    #print(nmse)
    
    
    endtime1 = datetime.datetime.now()
    endtime2 = datetime.datetime.now().microsecond
    
    duration1 = (endtime1 - starttime1).microseconds
    duration2 = endtime2 - starttime2
    
    strs = 'duration1: ' + '%f' %duration1
    print(strs)
    strs = 'duration2: ' + '%f' %duration2
    print(strs)
    
    snm += nmse
    sti += cost
    sdu1 += duration1
    sdu2 += duration2


snm = snm / n
sti = sti / n
duration1 = sdu1 / n
duration2 = sdu2 / n

print('-------------------------------------------------')
print('-------------------------------------------------')
print('The final results: ')
print(snm)
print('*********** time *************')
print(sti)
print('-----------------------------')
print(duration1)
print(duration2)
# -------------------------------------------------------------------


print('Done !')






