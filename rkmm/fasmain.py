
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


data = sio.loadmat('./data/fashion_ind.mat')
train = data['train']
test = data['test']

idx = data['idx']
idy = data['idy']
idx = idx.astype(int)
idx -= 1
idy = idy.astype(int)
idy -= 1


snm = 0
sdu1 = 0
sdu2 = 0
n = 5
for t in range(n):
    indx = idx[t, 0:500]
    indy = idy[t, 0:7000]
    
    X = train[:, indx]
    Y = test[:, indy]
    
############################################################################
    starttime1 = datetime.datetime.now()
    starttime2 = datetime.datetime.now().microsecond
    
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
    
    endtime1 = datetime.datetime.now()
    endtime2 = datetime.datetime.now().microsecond
    
    duration1 = endtime1 - starttime1
    duration2 = endtime2 - starttime2
    
    print(duration1)
    strs = 'duration2: ' + '%f' %duration2
    print(strs)
    
    snm += nmse
    #sdu1 = sdu1 + duration1
    sdu2 += duration2


nmse = snm / n
#duration1 = sdu1 / n
duration2 = sdu2 / n

print(nmse)
#print(duration1)
print(duration2)
# -------------------------------------------------------------------


print('Done !')






