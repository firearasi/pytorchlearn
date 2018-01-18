# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

with open('data.txt','r') as f:
    data_list=f.readlines()
    data_list=[f.split('\n')[0] for f in data_list]
    data_list=[f.split(',') for f in data_list]
    data_list=[[float(f[0]),float(f[1]),float(f[2])] for f in data_list]

class0 = [a for a in data_list if a[2]==0]
class1 = [a for a in data_list if a[2]==1]

plt.plot([c[0] for c in class0], [c[1] for c in class0], 'ro', label='c0')
plt.plot([c[0] for c in class1], [c[1] for c in class1], 'bo', label='c1')
plt.legend(loc='best')
plt.title('Original labels')
plt.show()

