import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random


def MLR_single_neuron(x1,x2,y,W):
    N=len(x1)
    for i in range(1,N):
        v=w0 + w1*x1[i] + w2*x2[i]
        wo = wo - 0.0001*(y[i]-v)*1
        w1 = w1 - 0.0001 * (y[i] - v) *x1[i]
        w2 = w2 - 0.0001 * (y[i] - v) *x2[i]

    error_E=0


    for i in range(1, N):
        v=w0 + w1*x1[i] + w2*x2[i]

    return error_E

df1 = pd.read_excel(r'excel regression multiple.xlsx')
print('df1=', df1)
print(type(df1))

y = list(df1[:]['Sales'])
print('y =', y)

x1 = list(df1[:]['Cost'])
print('x1 =', x1)

x2 = list(df1[:]['Time'])
print('x2 =', x2)

L=2

w0=random()
w1=random()
w2=random()
W=[w0,w1,w2]

E=MLR_single_neuron(x1,x2,y,W)

