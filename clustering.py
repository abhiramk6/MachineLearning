import pandas as pd
import os
import sys
import numpy as np


def error_k_means(X1, X2, C1, C2):
    # step1: assigning the points to centriods
    k = 3
    XP11 = []
    XP12 = []
    XP21 = []
    XP22 = []
    XP31 = []
    XP32 = []
    error1 = [0, 0, 0]
    err1 = 0
    err2 = 0
    err3 = 0

    for ii in np.arange(0, len(X1)):
        for i in np.arange(0, k):
            error1[i - 1] = (C1[i] - X1[ii]) ** 2 + (C2[i] - X2[ii]) ** 2
        min_val = min(error1)
        min_index = error1.index(min_val)
        if min_index == 0:
            err1 = err1 + min_val
        if min_index == 1:
            err2 = err2 + min_val
        if min_index == 2:
            err3 = err3 + min_val

    total_error = err1 + err2 + err3
    return total_error


def K_means_pgm(X1, X2, C1, C2):
    # step1: assigning the points to centriods
    k = 3
    XP11 = []
    XP12 = []
    XP21 = []
    XP22 = []
    XP31 = []
    XP32 = []
    error1 = [0, 0, 0]

    for ii in np.arange(0, len(X1)):
        for i in np.arange(0, k):
            error1[i - 1] = (C1[i] - X1[ii]) ** 2 + (C2[i] - X2[ii]) ** 2
        min_val = min(error1)
        min_index = error1.index(min_val)
        if min_index == 0:
            XP11.append(X1[ii])
            XP12.append(X2[ii])
        if min_index == 1:
            XP21.append(X1[ii])
            XP22.append(X2[ii])
        if min_index == 2:
            XP31.append(X1[ii])
            XP32.append(X2[ii])

    ## step 2 recomputing the centriods
    C1[0] = np.mean(XP11)
    C1[1] = np.mean(XP21)
    C1[2] = np.mean(XP31)

    C2[0] = np.mean(XP12)
    C2[1] = np.mean(XP22)
    C2[2] = np.mean(XP32)

    return C1, C2


data_km = pd.read_excel('Data_k_means.xlsx')
print(data_km)

X1 = list(data_km.loc[:, 'A1'])
X2 = list(data_km.loc[:, 'A2'])
print(X1)
print(X2)

### initializing the centriods
C1 = [3.8, 7.8, 6.2]
C2 = [9.9, 12.2, 18.5]

####
er_kmeans1 = error_k_means(X1, X2, C1, C2)
#### convergence
error_diff = 2
while error_diff > 0.0001:
    C1, C2 = K_means_pgm(X1, X2, C1, C2)
    er_kmeans2 = error_k_means(X1, X2, C1, C2)
    error_diff = er_kmeans1 - er_kmeans2
    er_kmeans1 = er_kmeans2

print(C1)
print(C2)

print(er_kmeans1)
print(er_kmeans2)