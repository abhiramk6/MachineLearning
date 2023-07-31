import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics


def Gaussian_model_pdf(x1p, mean_S0, var_S0):
  likelihood_S0 = 1/(math.sqrt(2*math.pi*var_S0))*math.exp(-0.5*(1/var_S0)*(x1p-mean_S0)**2)
  return likelihood_S0

df1 = pd.read_csv(r'banknote.csv')
print('df1 = ', df1)
print(type(df1))

# Reading a RV
x1 = list(df1[:]['X1'])
print('x1 =', x1)

## class labels
S = list(df1[:]['S'])
print('S =', S)

# selecting the set of points of class 0 and class 1
indx_class_0 = [i for i, n in enumerate(S) if n == 0]
indx_class_1 = [i for i, n in enumerate(S) if n == 1]

print(indx_class_0)
print(indx_class_1)

from operator import itemgetter
x1_class0 = list((itemgetter(*indx_class_0)(x1)))
x1_class1 = list((itemgetter(*indx_class_1)(x1)))

### Gaussian pdf parameters
mean_S0 = statistics.mean(x1_class0)
variance_S0 = statistics.variance(x1_class0)
mean_S1 = statistics.mean(x1_class1)
variance_S1 = statistics.variance(x1_class1)


## weightage probabilities
P_S0 = len(x1_class0)/len(x1)
P_S1 = len(x1_class1)/len(x1)

## unknown sample xp1
x1p = 0
likelihood_S0 = Gaussian_model_pdf(x1p,mean_S0, variance_S0)
likelihood_S1 = Gaussian_model_pdf(x1p,mean_S1, variance_S1)

## Aposteriori probabilities
aposteriori_S0 = likelihood_S0*P_S0
aposteriori_S1 = likelihood_S1*P_S1

aposteriori_S0_org = aposteriori_S0/(aposteriori_S0+aposteriori_S1)
aposteriori_S1_org = aposteriori_S1/(aposteriori_S0+aposteriori_S1)

print(aposteriori_S0_org)
print(aposteriori_S1_org)

print(P_S0)
print(P_S1)



