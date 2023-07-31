import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def Perceptron_neuron_linear_classifier(x1, x2, y, W):
    N = len(x1)
    for i in list(range(N)):
        v = W[0] + W[1] * x1[i] + W[2] * x2[i]
        if v < 0:
            si = -1
        else:
            si = 1
        W[0] = W[0] + 0.00001 * (y[i] - v) * 1
        W[1] = W[1] + 0.00001 * (y[i] - v) * x1[i]
        W[2] = W[2] + 0.00001 * (y[i] - v) * x2[i]

    # Computation of error of an iteration
    error_E = 0
    for i in list(range(N)):
        v = W[0] + W[1] * x1[i] + W[2] * x2[i]
        error_E = error_E + (y[i] - v) ** 2

    error_E = error_E / N
    return error_E, W


df1 = pd.read_excel(r'Book1.xlsx')
print('df1 = ', df1)
print(type(df1))

y = list(df1[:]['Y'])
print('y =', y)

x1 = list(df1[:]['X1'])
print('x1 =', x1)

x2 = list(df1[:]['X2'])
print('x2 =', x2)

### Randomly initialize the weight based on the dimensionality of input point
L = 2
import random

# seed random number generator
# random.seed(1)
W0 = random.random()
W1 = random.random()
W2 = random.random()
W = [W0, W1, W2]

i = 1
E_t = 0
Error_v = []
error_diff = 1000
while error_diff > 0.0001:
    E, W = Perceptron_neuron_linear_classifier(x1, x2, y, W)
    error_diff = abs(E_t - E)
    Error_v.append(E)
    E_t = E
    i = i + 1

print(i)

W = np.array(W)

## Plotting error for each iteration
plt.plot(Error_v)
plt.show

#### test case
xp1 = 7  # cost
xp2 = 33  # time of delivery
vi = W[0] + W[1] * xp1 + W[2] * xp2
if vi > 0:
    Si = 1
else:
    Si = -1

print(Si)

########### plotting
W = np.array(W)
#### plotting the function f(x1,x2) = W0 + W1X1 + W2X2
X_1 = np.linspace(0, 15, num=100)  ### generation of samples
X_2 = []
for ii in X_1:
    element = -(W[0] + W[1] * ii) / W[2]
    X_2.append(element)

indx_class_1 = [i for i, n in enumerate(y) if n == 1]
indx_class_2 = [i for i, n in enumerate(y) if n == -1]

from operator import itemgetter

fig = plt.figure()
######### plotting original points
plt.scatter(list((itemgetter(*indx_class_1)(x1))), list((itemgetter(*indx_class_1)(x2))));
plt.scatter(list((itemgetter(*indx_class_2)(x1))), list((itemgetter(*indx_class_2)(x2))));

plt.plot(X_1, X_2)

plt.show()

