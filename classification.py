import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Linear_classifier_LSM(x1,x2,y):
 y = np.array(y)
 x1 = np.array(x1)
 x1 = np.matrix(x1)
 x1 = x1.transpose()
 x2 = np.array(x2)
 x2 = np.matrix(x2)
 x2 = x2.transpose()
 on1 = np.ones(len(x1))
 on1 = np.matrix(on1)
 on1 = on1.transpose()
 X = np.hstack((on1,x1,x2))  ## Appending the column matrices tp form the matrix
 Y = np.matrix(y)
 Y = Y.transpose()
 XT = X.transpose()
 alpha = XT*X
 beta = XT*Y
 W = np.linalg.inv(alpha)*beta
 return W


df1 = pd.read_excel(r'Book1.xlsx')
print('df1 = ', df1)
print(type(df1))

y = list(df1[:]['Y'])
print('y =', y)

x1 = list(df1[:]['X1'])
print('x1 =', x1)

x2 = list(df1[:]['X2'])
print('x2 =', x2)

### modeling a multiple linear classifier
W = Linear_classifier_LSM(x1,x2,y)

print(W)

W = np.array(W)
#### plotting the function f(x1,x2) = W0 + W1X1 + W2X2
X_1 = np.linspace(0, 15, num=100) ### generation of samples
X_2 = []
for ii in X_1:
 element = -(W[0]+W[1]*ii)/W[2]
 X_2.append(element)


indx_class_1 = [i for i, n in enumerate(y) if n == 1]
indx_class_2 = [i for i, n in enumerate(y) if n == -1]

from operator import itemgetter
fig = plt.figure()
######### plotting original points
plt.scatter(list((itemgetter(*indx_class_1)(x1))),list((itemgetter(*indx_class_1)(x2))));
plt.scatter(list((itemgetter(*indx_class_2)(x1))),list((itemgetter(*indx_class_2)(x2))));

plt.plot(X_1,X_2)

plt.show()

