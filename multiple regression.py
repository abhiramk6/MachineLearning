import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Multiple_linear_regression_1(x1,x2,y):
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

df1 = pd.read_excel(r'excel regression multiple.xlsx')
print('df1 = ', df1)
print(type(df1))

y = list(df1[:]['sales'])
print('y =', y)

x1 = list(df1[:]['cost'])
print('x1 =', x1)

x2 = list(df1[:]['time'])
print('x2 =', x2)

### modeling a multiple linear regression
W = Multiple_linear_regression_1(x1,x2,y)

#### getting a value for the unknown sample
x_unknown = [88, 21]
pred_x_unknown_sales = W[0] + W[1]*x_unknown[0] + W[2]*x_unknown[1]
print(pred_x_unknown_sales)

#### plotting the function f(x1,x2) = W0 + W1X1 + W2X2
X_1 = np.linspace(68, 175, num=50) ### generation of samples
X_2 = np.linspace(8, 23, num=50) ### generation of samples

X_k1, X_k2 = np.meshgrid(X_1, X_2)

W = np.array(W)
Z = W[0] + W[1]*X_k1 + W[2]*X_k2

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X_k1, X_k2, Z, color='black')
ax.set_title('wireframe');
#plt.show()

######### plotting original points
ax.scatter(x1,x2,y, cmap='viridis', linewidth=1.5);
plt.show()
