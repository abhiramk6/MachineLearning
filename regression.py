import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def regression_1(x, y):
    yi = np.array(y)
    xi = np.array(x)
    alpha_1 = len(x)
    alpha_2 = np.sum(xi * xi)
    beta_1 = np.sum(-2 * yi)
    beta_2 = np.sum(-2 * (yi * xi))
    zeta = np.sum(2 * xi)
    gamma = np.sum(yi * yi)
    A = np.array([[2 * alpha_1, zeta], [zeta, 2 * alpha_2]])
    B = np.array([[-beta_1], [-beta_2]])
    A_inverse = np.linalg.inv(A)
    W = np.matmul(A_inverse, B)
    W_0 = W[0]
    W_1 = W[1]
    return W_0, W_1


df1 = pd.read_excel(r'cost model.xlsx')
print('df1 = ', df1)
print(type(df1))

x = list(df1[:]['cost'])
print('x =', x)

y = list(df1[:]['sales'])
print('y =', y)

W_0, W_1 = regression_1(x, y)

x_p = [10.5, 11.5, 12.5, 13.5, 14.4, 15.4, 16.3, 17.2, 18.1, 18.9, 19, 19.8]
ii = 0
y_p = []
for i in x_p:
    y_p.append(W_0 + W_1 * i)

plt.scatter(x, y, color='red')  # plotting the observation line
plt.plot(x_p, y_p, color='blue')  ## model
plt.xlabel("cost")  # adding the name of x-axis
plt.ylabel("sales")  # adding the name of y-axis
plt.show()

