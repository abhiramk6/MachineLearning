import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def polynomial_ridge_regression_1(x,y, M, lmbda):
 y = np.array(y)
 Y = np.matrix(y)
 Y = Y.transpose()
 x_org = np.array(x)
 x = np.array(x)
 x = np.matrix(x)
 x = x.transpose()
 on1 = np.ones(len(x))
 on1 = np.matrix(on1)
 on1 = on1.transpose()
 X = np.hstack((on1,x))  ## Appending the column matrices tp form the matrix
 if M>1:
   m_list = []
   for i in np.arange(2,M+1):
     m_list.append(i)
   for ik in m_list:
     x_org_m = x_org**ik
     x_org_mat = np.matrix(x_org_m)
     x_org_mat = x_org_mat.transpose()
     X =np.hstack((X,x_org_mat))
 XT = X.transpose()
 alpha = XT*X
 Id = np.identity(M+1, dtype = float)
 Idl = Id*lmbda
 alpha1 = alpha - Idl
 beta = XT*Y
 W = np.linalg.inv(alpha1)*beta
 return W


def Error_computation(x,y,W, M):
  ## reading each value of x for getting f(x)
 y = np.array(y)
 Y = np.matrix(y)
 Y = Y.transpose()
 x_org = np.array(x)
 x = np.array(x)
 x = np.matrix(x)
 x = x.transpose()
 on1 = np.ones(len(x))
 on1 = np.matrix(on1)
 on1 = on1.transpose()
 X = np.hstack((on1,x))  ## Appending the column matrices tp form the matrix
 if M>1:
   m_list = []
   for i in np.arange(2,M+1):
     m_list.append(i)
   for ik in m_list:
     x_org_m = x_org**ik
     x_org_mat = np.matrix(x_org_m)
     x_org_mat = x_org_mat.transpose()
     X =np.hstack((X,x_org_mat))
 Y_p = X*W
 E_1 = Y-Y_p
 E_1 = np.array(E_1)
 E = sum(E_1**2)
 E = E/len(x)
 return E


df1 = pd.read_csv(r'sports.csv')
#print('df1 = \n', df1)
#print(type(df1))

x = df1.iloc[:,0].values  #independent variable vector
y = df1.iloc[:,1].values  #dependent variable array

### creating training data and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=1/3,random_state=0)

lmbda = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009] ## order of the polynomial regression
M = 9

E_training = []
E_testing = []
for im in lmbda:
 W = polynomial_ridge_regression_1(X_train,y_train, M, im)
 ## Computing the training error
 E_tr = Error_computation(X_train,y_train, W, M)
 E_training.append(E_tr)
 ## Computing the testing error
 E_ts = Error_computation(X_test,y_test, W, M)
 E_testing.append(E_ts)


print('E_tr =', E_training)
print('E_ts =', E_testing)

plt.plot(lmbda,E_training, color='blue') ## model
plt.plot(lmbda,E_testing, color='red') ## model
plt.xlabel("Lambda (Regularization parameter)") # adding the name of x-axis
plt.ylabel("Error") # adding the name of y-axis
plt.show()