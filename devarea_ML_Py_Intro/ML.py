import numpy as np
import sklearn.linear_model as skl
import pylab as py
import pandas as pd
import seaborn as sb

model = skl.LinearRegression()

# Data with no error
xval = np.array([1,2,3,4,5]).reshape(-1,1)
yval = [1,2,3,4,5]

model.fit(xval, yval)

print model.predict(12)
print model.predict(44)

# Introduce some errors in the data
xval = np.array([1,2,3,3,4,3,6,8,9,10]).reshape(-1,1)
yval = [1,2,3,4,5,6,7,7,9,10]
model.fit(xval,yval)

print model.predict(12)
print model.predict(44)

py.scatter(xval,yval)


# Introduce more dimensions
samp=np.array([[1,2,300,14],
               [9,3,1,95],
               [5,7,11,58],
               [4,8,14,57],
               [2,1,2,27],
               [9,9,7,100],
               [12,3,21,126],
               [29,12,3,309],
               [2,40,11,90],
               [21,32,4,270],
               [7,13,8,79],
               [17,2,19,172],
               [13,24,13,159]])
df=pd.DataFrame(samp, columns=['X','Y','Z','W'])

sb.pairplot(df)

# Split the data
#   Only the first 8 rows for training
xval=df[:8][['X','Y','Z']]
yval=df[:8][['W']]
model.fit(xval,yval)

# Let's see some predictions
model.predict(df[8:][['X','Y','Z']])
model.coef_

