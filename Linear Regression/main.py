import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error


# using one of the built-in datasets of sklearn
diabetes = datasets.load_diabetes()

# use diabetes.keys() for its methods output => ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module']
# print(diabetes.data)

# printing the line of regression
diabetes_X = diabetes.data[:,np.newaxis,2]      # ~~~~~~~~~~~~~~~~~~~~~~ taking only second index feature (means only on feature). To take all the features just remove -> [:,np.newaxis,2]  ~~~~~~~~~~~~~~~~~~~~~~~~

diabetes_X_train = diabetes_X[:-30]             # taking last 30 for training       # X axis for features
diabetes_X_test = diabetes_X[-30:]              # taking first 30 for testing

diabetes_Y_train = diabetes.target[:-30]        # values should be same as above        # y axis for label
diabetes_Y_test = diabetes.target[-30:] 

# making linear regression model
model = linear_model.LinearRegression()

model.fit(diabetes_X_train,diabetes_Y_train)     # creating the line

# testing the model
diabetes_Y_predicted = model.predict(diabetes_X_test)

print("Mean Squared Error is :", mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))      # (actual,predicted)

print("Weights :",model.coef_)
print("Intercept :",model.intercept_)

# plotting on graph (if taking all the features this will throw error, we can not plot all the features)
plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_predicted)

plt.show()