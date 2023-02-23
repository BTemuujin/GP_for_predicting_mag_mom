import numpy as np
import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn import *

data = np.loadtxt('GP_train.txt')
test = np.loadtxt('GP_test.txt')

# Load your data into X (number of neighbors) and y (positions)
X = data[:, :-1]# Your array of number of neighbors
y = data[:, -1]# Your array of positions
#y = y.reshape(-1,1)
#print(X)
#print(X.shape)
#print(y)

# Define the GP kernel
#kernel = GPy.kern.RBF(input_dim=20, variance=1, lengthscale=1)
#kernel = GPy.kern.RBF(input_dim=20)

kernel = C(1.0, (0.1, 10.0)) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))

# Train the GP model
#model = GPy.models.GPRegression(X, y, kernel)
#model.optimize()

model = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
model.fit(X, y)

# Make predictions for new number of neighbors
X_new = test[:]
#X_new = X_new.reshape(-1, 1)
#X_new = X_new.transpose()
#print(X_new[:, None])
#X_new = X_new.reshape(48, 20) # Your array of new number of neighbors to predict on
#y_pred, sigma = model.predict(X_new[:, None])

y_pred, sigma = model.predict(X_new, return_std=True)

print(sigma.reshape(-1,1))
