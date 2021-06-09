import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# import matplotlib as mpl
# import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)

x, y = mnist["data"], mnist["target"]
# !!! Important to turn the strings into int's,
# otherwise the classfier will error.
y = y.astype(np.uint8)
x_train, x_test, y_train,  y_test = x[:60000], x[60000:], y[:60000], y[6000:]

neigh = KNeighborsClassifier()
param_grid = [
        {
            'weights': ['uniform', 'distance'],
            'n_neighbors': [5, 10, 20, 30]
        }]

grid_search = GridSearchCV(
        neigh,
        param_grid,
        verbose=3,
        scoring='neg_mean_squared_error',
        return_train_score=True)
grid_search.fit(x_train, y_train)
