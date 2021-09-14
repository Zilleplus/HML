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
x_train, x_test, y_train,  y_test = x[:60000], x[60000:], y[:60000], y[60000:]

x_grid, y_grid = x[:3000], y[:3000]

neigh = KNeighborsClassifier()

param_grid = [
        {
            'weights': ['distance', 'uniform'],
            'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8]
        }]

grid_search = GridSearchCV(
        neigh,
        param_grid,
        verbose=3,
        cv=3,
        scoring='neg_mean_squared_error',
        return_train_score=True)

grid_search.fit(x_train, y_train)
grid_search.best_params_
best_est = grid_search.best_estimator_


# use test input on best_estimator
y_test_predict = best_est.predict(x_test)
r = y_test_predict == y_test
success_rate = sum(1 if e else 0 for e in r)/len(r)

# last run had a 0.9714% success rate
# with params {n_neighbors=4, weights=distance}
print("currently at success rate on test date of " + str(success_rate))
