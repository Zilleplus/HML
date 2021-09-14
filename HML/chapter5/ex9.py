# Train a SVM classfier on the MNIST dataset.
# Since SVM classifiers are binary classifiers you will need to use
# one-versus-the-rest to classify all the 10 digits.
# You may want to tune the hyperparameters using small validation sets to
# speed up the process. What accuracy can you reach?

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_predict,\
        StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

mnist = fetch_openml('mnist_784', version=1)

X, y = mnist["data"].to_numpy(), mnist["target"].to_numpy().astype(np.int32)
split_indices = StratifiedShuffleSplit(n_splits=1, test_size=0.1)\
    .split(X, y)
i_train, i_test = next(split_indices)
y_train = y[i_train]
y_test = y[i_test]
X_train = X[i_train]
X_test = X[i_test]
# The training data is so large it takes forever to do the validation.
# We select a part of the training data to do the validation.
_, i_train_valid = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2)
                        .split(X_train, y_train))

param_grid = {
        'cf__kernel': ['linear', 'rbf', 'poly'],
        'cf__degree': [1, 2, 3, 4, 5],
        'cf__gamma': ['scale', 'auto'],
        'cf__shrinking': [True, False]
}
pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('cf', SVC())
])
search = GridSearchCV(pipeline, param_grid,cv=StratifiedShuffleSplit(n_splits=1, test_size = 0.2) ,verbose=3)
search.fit(X_train[i_train_valid], y_train[i_train_valid])
best_estimator_search = search.best_estimator_
best_estimator_search

# The above search takes a while the best result is
# kernel=poly
# degree=2
best_est = Pipeline([
    ('scale', StandardScaler()),
    ('cf', SVC(kernel='poly', degree=2))
])
best_est.fit(X_train, y_train)
y_test_predict = best_est.predict(X_test)
success_rate_test = sum([1 if x == 0 else 0 for x in
                         (y_test_predict - y_test)])/len(y_test_predict)
print(("success rate on training set={:.0%}").format(success_rate_test))
# I could do some more analysis... but gonna continue to next chapter for now
