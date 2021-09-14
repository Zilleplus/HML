# Train a linearSVC on a linearly separable dataset.
# Then train an SVC and a SGDClassfiier on the same dataset.
# See if you can get them to produce roughtly the same mode.
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

X, _ = make_moons(n_samples=100, noise=0.15)
# must be lin-sep:
y = [0 if x[1] > 0.25 else 1 for x in X]
colors = ['#1f77b4' if x == 0 else '#e377c2' for x in y]
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.show()


param_grid_linsvc = {
        'svc__loss': ['hinge', 'squared_hinge'],
        'svc__C': [0.25, 0.5, 1.0, 2.0, 4.0],
}
lin_svc_pipe = Pipeline([
    ("Scaler", StandardScaler()),
    ("svc", LinearSVC())
    ])
lin_svc_search = GridSearchCV(lin_svc_pipe, param_grid_linsvc, verbose=3)
lin_svc_search.fit(X, y)
y_pred_lin_svc = lin_svc_search.best_estimator_.predict(X)
colors = ['#1f77b4' if x == 0 else '#e377c2' for x in y_pred_lin_svc]
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.show()

# Grid search puts degree on 1, and we get same results as the linear one.
param_grid_svc = {
        'svc__C': [0.5, 1.0, 2.0],
        'svc__degree': [1, 2, 3, 4, 5],
        'svc__gamma': ['scale', 'auto'],
}
svc_pipe = Pipeline([
    ("Scaler", StandardScaler()),
    ("svc", SVC())
    ])
svc_search = GridSearchCV(svc_pipe, param_grid_svc, verbose=3)
svc_search.fit(X, y)
y_pred_svc = svc_search.best_estimator_.predict(X)
colors = ['#1f77b4' if x == 0 else '#e377c2' for x in y_pred_svc]
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.show()

# Grid search puts degree on 1, and we get same results as the linear one.
# the perceptron loss function wins (linear one, makes sence)
param_grid_sgd = {
        'sgd__loss': ['perceptron', 'log', 'squared_hinge']
}
sgd_pipe = Pipeline([
    ("Scaler", StandardScaler()),
    ("sgd", SGDClassifier())
    ])
sgd_search = GridSearchCV(sgd_pipe, param_grid_sgd, verbose=3)
sgd_search.fit(X, y)
y_pred_sgd = sgd_search.best_estimator_.predict(X)
colors = ['#1f77b4' if x == 0 else '#e377c2' for x in y_pred_sgd]
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.show()
