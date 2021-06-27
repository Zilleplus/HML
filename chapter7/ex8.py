import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

# split 1e4 for testing 6e4 for train and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000)

# Random forest clasifier:
# requires not scaling, so no pipeline here
# Takes to long to train, so I won't do any folding.
param_grid = {
    'n_estimators': [30, 40, 50, 100],  # higher is no better
    'max_depth': list(range(2, 5)),
    'min_samples_split': list(range(2, 5)),
    'min_samples_leaf': list(range(1, 5)),
}
forrest = RandomForestClassifier(max_depth=4, min_samples_split=3, oob_score=True)
search = GridSearchCV(forrest, param_grid, verbose=3,
                      cv=ShuffleSplit(n_splits=1, test_size=1000))
search.fit(X_train, y_train)
best_forrest = search.best_estimator_
print("Accuracy of best forrest {0}"
      .format(accuracy_score(y_test, best_forrest.predict(X_test))))
# best results:
# max_depth=4, min_samples_split=3, n_estimators=30

# extra-trees
param_grid_extra = {
    'n_estimators': [30, 40, 50, 100],
    'max_depth': list(range(2, 5)),
    'min_samples_split': list(range(2, 5)),
    'min_samples_leaf': list(range(1, 5))
}
extra_trees = ExtraTreesClassifier()
search = GridSearchCV(extra_trees, param_grid_extra, verbose=3,
                      cv=ShuffleSplit(n_splits=1, test_size=1000))
search.fit(X_train, y_train)
best_extra_trees = search.best_estimator_
print("Accuracy of best extra trees {0}"
      .format(accuracy_score(y_test, best_extra_trees.predict(X_test))))
# best results:
# max_depth=4, min_samples_split=3, n_estimators=30

# TODO::determine hyper parameters SVC
# this is a rather slow classifier... so takes a while.
# SVM:
param_grid = {
    'cf__kernel': ['linear', 'rbf', 'poly'],
    'cf__degree': [1, 2, 3, 4, 5],
    'cf__C': [0.5, 1.0, 1.5]
}
svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('cf', SVC())
])
search = GridSearchCV(svm_pipe, param_grid, verbose=3,
                      cv=ShuffleSplit(n_splits=1, test_size=1000))
search.fit(X_train, y_train)
best_svm = search.best_estimator_
print("Accuracy of best svc {0}"
      .format(accuracy_score(y_test, best_svm.predict(X_test))))

# In order to save some time you can skip the above and run:
forrest_manual = RandomForestClassifier(max_depth=4,
                                        min_samples_split=3,
                                        n_estimators=30)
forrest_manual.fit(X_train, y_train)
print("Accuracy of the manual forrest {0}"
      .format(accuracy_score(y_test, forrest_manual.predict(X_test))))
# accuracy of 0.7856

extra_trees_manual = ExtraTreesClassifier(max_depth=4,
                                          min_samples_split=3,
                                          n_estimators=30)
extra_trees_manual.fit(X_train, y_train)
print("Accuracy of the extra trees {0}"
      .format(accuracy_score(y_test, extra_trees_manual.predict(X_test))))
# accuracy of 0.7648

manual_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('cf', SVC())
])
manual_svm.fit(X_train, y_train)
print("Accuracy of the manual svm {0}"
      .format(accuracy_score(y_test, manual_svm.predict(X_test))))
# accuracy or 0.9645

voting_cf = VotingClassifier(
    estimators=[('extra', extra_trees_manual), ('forrest', forrest_manual), ('svm', manual_svm)],
    voting='hard')
voting_cf.fit(X_train, y_train)
print("Accuracy of the voting cf {0}"
      .format(accuracy_score(y_test, voting_cf.predict(X_test))))

voting_cf_only_trees = VotingClassifier(
    estimators=[('extra', extra_trees_manual), ('forrest', forrest_manual)],
    voting='hard')
voting_cf_only_trees.fit(X_train, y_train)
print("Accuracy of the voting cf {0}"
      .format(accuracy_score(y_test,  voting_cf_only_trees.predict(X_test))))
