# Train and fine-tune a Decision Tree for the oons dataset
# by following these steps:
# a. Use make_moons(n_samples=10000, noise=0.4)
#
# b. Use train_test_split() to split the datset in to a training
# set and test set.
#
# c. Use grid search with cross-validation (with the help of the
# GridSearchCV class) to find good hyperparameter values for a
# DecisionTreeClassifier
#
# d. Train it on the ful training set using these hyper paremeters,
# and measure your model's performance on the test set.
# You should get roughtly 85% to 87% accuracy.

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=10000, noise=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

param_grid = {
    'cf__max_depth': list(range(2, 10)),
    'cf__min_samples_split': list(range(2, 10)),
    'cf__min_samples_leaf': list(range(1, 10)),
    'cf__min_weight_fraction_leaf': [0.0, 0.1, 0.2]
}
pipeline = Pipeline([
    ('cf', DecisionTreeClassifier())
])
search = GridSearchCV(pipeline, param_grid, verbose=3)
search.fit(X_train, y_train)

best_estimator = search.best_estimator_
# It runs pretty quick, you get the result:
# max_depth=7, min_samples_leaf=5
best_estimator

y_pred_test = best_estimator.predict(X_test)
succ_rate = sum([1 if x == 0 else 0 for x in (y_pred_test - y_test)])\
    / len(y_pred_test)
succ_rate
