# Growing a forest by following the following steps:

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

number_of_trees = 100

X, y = make_moons(n_samples=10000, noise=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

group_indices = ShuffleSplit(n_splits=number_of_trees, train_size=100, test_size=1)\
    .split(X_train, y_train)
classf = []
succ_rates = []
total_test_pred = [0]*len(y_test)
for train, _ in group_indices:
    X_partial = X[train]
    y_partial = y[train]
    cf = DecisionTreeClassifier(max_depth=7, min_samples_leaf=5)
    _ = cf.fit(X_partial, y_partial)
    y_pred_test = cf.predict(X_test)
    succ_rate = sum([1 if x == 0 else 0 for x in (y_pred_test - y_test)])\
        / len(y_pred_test)
    classf.append(cf)
    succ_rates.append(succ_rate)
    for i in range(0, len(y_pred_test)):
        total_test_pred[i] = total_test_pred[i] + y_pred_test[i]

average_succ_rate = sum(succ_rates)/len(succ_rates)
print("The average success rate of trees is {0:.3f}".format(average_succ_rate))
preds_forest = [0 if (float(x)/number_of_trees) <= 0.5 else 1 for x in total_test_pred]
preds_forest = sum([1 if x == 0 else 0 for x in (preds_forest - y_test)])\
        / len(preds_forest)
print("The random forest has an accuracy of {0:.3f}%".format(preds_forest))
