import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000)


def test(name: str, model, test_x, test_y):
    y_pred = model.predict(test_x)
    print(name, "accuracy ={0}".format(accuracy_score(test_y, y_pred)))
    return


def train(name: str, model, train_x, train_y):
    start = time.time()
    model.fit(train_x, train_y)
    end = time.time()
    print(name + ": execution time ={0}".format(end-start))


# Train forrest with raw data:
forrest = RandomForestClassifier()
# train: time = 39 seconds
train("without pca", forrest, X_train, y_train)
# test: accuracy = 96.79%
test("without pca", forrest, X_test, y_test)

# Train forrest with pca 95% variance first.
pca = PCA(n_components=0.95)
pca.fit(X_train, y_train)

X_train_reduced = pca.transform(X_train)
X_test_reduced = pca.transform(X_test)

forrest_pca = RandomForestClassifier()
train("with pca", forrest_pca, X_train_reduced, y_train)
# train: time = 114 seconds
test("with pca", forrest_pca, X_test_reduced, y_test)
# test: accuracy = 94.47%
