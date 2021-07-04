import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time as t

data_set = fetch_olivetti_faces()
X, y = data_set.data, data_set.target


def visualize(img):
    img = np.reshape(img, (64, 64))
    plt.imshow(img)
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Don't use pipeline on purpose so we can see what's going on with the data.
km = KMeans(n_clusters=122)
km.fit(X_train, y_train)

X_train_transformed = km.predict(X_train)
X_train_transformed = np.reshape(X_train_transformed,
                                 (len(X_train_transformed), 1))

rf = RandomForestClassifier()
start_time = t.time()
rf.fit(X_train_transformed, y_train)
end_time = t.time()
X_test_transformed = km.predict(X_test)
X_test_transformed = np.reshape(X_test_transformed,
                                (len(X_test_transformed), 1))
y_pred_test = rf.predict(X_test_transformed)
acc_test = accuracy_score(y_test, y_pred_test)
print("The accuracy on the test data set = {0:0.3f} with excution time {1:0.3f}"
      .format(acc_test, (end_time - start_time)))
# about 0.2 seconds

rf_slow = RandomForestClassifier()
start_time = t.time()
rf_slow.fit(X_train, y_train)
end_time = t.time()
y_pred = rf_slow.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)
print("The accuracy on the test data set = {0:0.3f} with excution time {1:0.3f}"
      .format(acc_test, (end_time - start_time)))
# about 2.1 seconds, over time slower, but the k-means does take quiet some time.
# accuracy is both 0.887

# let's try to combine these two approaches:
X_train_combined = np.concatenate((X_train_transformed, X_train), axis=1)
rf_combined = RandomForestClassifier()
rf_combined.fit(X_train_combined, y_train)

X_test_combined = np.concatenate((X_test_transformed, X_test), axis=1)
y_test_combined_predict = rf_combined.predict(X_test_combined)
acc_test = accuracy_score(y_test, y_test_combined_predict)
print("The accuracy on the test data set = {0:0.3f}"
      .format(acc_test))
# accuracy is 0.938 -> significant improvement !!!
# I did not tune the cluster sizes, as this takes
# way to long on my current laptop...
# It's very suprising that the combined approach is that much better !
