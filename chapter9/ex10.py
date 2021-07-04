import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data_set = fetch_olivetti_faces()
X, y = data_set.data, data_set.target


def visualize(img):
    img = np.reshape(img, (64, 64))
    plt.imshow(img)
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

n_clusters = []
silhs = []
inertias = []
for i in range(2, 140):
    km = KMeans(n_clusters=i)
    _ = km.fit(X_train)
    score = silhouette_score(X_train, km.labels_)
    n_clusters.append(i)
    silhs.append(score)
    inertias.append(km.inertia_)
    print("n_clusters={0} has silhouette_score of {1:.3f}\
          and intertia of {2:3.3f}"
          .format(i, score, km.inertia_))

plt.plot(n_clusters, inertias)
plt.show()

plt.plot(n_clusters, silhs)
plt.show()

optimal_clusters_index = np.argmax(silhs)
print("The optimal silhouette score{0} wich is with {1} clusters"
      .format(silhs[optimal_clusters_index],
              n_clusters[optimal_clusters_index]))
# 122 seems to be the optimal number of clusters
# the interia is just a smooth curve, to elbow, so we can't really use it.
