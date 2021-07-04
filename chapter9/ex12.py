import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

data_set = fetch_olivetti_faces()
X, y = data_set.data, data_set.target


def visualize(img):
    img = np.reshape(img, (64, 64))
    plt.imshow(img)
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# TODO:: generate new faces using the sample method (inverse_transformation)
# -> visualize these new faces.
pca = PCA(n_components=0.99)
pca.fit(X_train)
X_train_transformed = pca.transform(X_train)

gm = GaussianMixture(n_components=40, random_state=42)
gm.fit(X_train_transformed)

faces = list(gm.sample(n_samples=1))
visualize(pca.inverse_transform(faces[0]))

# anomaly detection:
gm.score_samples(np.array(faces[0]))
# Score is about 1191 which is good,
# A bad score would be like 1e7 orso.
