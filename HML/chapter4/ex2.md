# Suppose the features in your training set have very different scales. Which algorithm might suffer from this, and how? What can you do about it?

Table at page 128

All Gradient descent algorithms have a problem with scaling. The learning rate determines the size of the step that will be taken. But is the same for each dimension (it's a scalar) If the features are not scaled, the size of the gradient of earch feature will be different. And the optimum in one dimension(feature) will be reached before the optimum in the other dimension(feature). Sklean provides a simpel scaler(StandardScaler) that can scale the amplitude of the features, and solves this problem.
