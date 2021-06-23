# Shoud you use the primal or the dual form of the SVM problem to train a model on a training set with millions of instances and hundreds of features.

The QP problem of the primal problem would be hard to solve, as the matrix H would be very large (lots of features). The dual form can use the kernel trick to avoid this problem all together and has a dot product instead.

It's very clear that the dual form wins.
