# Why would you want to use 1D Convolutional layers in an RNN.
A convolutional neural network has less connections, hence it has less parameters, hence it's faster to train.

The convolutional neural network is easier to parallelize as every output element depends only on a limited number of input values. It also has no memory.

It's not recurrent so it suffers less from the unstable gradient problem.
