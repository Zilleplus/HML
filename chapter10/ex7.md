# How many neurons do yo uneed in the output layer if you want to classfiy email into spam or ham. What activation function should you use in the output layer. If instead you want to tackle MNIST, how many neurons do you need in the ouput layer and which activation function should you use? What about gor getting your network to predict housing prices, as in Chapter2.

- Spam/Ham
Just 1 output neuron is enough, use ReLU function so you get either 0 or 1. (eg. spam or ham)

- MNIST
10 output neurons, 1 for each number. And you should use the softmax activation function on the output.
