# What is a fully convolutional network? How can you convert a dense layer into a convolutional layer?

A fully convolutional network is a CCN where the last layer is not a dense layer but a convolution layer. The model could be trained using a dense layer. The weights can be used to configure the new convolution layer at the end.

How to convert?

A dense layer has ever neuron connected to every output. If the previous layer was a CNN that means you have k filters, each of size n*n. This means you have k*n*n inputs connected to one neuron. A CNN at the end with a kernel of n*n with k filters, does exactly the same amount of calculations. Each neuron at the previous output is now a filter, use the same weights.

Now if larger images are provided the same weights still work, and it's like sliding the a window over the image -> YOLO.
