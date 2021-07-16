# Suppose you have an MLP composed of one input layer with 10 passthrough neurons, followed by one hidden layer with 50 Artificial neurons, and finaly one output layer with 3 artificial neurons. All artificial neurons use the ReLU activation function.

model = 10 passtrough | 50 hidden | output layert (3 neurons)

h(X) = act(XW+B)

- What is the shape of the input matrix X?
One row per instance
One column per feature = 10 features

so the shape is [number_of_instances*number_of_features]=[number_of_instances*10]

- What are the shapes of the hidden layer's weight vector W_h and its basic vector b_0?

We get vector of 10 elements as input, and we have 50 neurons so 50 outputs.

X=[batch_size*10]
W_h = [10*50]
In the text b is set as a vector, but if the batch size is larger then 1. This won't work.
b = [batch_size*50] -> the appendix say's [50] but that means we need to do "broadcasting", which is not a normal lin-alg operation. So I used a matrix here, works just fine.

- What are the shapes of the output layer's weight vector W_0 and its bias vector b_0?

We get vector of 50 elements as input, and we have 3 neurons so 3 outputs.
X=[batch_size*50]
W_0 = [50*3]
b = [batch_size*3]

- What is the shape of the network's output matrix Y?
Y = [batch_size*3]

- Write the equation that computes the network's output matrix Y as a function of X, W_h, b_h, W_o and b_o
Y = R(R(XW_h+B_h)*W_0+B_0)
