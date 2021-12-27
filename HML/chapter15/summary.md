# Processing Sequences using RNNS and CNNS

## Recurrent neural layers

The outputs of a recurrent layer for a single instance:

y(t) = activation_function(Wx^T x_{t} + Wy^T y_{t-1} + b)

The outputs of a recurrent layer for a mini-batch:

y(t) = activation_function(X_tWx^T + Y_{t-1}Wy + b) = activation_function([X_t Y_{t-1}]W + b) with W = [W_x;W_y]

- m: batch size
- Wx: n_input * n_neurons
- Wy: n_neurons * n_neurons
- x_t: m * n_input
- y_t: m * n_neurons
- b: n_neurons * 1

So we know that:
- Every neuron takes all of the inputs
- Every neuron takes all of the other inputs of a previous timestamp

## Memory Cells
- A part of a neural network that preserves some state across time steps is called a memory cell.
- A neuron in practice has about a memory of 10 timestamps. As everytime the information gets mixed with y, a bit of information is lost.
- If you want memory longer then 10 timestamps, you need to use hidden layers.

## Input Output Sequence
- Sequence-to-sequence network: input sequence length is same as output sequence length
- sequence-to-vector network(decoder): fixed number of outputs, input sequence could be longer/shorter.
- vector-to-sequence network(encoder): fixed number of inputs, output is sequence could be longer/shorter.
- encoder+decoder: from sequence to vector and back to sequence (for example translations, more in chapter 16)

## Training RNN

Just like with regular Neural networks backward propagation is used to find the optimal parameters. Obviously the gradient now travels through time instead of space, so we call it BPTT: backpropagation through time.

## Forecasting
3 major types:

1. Univariate time series: only 1 output parameter
2. Multivariate time series: multiple output parameters
3. Imputation: Filling in data, don't predict the future, but predict missing values in existing data.

## Deep RNN

There are 2 simple way's to predict multiple samples:

1. Predict one value, concat it to the input values, and predict the next one. This will accumulate errors, and will not be very efficient.
2. Make a model with N outputs, The the best results come from changing the model into a sequence to sequence model. The output layer (typical dense layer) must be applied exactly the same way to each output. (so only have weights for one output). In tensorflow this is accomplished  by putting "keras.leyers.TimeDistributed(...)" around the output layer. Tensorflow requires you to enable return_sequences to get a sequence out, otherwise it will only output the last value after that layer. And it needs to put the output at every timestep onto the next layer. If return_sequences is on, you want to replace the mse metric with one that only works on the last element.

## Fighting unstable gradient

1. Use tanh function, as ReLU may be unstable during training. (As the same weights are used at every timestep, ReLU will blow up the value or kill off the value, leading to unstable behavior when training)
2. Use dropout
3. use LayerNormalization: Normalizes over the time series itself, and not over the batch. Batch normalization doesn't work here (except on the input of the whole model once).

## Tackling the Short-Term memory Problem
Each step in the RNN loses some information, this limits the practical length of the system. The book mentions 2 alternatives to the classical node:

1. LSTM: Long Short Timer Memory: page 514
2. GRU: Gated recurrent unit: page 518

They are both optimized to run very quickly on gpu's, and have a much longer memory.

## 1D Convolution layers
Exactly as with 2D convolution layers, you can have 1D convolution layers. Using kernels, padding, dilation rate and strides parameters just like with the 2D case.

## WaveNet
By doubling the dilation rate, the first layers will only look locally, while the final layers will look globally. As this makes the whole network quiet sparse, this speeds up learning by a lot, and it has state of the art performance.

```
keras.model.Sequential([
    keras.layers.Conv1D(dilation_rate=2, ...),
    keras.layers.Conv1D(dilation_rate=4, ...),
    keras.layers.Conv1D(dilation_rate=8, ...),
    keras.layers.Conv1D(dilation_rate=16, ...),
])
```
