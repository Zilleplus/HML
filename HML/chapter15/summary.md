# Processing Sequences using RNNS and CNNS

## Recurrent neural layers

y(t) = activation_function(Wx^T x(t) + Wy^T y(t-1) + b)

- m: batch size
- Wx: n_input * n_neurons
- Wy: n_neurons * n_neurons
- x_t: m * n_input
- y_t: m * n_neurons
- b: n_neurons * 1

So we know that:
- Every neuron takes all of the inputs
- Every neuron takes all of the other inputs
- A neuron in practice has about a memory of 10 timestamps. As everytime the information gets mixed with y, a bit of information is lost.
- If you want memory longer then 10 timestamps, you need to use hidden layers.

## Training RNN

Just like with regular Neural networks backward propagation is used to find the optimal parameters. Obviously the gradient now travels through time instead of space.

## Forecasting
3 major types:

1. Univariate time series: only 1 output parameter
2. Multivariate time series: multiple output parameters
3. Imputation: Filling in data, don't predict the future, but predict missing values in existing data.

## Deep RNN

There are 2 simple way's to predict multiple samples:

1. Predict one value, concat it to the input values, and predict the next one. This will accumulate errors, and will not be very efficient.
2. Make a model with N outputs, The the best results come from changing the model into a sequence to sequence model. The output layer (typical dense layer) must be applied exactly the same way to each output. (so only have weights for one output). In tensorflow the can be accomplished  by putting "keras.leyers.TimeDistributed(...)" around the output layer.

! Tensorflow requires you to enable return_sequences to get multiple out values, otherwise it will only output the last value. !

## Fighting unstable gradient

1. Use dropout
2. use LayerNormalization: Normalizes over the time series itself, and not over the batch. Batch normalization doesn't work here (except on the input of the whole model once).

## Tackling the Short-Term memory Problem
Each step in the RNN loses some information, this limits the practical length of the system. The book mentions 2 alternatives to the classical node.

1. LSTM: Long Short Timer Memory
2. GRU: Gated recurrent unit

They are both optimized to run very quickly on gpu's, and have a much longer memory.

## 1D Convolution layers
TODO

## WaveNet
TODO
