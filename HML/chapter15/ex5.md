# What are the main difficulties when training RNNs? How can you handle them?
2 problems:
1. Short memory 
2. Exploding gradient

Metigate Short memory:
- use LSTM or GRU layers.

Metigate Exploding gradient:
- If you use ReLU, you will get gradient issues's, so use tanh
- You can only batch normalize the series at the input, you can't do it in between the layers. But you can normalize over the feature space. This is called LayerNormalization
- If the gradient is unstable you can use dropout.
