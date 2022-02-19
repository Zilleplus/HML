# Suppose you have a daily univariate time series, and you want to forecast the next seven day's. Which RNN architecture should you use?

It's univariate so we only have 7 outputs, we need to use a sequence to vector architecture.

- We could use a wavenet architecture, and create a few 1DConv layers with increasing dilation. And then put a layer with seven filters at the end. This is the preferred choice, as the convolutional network is easier to parallelize. And it's not recurrent so it has less gradient issues, and will converge more easily.
- We could use a similar architecture from the book with a few SimpleRNN layers. It's sequence to vector so the layers have return_sequences=True except for the last one.
