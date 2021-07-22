# In which cases would you want to use each of the following activation functions:
- SELU: When you don't want to normalize in neural net. (this self-normalizes) Not usable in nonsequential architectures.
- leaky ReLu (and its variants): When things need to be fast, and you don't mind tuning an extra parameter.
- ReLU: When things need to be very fast
- tanh: alternative to ReLu, often better results the ReLu and faster then logisitic. (has output [-1;1] so if that is the wanted range it's a good option)
- logistic: good option if you want probabilities, but rarely used in practice.
- softmax: On the output layer of the neural net, to get nice probabilities on the output. (all add up to 1)
