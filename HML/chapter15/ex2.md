# How many dimensions must the inputs of an RNN layer have? What does each dimension represent? What about it's output?

1. Input:  (batch_size, sequence_length, input_dimension)
2. Output(with return_sequence=True): (batch_size, sequence_length, number_of_neurons)
