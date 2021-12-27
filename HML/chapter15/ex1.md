# Can you think of a few applications for a sequence-to-sequence RNN? What about a sequence-to-vector RNN, and a vector-to-sequence?

1. sequence-to-sequence: Translation system, sentence of 1 language is the input sequence, and the output sequence is the sentence in the ouput language.
2. sequence-to-vector: Time series and next n values (n beeing a fixed number so it's a vector not a sequence)
3. vector-to-sequence: Music generator, the input could be the values representing kinds of music, theme's ect... These parameters have a fixed length so it's a vector. The output is the music itself, which is not alway's the same length.
