# If you want to build a deep sequence-to-sequence RNN, which RNN layers should have return_sequences=True? What about a sequence-to-vector RNN?

1. Sequence-to-sequence: At the very least the last one should have return_sequences=True, otherwise we return a vector instead of a sequence. The most obvious architecture is to put on all layers return_sequences=True
2. Sequence-to-vector: Any but the last one, as we must return a vector, and not a sequence.
