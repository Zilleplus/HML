# If you gpu runs out of memory while training a CNN, what are five things you  could try to solve the problem?

my guesses:

1. Smallers filters (but more filters/layters?)
2. Add maxpool between layers
3. less filters
4. less layers
5. set the padding to valid (instead of same)

answers from the back of the book:

1. Reduce the mini-batch size.
2. Reduce dimensionaliry using a larger stride in one or more layers.
3. Remove one or more layers.
4. Use 16-bit floats instead of 32-bit floats.
5. Distribute the CNN across multiple devices.
