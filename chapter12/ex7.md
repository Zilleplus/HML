# When should you create a custom layer versus a custom model?
A costum layer is used when the architecture of your requested layer is not supported by keras. It's also convenient to merge several layers together in one costum layer. As it avoids some repetition in the code.

A costum model must be used when the model itself is not just a sequence of layers. But containts for example some kind of feedback/residual.
