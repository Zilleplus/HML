# Suppose you use Batch Gradient Descent and you plot the validation error at every epoch. If you notice that the validation error conssistently goes up. What is likely going on? How can you fix this?

You are overfitting on you dataset, you can add regularization to the model. Which will reduce the overfitting as you put a cost on increasing the dimension. If you are setting the dimension by hand, obviously you can just reduce it. If the error on the training data starts to go down, you went too far and are underfitting.
