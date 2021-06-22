# Suppose you are using Polynomial Regression. You plot the learning curves and you  notice that theire is a large gab between the training error and the validation error. What is happening? What are 3 way's to solve this?

A clear case of overfitting.

1. Reduce the dimension of the polynomial function.
2. Add regularisation.
3. Add more training data (The training and validation will be about the same, but the model is still pretty bad on training data itself.)
