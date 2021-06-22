# Suppose you are using Ridge Regression and you notice that the training and validation error are almost equal and fairly high. Would you say the the model suffers from a high bias or high variance? Should you increase the regularizatoin hyperparameter or reduce it?

The traiing and validation error is about the same, it indicates a bias. It's unlikely (unless you a LOT of data) that this is overfitting. It's more likely that you are underfitting. You should decrease the regularization hyperparameter. (redge = poly + Tiknonov regularization)
