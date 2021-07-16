# Why is it generally preferable to use Logistic Regression classifier rather than a classical Perceptron (i.e. , a single layer of threshold logic units trained using the Percceptro training algorithm) ? How can you tweak a Perceptron to make it equivalent to a Logistic Regression Classifier.
Logistic regression:
* uses logistic function: LO(t) = 1 / (1 + exp(-t)
* The model outputs a probability P(x) = LO(x*w^t) with w=weights

Perceptron:
* uses step function: U(t) = 0 iff t smaller or equals then 0 otherwise 1
* The model outputs 0 or 1 -> h(x) = U(x*w^t) with w=weights

The logistic is preferred as it return probabilities which give you more usefull information especially if you have a multiclass problem. 

If the Unit function in the Perceptron is replaced by the logistic function, you get exactly the same models.
