# Can a SVM classifier output a confidence score when it classifies an instance? What about probability?

The SVM classfier can output a per sample score, and depending on it's sign it's classified in a perticular class. The closer the score is to zero, the less confident the classifier is.

It can however not directly return a probability. The svm class in scikit does however offer such a feature, by using cross validation of the training data. It can estimate the probability for a certain score.
