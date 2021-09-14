# What is the benefit of out-of-bag evaluation?
The bagging classifier trains it's different sub-models with part of the training data.
This means that for each sub-model their exists training data it has never seen.
This data can be used to evaluate the classifier just as accurate as the test data.
So this eliminates the need for test data.
