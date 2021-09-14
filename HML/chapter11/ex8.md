# Report 
## Training simple network
The optimal learning rate seems to be e-4 which results in an accuracy of only 44%.

## Training with Batch normalization
With a learning rate of 1e-3 we get 52%, with Batch normalization we get less iterations. But every iteration takes longer, the total learn time seem just as fast.(didn't actually time it though)

## Training with Batch normalization and selu
With a learning rate of 1e-4 we can get to 55%, lowering the learning rate even more results in overfitting. We get accuracy of 64% on data, but only 55% on validatio.

## Training with Batch normalization, selu and dropout
dropoutrate=0.2 seem to work the best, we don't have the overfitting anymore. But it does take a very very long time to train this.

learning rate=??e-4???
