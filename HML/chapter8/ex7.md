# How can you evaluate the performance of a dimensionality reduction algorithm on your dataset.
Try to recreate the original dataset from the reduced, and compare the different with the original dataset. The similar the 2 are the better the reduction. Also the more dimensions you can get rid of without taking a hit on accuracy the better.
If the reduction is used in combination with a machinelearning algorithm, you can also look at that algorithms performance when you change the dimension reduction.
