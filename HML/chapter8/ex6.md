# In What cases would you use:
## Vanilla PCA
When the dataset is not too large, and higly linear in nature.

## Incremental PCA
When the entire dataset does not fit in memory, the incremental PCA will be faster.

## Randomized PCA
When the number of dimensions get large e.g. > 500, as it takes to long to do the svd in the direct way.

## Kernel PCA
With a highly non-linear dataset
