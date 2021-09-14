# Does it make any sense to chain two different dimensionality reduction algorithms together.
Yes, for example vanilla PCA can be used to reduce the number of dimensions very quickly before sending it to an LLE. This can potentially speed up the pipeline, vs only LLE.
