# If you training set contains 100.000 instances, will setting presort=True speedup training?

No it will not, it will slow down the training. This only works on small training sets. If it were only 100 samples, this would be a great idea to speed up the process.
