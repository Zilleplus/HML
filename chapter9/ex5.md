# Can you name two clustering algorithms that can scale to large datasets? And two that look for regions of high density?
## large scale:
- k-means: very simple efficient algorithm, so scales really well.
- DBSCAN: m(log(m)) complexity, so also scales really well. 

## look for regions of high density:
- DBSCAN: inherently looks for high density areas
- Bayesian Gaussian mixture models: If the hyperparameter "concentration weight" is set to a low value. It will find high density area's.

The solution does not mention Bayesian Gaussina mixture models to find high density. But mentions "Mean shift look" algoritm instead. This algo is mentioned on page 259, but not discussed in detail.
