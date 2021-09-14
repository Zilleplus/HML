# Describe two techniques to select the right number of clusters when using K-Means.
- Intertia: Cluster the dataset with different number of clusters, find the interia parameter(square distance between sample and it's cluster centroid). Find the elbow in the curve, and that's about the optimal number of clusters.
- The inertia technique is rather inaccurate, a more complicated exact way is using the silhouette scores.
Silhouette_score: (b - a)/max(a, b)
a: Mean distance to the other instances in the same cluster.
b: Mean distance to the instances in the nearby cluster.

The higher this score the better. (you should see a hill like shape in the curve, the top is the optimum)
