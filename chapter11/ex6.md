# Name three way's you can produce a sparse model
- get rid of tiny weights (set them too zero), this degrades performance.
- Add l1 regularization.
- Use TensorFlow Model Optimization Toolkit (TF-MOT) to removed connections during training.
