# What may happen if you set the momentum hyperparameter too close to 1 (e.g. 0.9999) when using an SGD optimizer.

my answer:
v^  = v^*momentum + v*(1-momentum)
The second term will become nearly zero so v^ ~= v^, and there will be nearly no normalization.

The appendix talks about the momentum as it's used int he momentum optimization. In which a too high momentum will hop around the optimimum, like you get with a too high learning rate. I find the question rather strange without more details.
