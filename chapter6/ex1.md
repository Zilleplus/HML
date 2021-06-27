# What is the approximate depth of a Decision Tree trained (Without restrictions) on a training set with one million instances.

As there is not limit on the number of splits that can be made, the best tree in the worst case senario will keep splitting untill all training samples have their own node. This obviously is overfitting, but none the less predicts 100% time right on the training samples. The depth of this tree is about log(1e6) ~= 20.

-> Is ok according to Appendix A.
