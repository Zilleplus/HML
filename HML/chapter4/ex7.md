# Which Gradient Descent algorihm (Among those discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make the other converge as well?

The Stochastic gradient descent will get the fastest in the vicinty of the optimal solution, but as with any monte-carlo like algorihm. As soon as you reach one digit of correctness, the convergence will slow down significantly. The mini-batch is the second fastests, and the batch-gradient is the slowest.

The stochastic Descent nor the mini-batch can converge too one point, there is no way to avoid this. Unless yo switch to the batch gradient at the end to converge it.
