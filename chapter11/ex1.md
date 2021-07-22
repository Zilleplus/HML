# Is it ok to initialize all the weights to the same value as longs as that value is selected randomly using HE initialization.
No it is not, if all of the weights are on the same value. It will not converge to a usefull value. Neurons will often do the same thing, you need to randomization to get different behavior on the different neurons.
