# What is backpropagation and how does it work? What is the difference between backpropagation and reverse-model autodiff.

This is a rather annoying question, as it differs from literature to literature. Some books take back prop as the whole optimization algo. Others define it as the reverse-model autodiff with an output of dim=1. Which means the Jacobian is a vector. (so we get gradient) In the end it doesn't really matter, it's just naming.
