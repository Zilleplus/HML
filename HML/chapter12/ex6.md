# Similarly a costum metric can be defined in a function or a subclas of kera.metrics.Metric. When would you use each option.
If it's a streaming metric, then you have state, and you'l need to class version. If it's stateless, as in it evaluates per batch, then you can use a function. Tensorflow will keep track of the average value.
