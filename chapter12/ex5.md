# A custom loss function can be defined by writing a function or by subclassing the kera.losses.Loss. When would you use each option.
A simple function will suffice if you don't need any parameters. If you do need parameters, you'l need the config part. And you will have to use the class, as it has the "config" part to save away the value of these parameters.
