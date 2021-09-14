# Do you get the same result with tf.range(10) and tf.constant(np.arange(10))

Yes, the creation itself is slightly different, but the results are the same.

from solutions: tensorflow uses 32 bit ints, while numpy will default to 64 bit. So apparently it's not quiet the same.
