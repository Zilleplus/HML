# Can custom Keras components contain arbitraty Python code, or must they be convertible to TF functions?
Everything that is part of the cost function must be convertible to tf, the metrics can however contain non convertable code.
