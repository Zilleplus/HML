import tensorflow.keras as keras # type: ignore

model = keras.Sequential([
    keras.layers.Dense(units=64, activation="relu"),
    keras.layers.Dense(units=10, activation="softmax")])

# keras specific:
# When using sequential model we don't need to say
# input and output layer, the build function does this
# when the first data is applied to the system.
# In order to see the weights we will call it manually here.
# -> batch size doesn't matter here...
model.build(input_shape=(None, 3))

# 3 inputs
# each layer is of the shape activation(a*x + b)

# First layer:
model.weights[0].shape  # [3, 64] weights -> a
model.weights[1].shape  # [64] weights -> b

# Second layer:
model.weights[2].shape  # [64, 10] -> a
model.weights[3].shape  # [10] -> b

model.summary()
